"""Unit tests for the 2026-08-10 robust-loss additions to
`models.phase4_mlp` (`loss`, `huber_delta`, `huber_delta_dev`,
`score_warmup_epochs`).

Written after the 2026-08-11 code review, which found this code path had
no test coverage at all: the reviewer's mutation list noted that
corrupting the Huber continuity or deleting the warmup-reset block would
not have been caught by anything. Each test below is the guard for one
of those mutations.
"""
import numpy as np
import pytest
import torch

from models.phase4_mlp import (
    HUBER_DELTA,
    LOSS_KINDS,
    SCORE_WARMUP_EPOCHS,
    _elementwise_loss,
    fit_mlp,
    predict_mlp,
)


def _loss(d, kind, delta=HUBER_DELTA):
    t = torch.tensor(np.atleast_1d(d), dtype=torch.float64)
    return _elementwise_loss(t, torch.zeros_like(t), kind, delta).numpy()


# --------------------------------------------------------------------------
# `mse` -- must stay exactly d**2, since the whole +0.0616 Huber measurement
# is a comparison against baselines produced by the pre-2026-08-10
# `nn.MSELoss()` code path.
# --------------------------------------------------------------------------


def test_mse_is_exactly_squared_residual():
    d = np.array([-5.0, -0.3, 0.0, 0.25, 3.0])
    assert np.allclose(_loss(d, "mse"), d ** 2, rtol=0, atol=0)


def test_mse_matches_torch_mseloss_reduction():
    rng = np.random.default_rng(0)
    pred = torch.tensor(rng.normal(size=500), dtype=torch.float64)
    target = torch.tensor(rng.normal(size=500), dtype=torch.float64)
    ours = torch.mean(_elementwise_loss(pred, target, "mse", HUBER_DELTA))
    assert torch.allclose(ours, torch.nn.MSELoss()(pred, target))


# --------------------------------------------------------------------------
# `huber` -- mutation guard: the quadratic arm must be exactly d**2 (not the
# textbook 0.5*d**2), and the two arms must meet with matching value AND
# derivative at |d| == delta.
# --------------------------------------------------------------------------


def test_huber_quadratic_arm_is_plain_squared_residual():
    d = np.array([-0.5, -0.1, 0.0, 0.2, 0.6])  # all strictly inside delta=log2
    assert np.all(np.abs(d) < HUBER_DELTA)
    assert np.allclose(_loss(d, "huber"), d ** 2)


def test_huber_is_continuous_at_the_knee():
    eps = 1e-9
    lo = _loss(HUBER_DELTA - eps, "huber")[0]
    hi = _loss(HUBER_DELTA + eps, "huber")[0]
    assert lo == pytest.approx(HUBER_DELTA ** 2, rel=1e-6)
    assert hi == pytest.approx(lo, rel=1e-6)


def test_huber_derivative_is_continuous_at_the_knee():
    def grad(x):
        t = torch.tensor([x], dtype=torch.float64, requires_grad=True)
        _elementwise_loss(t, torch.zeros(1, dtype=torch.float64), "huber", HUBER_DELTA).backward()
        return float(t.grad.item())

    eps = 1e-6
    assert grad(HUBER_DELTA - eps) == pytest.approx(2 * HUBER_DELTA, rel=1e-4)
    assert grad(HUBER_DELTA + eps) == pytest.approx(2 * HUBER_DELTA, rel=1e-4)


def test_huber_tail_gradient_is_bounded_unlike_mse():
    """The whole point: far-out residuals stop dominating the gradient."""
    def grad(x, kind):
        t = torch.tensor([x], dtype=torch.float64, requires_grad=True)
        _elementwise_loss(t, torch.zeros(1, dtype=torch.float64), kind, HUBER_DELTA).backward()
        return abs(float(t.grad.item()))

    assert grad(20.0, "huber") == pytest.approx(2 * HUBER_DELTA, rel=1e-6)
    assert grad(20.0, "mse") == pytest.approx(40.0, rel=1e-6)


def test_huber_respects_a_smaller_delta():
    delta = 0.2
    assert _loss(0.1, "huber", delta)[0] == pytest.approx(0.01)          # quadratic arm
    assert _loss(1.0, "huber", delta)[0] == pytest.approx(delta * (2 * 1.0 - delta))


# --------------------------------------------------------------------------
# `score` -- must equal the contest metric in log-ratio space exactly:
# e = min(1, |exp(d) - 1|), loss = e**2.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("d", [0.0, 0.2, 0.6931471805599453, 1.5, -0.5, -4.0, -20.0])
def test_score_loss_equals_the_capped_contest_metric(d):
    want = min(1.0, abs(np.expm1(d))) ** 2
    assert _loss(d, "score")[0] == pytest.approx(want, rel=1e-9, abs=1e-12)


def test_score_loss_saturates_and_never_overflows():
    out = _loss([50.0, 700.0], "score")
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 1.0)


def test_score_loss_gradient_vanishes_past_the_cap():
    t = torch.tensor([5.0], dtype=torch.float64, requires_grad=True)
    _elementwise_loss(t, torch.zeros(1, dtype=torch.float64), "score", HUBER_DELTA).backward()
    assert float(t.grad.item()) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# Argument validation
# --------------------------------------------------------------------------


def _tiny():
    # float32 throughout: mixing a float32 X with a float64 weight vector
    # makes numpy's matmul emit spurious overflow/divide-by-zero warnings
    # even though every value is finite.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6)).astype(np.float32)
    y = (X @ rng.normal(size=6).astype(np.float32)).astype(np.float32)
    assert np.all(np.isfinite(X)) and np.all(np.isfinite(y))
    return X[:150], y[:150], X[150:], y[150:]


def test_unknown_loss_is_rejected():
    Xf, yf, Xd, yd = _tiny()
    with pytest.raises(ValueError, match="unknown loss"):
        fit_mlp(Xf, yf, Xd, yd, width=4, n_blocks=1, max_epochs=1, loss="nope")


def test_unknown_loss_kind_is_rejected_at_the_elementwise_level():
    with pytest.raises(ValueError, match="unknown loss kind"):
        _loss(0.0, "nope")


def test_score_loss_rejects_a_patience_that_could_end_the_fit_during_warmup():
    """Otherwise fit_mlp silently returns an MSE-trained model."""
    Xf, yf, Xd, yd = _tiny()
    with pytest.raises(ValueError, match="score_warmup_epochs"):
        fit_mlp(Xf, yf, Xd, yd, width=4, n_blocks=1, max_epochs=50,
                patience=SCORE_WARMUP_EPOCHS, loss="score")
    with pytest.raises(ValueError, match="score_warmup_epochs"):
        fit_mlp(Xf, yf, Xd, yd, width=4, n_blocks=1, max_epochs=SCORE_WARMUP_EPOCHS,
                patience=50, loss="score")


def test_per_row_delta_requires_a_matching_dev_delta():
    Xf, yf, Xd, yd = _tiny()
    with pytest.raises(ValueError, match="requires huber_delta_dev"):
        fit_mlp(Xf, yf, Xd, yd, width=4, n_blocks=1, max_epochs=1, loss="huber",
                huber_delta=np.full(len(yf), 0.3, dtype=np.float32))


def test_per_row_delta_shape_is_validated():
    Xf, yf, Xd, yd = _tiny()
    with pytest.raises(ValueError, match=r"shape \(150,\)"):
        fit_mlp(Xf, yf, Xd, yd, width=4, n_blocks=1, max_epochs=1, loss="huber",
                huber_delta=np.full(5, 0.3, dtype=np.float32),
                huber_delta_dev=np.full(len(yd), 0.3, dtype=np.float32))


# --------------------------------------------------------------------------
# End-to-end: every loss kind trains, and a per-row delta is actually applied
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", LOSS_KINDS)
def test_every_loss_kind_fits_and_predicts(kind):
    Xf, yf, Xd, yd = _tiny()
    res = fit_mlp(Xf, yf, Xd, yd, width=8, n_blocks=1, max_epochs=8, patience=8, loss=kind)
    pred = predict_mlp(res, Xd)
    assert pred.shape == yd.shape
    assert np.all(np.isfinite(pred))
    assert len(res.dev_val_losses) >= 1


def test_a_per_row_delta_of_log2_matches_the_scalar_default():
    """Guards the delta_fit[idx] minibatch indexing: a constant per-row
    array must behave exactly like the equivalent scalar."""
    Xf, yf, Xd, yd = _tiny()
    kw = dict(width=8, n_blocks=1, max_epochs=6, patience=6, loss="huber", seed=7)
    scalar = predict_mlp(fit_mlp(Xf, yf, Xd, yd, **kw), Xd)
    per_row = predict_mlp(
        fit_mlp(Xf, yf, Xd, yd,
                huber_delta=np.full(len(yf), HUBER_DELTA, dtype=np.float32),
                huber_delta_dev=np.full(len(yd), HUBER_DELTA, dtype=np.float32),
                **kw),
        Xd,
    )
    assert np.allclose(scalar, per_row, atol=1e-5)


def test_a_huge_delta_degenerates_huber_into_mse():
    """The reviewer's mutation #1 in positive form: with delta far beyond
    any residual, Huber must reproduce the MSE arm exactly."""
    Xf, yf, Xd, yd = _tiny()
    kw = dict(width=8, n_blocks=1, max_epochs=6, patience=6, seed=11)
    as_mse = predict_mlp(fit_mlp(Xf, yf, Xd, yd, loss="mse", **kw), Xd)
    as_huber = predict_mlp(fit_mlp(Xf, yf, Xd, yd, loss="huber", huber_delta=1e6, **kw), Xd)
    assert np.allclose(as_mse, as_huber, atol=1e-5)
