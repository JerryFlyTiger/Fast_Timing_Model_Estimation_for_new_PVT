"""Phase 4 version B: a PyTorch MLP, one per delivery corner, on the same
plain-array contract as models.phase4_gbdt (see that module's docstring
for why fit/predict take raw numpy arrays rather than a Dataset object).

2026-07-27 improvement round (docs/phase4_results.md lever 4 / user
steering): the timing/power surfaces are highly nonlinear, and the
user explicitly asked to weight the MLP lever at least as heavily as the
feature-engineering lever. Changes from the original small
(128, 64, 32) plain-MLP version:

- **Residual/skip-connection architecture**: an input projection into a
  wide hidden dimension, followed by N pre-activation residual blocks
  (LayerNorm -> Linear -> ReLU -> Linear, added back to the block input),
  then a linear output head. Residual connections let a much deeper net
  (4-5 blocks, width 256-512) train without the vanishing-gradient /
  optimization-degradation problems a plain deep MLP hits.
- **LR schedule**: `ReduceLROnPlateau` on the dev-val loss (halve the LR
  after a few non-improving epochs) instead of a fixed LR for the whole
  run.
- **Larger early-stopping patience** (default 15 vs. the old 5): the
  previous run's best_epoch landed as early as 11-47 out of a 60-epoch
  cap with patience=5 -- consistent with stopping before the LR schedule
  (which didn't exist) had room to refine, not necessarily with genuine
  convergence. `max_epochs` raised to match.
- **Optional MPS acceleration** (Apple-silicon GPU backend): auto-used
  when available since the wider/deeper net is materially more FLOPs per
  epoch than the original one; falls back to plain CPU otherwise. Purely
  a speed optimization -- same math, same result modulo ordinary
  floating-point nondeterminism.

Feature standardization (`FeatureScaler`) is unchanged from the original
version (already present, so "加輸入標準化" from the steering note is
already satisfied structurally -- confirmed still wired in below).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
import torch
from torch import nn

HIDDEN_WIDTH = 256
N_RES_BLOCKS = 4
BLOCK_EXPANSION = 2  # each residual block's inner Linear widens by this factor
DROPOUT = 0.05
LEARNING_RATE = 2e-3
BATCH_SIZE = 8192
MAX_EPOCHS = 150
PATIENCE = 15
LR_PATIENCE = 4
LR_FACTOR = 0.5
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-5
RANDOM_STATE = 0
PREDICT_CHUNK = 200_000

# --- Robust / score-aligned training losses (2026-08-10) -------------------
# The contest metric is a *capped* relative error, e = min(1, |y-yhat|/|y|).
# With the `ratio` label the residual d = pred_log - y_log maps to it
# exactly (for sign-correct points) as e = min(1, |exp(d) - 1|), which
# saturates at d >= log 2 and asymptotes to 1 as d -> -inf. Plain MSE
# instead keeps growing without bound, so the ~0.085% of points that are
# sign flips -- which score 0 no matter what, and whose log-ratio labels
# have std 1.90 -- dominate the gradient. Down-weighting their influence
# is the untested mirror image of the (harmful, -3.57) 5x up-weighting
# experiment; see docs/recheck_20260809.md and the 2026-08-10 per-cell
# error regression (R^2=0.906 on flip share + near-zero share).
HUBER_DELTA = 0.6931471805599453  # log 2 == where over-prediction hits the metric's cap
SCORE_WARMUP_EPOCHS = 5  # "score" loss has vanishing tails; warm up on MSE first
_SCORE_EXP_CLAMP = 10.0  # expm1 overflow guard (metric is already capped well before this)
LOSS_KINDS = ("mse", "huber", "score")


def _elementwise_loss(
    pred: "torch.Tensor", target: "torch.Tensor", kind: str,
    delta: "float | torch.Tensor",
) -> "torch.Tensor":
    """Per-element loss on the label-space residual, same scale as `d**2`
    for small residuals so the existing LR / schedule stays calibrated."""
    d = pred - target
    if kind == "mse":
        return d ** 2
    if kind == "huber":
        ad = torch.abs(d)
        # 2x the textbook Huber so the quadratic arm is exactly d**2 and
        # the two arms meet with matching value and derivative at |d|=delta.
        return torch.where(ad <= delta, d ** 2, delta * (2.0 * ad - delta))
    if kind == "score":
        e = torch.abs(torch.expm1(torch.clamp(d, max=_SCORE_EXP_CLAMP)))
        return torch.clamp(e, max=1.0) ** 2
    raise ValueError(f"unknown loss kind {kind!r} (expected one of {LOSS_KINDS})")


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ResBlock(nn.Module):
    """Pre-activation residual block: LayerNorm -> Linear -> ReLU ->
    Dropout -> Linear, added back to the block's input. Keeps the
    identity path clean (no nonlinearity on the skip connection) so
    gradients flow straight through even with several blocks stacked."""

    def __init__(self, dim: int, expansion: int = BLOCK_EXPANSION, dropout: float = DROPOUT):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim * expansion, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


class Phase4MLP(nn.Module):
    """Input projection -> N residual blocks -> linear head. Default
    width 256 x 4 blocks (docs/plan.md 2026-07-27 steering: "4-5 層、每層
    256-512"); `hidden_sizes` kept as a constructor arg name for
    call-site compatibility but now means `(width, n_blocks)` when a
    2-tuple, or is ignored in favor of `width`/`n_blocks` kwargs -- see
    `fit_mlp`."""

    def __init__(
        self,
        n_features: int,
        width: int = HIDDEN_WIDTH,
        n_blocks: int = N_RES_BLOCKS,
        expansion: int = BLOCK_EXPANSION,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, width)
        self.input_act = nn.ReLU()
        self.blocks = nn.ModuleList([ResBlock(width, expansion, dropout) for _ in range(n_blocks)])
        self.out_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_act(self.input_proj(x))
        for block in self.blocks:
            h = block(h)
        h = self.out_norm(h)
        return self.head(h).squeeze(-1)


@dataclass
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


def fit_scaler(X: np.ndarray) -> FeatureScaler:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)  # guard constant columns
    return FeatureScaler(mean=mean.astype(np.float32), std=std.astype(np.float32))


@dataclass
class MLPFitResult:
    model: Phase4MLP
    scaler: FeatureScaler
    best_epoch: int
    device: torch.device
    # Per-epoch unweighted dev loss *in whichever loss form was active that
    # epoch*. For loss="score" that means the first `score_warmup_epochs`
    # entries are on the MSE scale (unbounded) and the rest on the capped
    # metric scale ([0,1]), with no marker at the switch -- do not plot this
    # as a single curve for a "score" fit without splitting it there.
    dev_val_losses: List[float] = field(default_factory=list)
    lrs: List[float] = field(default_factory=list)
    train_seconds: float = 0.0


def fit_mlp(
    X_fit_raw: np.ndarray,
    y_fit: np.ndarray,
    X_dev_raw: np.ndarray,
    y_dev: np.ndarray,
    *,
    width: int = HIDDEN_WIDTH,
    n_blocks: int = N_RES_BLOCKS,
    expansion: int = BLOCK_EXPANSION,
    dropout: float = DROPOUT,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    lr_patience: int = LR_PATIENCE,
    lr_factor: float = LR_FACTOR,
    min_lr: float = MIN_LR,
    weight_decay: float = WEIGHT_DECAY,
    seed: int = RANDOM_STATE,
    device: torch.device | None = None,
    sample_weight: np.ndarray | None = None,
    loss: str = "mse",
    huber_delta: "float | np.ndarray" = HUBER_DELTA,
    huber_delta_dev: "float | np.ndarray | None" = None,
    score_warmup_epochs: int = SCORE_WARMUP_EPOCHS,
) -> MLPFitResult:
    """`sample_weight` (2026-07-27 fall_power improvement round, optional,
    default None == uniform): per-fit-row training weight, aligned with
    `X_fit_raw`/`y_fit` (e.g. up-weighting near-zero fall_power rows --
    docs/phase4b_screen_log.md diagnosis). Only used in the training loss;
    the dev-val loss driving early stopping/LR-scheduling stays unweighted
    so it tracks the actual (unweighted) contest scorer, matching
    models.phase4_gbdt.fit_gbdt's identical convention.

    `loss` (2026-08-10, default "mse" == the historical behavior):
    "huber" clips the gradient of large label-space residuals at
    `huber_delta` (default log 2, where over-prediction saturates the
    contest metric's cap); "score" trains on the capped metric itself,
    min(1, |exp(d)-1|)**2, after `score_warmup_epochs` MSE epochs (its
    gradient vanishes in both tails, so it cannot bootstrap from a random
    init). The early-stopping / LR-schedule dev criterion uses the *same*
    loss form as training, unweighted -- each arm is selected on its own
    objective, and arms are compared on the real contest score computed
    downstream by the caller (scripts/phase4b_screen.py).

    `huber_delta` may be a scalar or a per-row array aligned with
    `X_fit_raw` (with `huber_delta_dev` then aligned with `X_dev_raw`).
    The 2026-08-10 screen found the optimal clip strength is
    table-type-dependent -- the four delay tables improve monotonically as
    delta shrinks (their large residuals are noise) while both power
    tables degrade (theirs carry signal) -- which a per-row delta captures
    inside one model, without the 6x data dilution of fitting one model
    per table type."""
    if loss not in LOSS_KINDS:
        raise ValueError(f"unknown loss {loss!r} (expected one of {LOSS_KINDS})")
    if loss == "score" and (patience <= score_warmup_epochs or max_epochs <= score_warmup_epochs):
        # Early stopping runs during warmup too, so a short patience (or a
        # max_epochs cap below the warmup length) can end the fit before the
        # objective ever switches -- silently returning an MSE-trained model
        # that the caller believes was trained on the capped metric.
        raise ValueError(
            f"loss='score' needs patience ({patience}) and max_epochs ({max_epochs}) "
            f"both > score_warmup_epochs ({score_warmup_epochs}), otherwise the fit can "
            f"stop before the objective switches"
        )
    t0 = time.time()
    torch.manual_seed(seed)
    device = device or _select_device()

    y_fit = y_fit.astype(np.float32)
    y_dev = y_dev.astype(np.float32)

    scaler = fit_scaler(X_fit_raw)
    X_fit = scaler.transform(X_fit_raw).astype(np.float32)
    X_dev = scaler.transform(X_dev_raw).astype(np.float32)

    model = Phase4MLP(X_fit.shape[1], width=width, n_blocks=n_blocks, expansion=expansion, dropout=dropout)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr
    )
    X_fit_t = torch.from_numpy(X_fit).to(device)
    y_fit_t = torch.from_numpy(y_fit).to(device)
    X_dev_t = torch.from_numpy(X_dev).to(device)
    y_dev_t = torch.from_numpy(y_dev).to(device)
    w_fit_t = None
    if sample_weight is not None:
        w_fit_t = torch.from_numpy(np.asarray(sample_weight, dtype=np.float32)).to(device)

    def _as_delta(v, n_expected, what):
        if np.isscalar(v):
            return float(v)
        arr = np.asarray(v, dtype=np.float32)
        if arr.shape != (n_expected,):
            raise ValueError(f"{what} must be a scalar or shape ({n_expected},), got {arr.shape}")
        return torch.from_numpy(arr).to(device)

    if not np.isscalar(huber_delta) and huber_delta_dev is None:
        raise ValueError("a per-row huber_delta requires huber_delta_dev aligned with X_dev_raw")
    delta_fit = _as_delta(huber_delta, X_fit.shape[0], "huber_delta")
    delta_dev = _as_delta(
        huber_delta if huber_delta_dev is None else huber_delta_dev,
        X_dev.shape[0], "huber_delta_dev",
    )

    n = X_fit_t.shape[0]
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0
    dev_val_losses: List[float] = []
    lrs: List[float] = []

    rng = np.random.default_rng(seed)
    active_kind = "mse" if loss == "score" else loss
    for epoch in range(1, max_epochs + 1):
        kind = "mse" if (loss == "score" and epoch <= score_warmup_epochs) else loss
        if kind != active_kind:
            # The objective just changed: dev losses recorded during warmup
            # are on a different scale, so restart the early-stopping and
            # LR-plateau bookkeeping rather than comparing across scales.
            active_kind = kind
            best_loss = float("inf")
            no_improve = 0
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr
            )

        model.train()
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = X_fit_t[idx]
            yb = y_fit_t[idx]
            opt.zero_grad()
            pred = model(xb)
            db = delta_fit if isinstance(delta_fit, float) else delta_fit[idx]
            per_element = _elementwise_loss(pred, yb, kind, db)
            if w_fit_t is None:
                loss_value = torch.mean(per_element)
            else:
                loss_value = torch.mean(w_fit_t[idx] * per_element)
            loss_value.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            dev_pred = model(X_dev_t)
            dev_loss = float(torch.mean(
                _elementwise_loss(dev_pred, y_dev_t, kind, delta_dev)
            ).item())
        dev_val_losses.append(dev_loss)
        scheduler.step(dev_loss)
        lrs.append(opt.param_groups[0]["lr"])

        if dev_loss < best_loss - 1e-6:
            best_loss = dev_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return MLPFitResult(
        model=model,
        scaler=scaler,
        best_epoch=best_epoch,
        device=device,
        dev_val_losses=dev_val_losses,
        lrs=lrs,
        train_seconds=time.time() - t0,
    )


def predict_mlp(fit_result: MLPFitResult, X: np.ndarray) -> np.ndarray:
    """Batched inference (avoid materializing a huge activation tensor for
    a large validation/inference set in one shot)."""
    fit_result.model.eval()
    device = fit_result.device
    X_scaled = fit_result.scaler.transform(X.astype(np.float32)).astype(np.float32)
    out = np.empty(X_scaled.shape[0], dtype=np.float64)
    with torch.no_grad():
        for start in range(0, X_scaled.shape[0], PREDICT_CHUNK):
            chunk = torch.from_numpy(X_scaled[start : start + PREDICT_CHUNK]).to(device)
            out[start : start + PREDICT_CHUNK] = fit_result.model(chunk).cpu().numpy()
    return out
