import numpy as np
import pytest

from features.corners import parse_corner_filename
from liberty.parser import parse_file, parse_text
from liberty.writer import fill_template
from models.phase2_scaling import fit_phase2_model, predict_corner, select_anchors
from scoring.audits import (
    DELAY_SCALING_BANDS,
    POWER_K_BAND,
    VIOLATION_RATE_THRESHOLD,
    check_delay_scaling_bands,
    check_power_k_band,
    run_cross_corner_inequality_audit,
    scaling_factor_quantiles,
)

from helpers import FULL_DIR, PARTIAL_DIR


@pytest.fixture(scope="module")
def full_libs():
    return {parse_corner_filename(p): parse_file(str(p)) for p in sorted(FULL_DIR.glob("*.lib"))}


@pytest.fixture(scope="module")
def phase2_model(full_libs):
    return fit_phase2_model(full_libs)


@pytest.fixture(scope="module")
def delivered(phase2_model, full_libs):
    """Reproduce scripts/phase2_predict.py's real delivery path (writer
    round-trip included) for all 10 partial corners, plus lookup tables
    keyed by corner name the way scripts/phase2_predict.py's audit step
    (added alongside this test) needs them."""
    predicted = {}
    anchor_of = {}
    for path in sorted(PARTIAL_DIR.glob("*.lib")):
        target_meta = parse_corner_filename(path)
        target_lib = parse_file(str(path))
        anchors = select_anchors(target_meta, full_libs)
        preds = predict_corner(phase2_model, target_lib, target_meta, anchors, full_libs)
        filled_text = fill_template(target_lib, preds)
        predicted[target_meta.name] = parse_text(filled_text)
        anchor_of[target_meta.name] = anchors[0].name
    truth = {m.name: lib for m, lib in full_libs.items()}
    anchor_libs_by_name = {m.name: lib for m, lib in full_libs.items()}
    return predicted, truth, anchor_of, anchor_libs_by_name


# ---------------------------------------------------------------------------
# cross-corner inequality audit
# ---------------------------------------------------------------------------


def test_cross_corner_inequality_audit_runs_all_three_checks(delivered):
    predicted, truth, _anchor_of, _anchor_libs = delivered
    results = run_cross_corner_inequality_audit(predicted, truth)
    names = {r.name for r in results}
    assert any("ss0p9v125c" in n for n in names)
    assert any("ss0p9vm40c" in n for n in names)
    assert any("tt1p0v25c" in n for n in names)
    assert any("tt0p8v25c" in n for n in names)
    assert len(results) == 4


def test_cross_corner_inequality_audit_passes_on_real_delivery(delivered):
    """docs/phase2_review.md item 3: violation rate must be <=1% for
    each of the three physical-ordering checks on the real, delivered
    predictions."""
    predicted, truth, _anchor_of, _anchor_libs = delivered
    results = run_cross_corner_inequality_audit(predicted, truth)
    for r in results:
        assert r.n_points > 0
        assert r.violation_rate <= VIOLATION_RATE_THRESHOLD, r.summary_line()
        assert r.passed


def test_inequality_result_violation_rate_and_passed_are_consistent():
    from scoring.audits import InequalityResult

    ok = InequalityResult(name="x", n_points=1000, n_violations=5)
    assert ok.violation_rate == pytest.approx(0.005)
    assert ok.passed

    bad = InequalityResult(name="x", n_points=1000, n_violations=50)
    assert bad.violation_rate == pytest.approx(0.05)
    assert not bad.passed


def test_compare_pointwise_direction_ge_flags_violations_correctly():
    """A synthetic case where the prediction is deliberately too small
    (violates '>=') must be caught, and a case that is high enough must
    not be."""
    from liberty.parser import LibertyFile, ValueTable
    from scoring.audits import _compare_pointwise

    key = ("CELLX", "Z", "timing", 0, "cell_rise")
    pred_values = np.array([[1.0, 2.0], [3.0, 0.5]])  # last point is too small
    threshold = np.array([[0.5, 1.0], [2.0, 1.0]])

    table = ValueTable(
        table_type="cell_rise", index_1=(), index_2=(), is_blank=False,
        values=pred_values, row_spans=[], key=key,
    )
    pred_lib = LibertyFile(
        path=None, text="", library_name="x", cells={},
        tables=[table], tables_by_key={key: table},
    )
    result = _compare_pointwise("test", pred_lib, {key: threshold}, direction=">=")
    assert result.n_points == 4
    assert result.n_violations == 1
    assert result.violation_rate == pytest.approx(0.25)
    assert not result.passed


# ---------------------------------------------------------------------------
# scaling-factor distribution report
# ---------------------------------------------------------------------------


def test_scaling_factor_quantiles_are_ordered_and_positive(delivered):
    predicted, _truth, anchor_of, anchor_libs = delivered
    for name in ("ss0p72v125c", "ff1p1v125c", "tt0p8v25c"):
        rows = scaling_factor_quantiles(name, predicted[name], anchor_libs[anchor_of[name]])
        assert rows, f"no quantile rows for {name}"
        for row in rows:
            assert row.n_points > 0
            assert row.p1 > 0
            assert row.p1 <= row.p50 <= row.p99


def test_delay_scaling_bands_pass_on_real_delivery(delivered):
    """docs/phase2_review.md item 3: ss0p72 delay scaling p50 in [1.1,
    2.0], ff1p1 delay scaling p50 in [0.7, 1.0). Recorded honestly --
    this test only asserts the pass, it must not be the thing tuned to
    force a pass (docs/phase2_review.md red line)."""
    predicted, _truth, anchor_of, anchor_libs = delivered
    checks = check_delay_scaling_bands(predicted, anchor_libs, anchor_of)
    assert {c.name.split()[0] for c in checks} == set(DELAY_SCALING_BANDS)
    for c in checks:
        assert c.passed, c.summary_line()


def test_power_k_band_check_reports_actual_fitted_value(phase2_model):
    checks = check_power_k_band(phase2_model.params)
    assert len(checks) == 2
    for c in checks:
        assert (c.lo, c.hi) == POWER_K_BAND
        assert np.isfinite(c.value)
        # Reported, not silently forced: the check's `passed` must agree
        # with a direct re-computation from POWER_K_BAND.
        lo, hi = POWER_K_BAND
        assert c.passed == (lo <= c.value <= hi)


def test_band_check_upper_exclusive_semantics():
    from scoring.audits import BandCheck

    exclusive = BandCheck(name="x", value=1.0, lo=0.7, hi=1.0, hi_inclusive=False)
    assert not exclusive.passed
    inclusive = BandCheck(name="x", value=1.0, lo=0.7, hi=1.0, hi_inclusive=True)
    assert inclusive.passed
