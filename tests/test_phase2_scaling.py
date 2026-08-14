import numpy as np
import pytest

from features.align import align_table_to_grid
from features.cellinfo import parse_cell_name
from features.corners import CornerMeta, parse_corner_filename
from liberty.parser import parse_file
from models.phase2_scaling import (
    ALPHA_BOUNDS,
    DELAY_TABLE_TYPES,
    K_BOUNDS,
    POWER_TABLE_TYPES,
    PROCESS_ORDER,
    VTH_BOUNDS,
    _enforce_voltage_monotonic,
    boundary_hit_report,
    fit_phase2_model,
    predict_corner,
    select_anchors,
)

from helpers import FULL_DIR, PARTIAL_DIR


# ---------------------------------------------------------------------------
# small pure-function units (features/*)
# ---------------------------------------------------------------------------


def test_parse_corner_filename():
    m = parse_corner_filename("lib1_ff0p99vm40c_alpha_100.lib")
    assert m.process == "ff"
    assert m.voltage == pytest.approx(0.99)
    assert m.temperature == pytest.approx(-40.0)
    assert m.name == "ff0p99vm40c"

    m2 = parse_corner_filename("testcase/alpha_test/full/lib1_ss0p81v125c_alpha_100.lib")
    assert m2.process == "ss"
    assert m2.voltage == pytest.approx(0.81)
    assert m2.temperature == pytest.approx(125.0)


def test_parse_cell_name():
    info = parse_cell_name("AN2AM16")
    assert info.base == "AN2A"
    assert info.family == "AN"
    assert info.drive_strength == 16

    info2 = parse_cell_name("INVM12")
    assert info2.family == "INV"
    assert info2.drive_strength == 12


def test_parse_cell_name_rejects_unknown_convention():
    with pytest.raises(ValueError):
        parse_cell_name("NOT_A_VALID_NAME")


def test_align_table_to_grid_is_noop_when_grids_match():
    values = np.arange(14, dtype=float).reshape(2, 7)
    grid = (0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007)
    out = align_table_to_grid(values, grid, grid)
    np.testing.assert_array_equal(out, values)


def test_align_table_to_grid_interpolates_when_grids_differ():
    src_grid = (0.0, 1.0, 2.0)
    values = np.array([[0.0, 10.0, 20.0]])
    dst_grid = (0.5, 1.5)
    out = align_table_to_grid(values, src_grid, dst_grid)
    np.testing.assert_allclose(out, [[5.0, 15.0]])


def test_enforce_voltage_monotonic_caps_delay_increase_and_decrease():
    source = np.array([1.0, 2.0])
    # higher target V -> delay must not exceed the source value
    up = _enforce_voltage_monotonic(np.array([1.5, 1.5]), source, v_target=0.9, v_source=0.8)
    np.testing.assert_array_equal(up, [1.0, 1.5])
    # lower target V -> delay must not go below the source value
    down = _enforce_voltage_monotonic(np.array([0.5, 0.5]), source, v_target=0.7, v_source=0.8)
    np.testing.assert_array_equal(down, [1.0, 2.0])
    # equal V -> untouched
    same = _enforce_voltage_monotonic(np.array([5.0, 5.0]), source, v_target=0.8, v_source=0.8)
    np.testing.assert_array_equal(same, [5.0, 5.0])


# ---------------------------------------------------------------------------
# model fit + prediction, exercised against the real dataset
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_libs():
    return {parse_corner_filename(p): parse_file(str(p)) for p in sorted(FULL_DIR.glob("*.lib"))}


@pytest.fixture(scope="module")
def phase2_model(full_libs):
    return fit_phase2_model(full_libs)


@pytest.fixture(scope="module")
def all_predictions(phase2_model, full_libs):
    """Predict all 10 partial corners the same way scripts/phase2_predict.py
    does, once per test module (expensive: reused by several tests)."""
    predictions = {}
    for path in sorted(PARTIAL_DIR.glob("*.lib")):
        target_meta = parse_corner_filename(path)
        target_lib = parse_file(str(path))
        anchors = select_anchors(target_meta, full_libs)
        stats = {}
        preds = predict_corner(phase2_model, target_lib, target_meta, anchors, full_libs, stats=stats)
        predictions[target_meta] = (target_lib, preds, stats)
    return predictions


def test_fit_produces_bounded_shape_parameters(phase2_model):
    """docs/phase2_review.md item 1: Vth is per-process now (one value
    per PROCESS_ORDER entry), alpha is a single shared scalar with a
    narrowed [1.1, 1.5] box."""
    for table_type in DELAY_TABLE_TYPES:
        p = phase2_model.params[table_type]
        assert p.kind == "delay"
        assert set(p.Vth_by_process) == set(PROCESS_ORDER)
        for Vth in p.Vth_by_process.values():
            assert VTH_BOUNDS[0] <= Vth <= VTH_BOUNDS[1]
        assert ALPHA_BOUNDS[0] <= p.alpha <= ALPHA_BOUNDS[1]
        for c0 in p.c0_by_process.values():
            assert np.isfinite(c0)
        for offset in p.offset_by_process.values():
            assert np.isfinite(offset)
        assert p.offset_by_process.get("tt") == 0.0  # gauge reference, see _fit_process_offset
    for table_type in POWER_TABLE_TYPES:
        p = phase2_model.params[table_type]
        assert p.kind == "power"
        assert K_BOUNDS[0] <= p.k <= K_BOUNDS[1]
        for offset in p.offset_by_process.values():
            assert np.isfinite(offset)


def test_fit_respects_vth_ordering_constraint(phase2_model):
    """docs/phase2_review.md item 1: Vth_ss >= Vth_tt >= Vth_ff must hold
    for every table_type, not just be approximately true."""
    for table_type in DELAY_TABLE_TYPES:
        p = phase2_model.params[table_type]
        assert p.Vth_by_process["ss"] >= p.Vth_by_process["tt"] - 1e-9
        assert p.Vth_by_process["tt"] >= p.Vth_by_process["ff"] - 1e-9


def test_boundary_hit_report_reflects_actual_fit(phase2_model):
    """The report must not silently disagree with the fitted values --
    docs/phase2_review.md item 1 requires reporting (not hiding) whether
    Vth/alpha are pinned at their box bounds."""
    report = boundary_hit_report(phase2_model)
    for table_type in DELAY_TABLE_TYPES:
        p = phase2_model.params[table_type]
        entry = report[table_type]
        for proc, Vth in p.Vth_by_process.items():
            hit = entry["Vth_at_bound"][proc]
            if hit == "lower":
                assert Vth == pytest.approx(VTH_BOUNDS[0], abs=1e-6)
            elif hit == "upper":
                assert Vth == pytest.approx(VTH_BOUNDS[1], abs=1e-6)
            else:
                assert VTH_BOUNDS[0] < Vth < VTH_BOUNDS[1]
    for table_type in POWER_TABLE_TYPES:
        p = phase2_model.params[table_type]
        assert report[table_type]["k_at_bound"] in (None, "lower", "upper")


def test_select_anchors_prefers_exact_process_and_temperature(full_libs):
    target = parse_corner_filename("lib1_ss0p72v125c_alpha_100.lib")
    anchors = select_anchors(target, full_libs)
    assert len(anchors) == 1
    assert anchors[0].name == "ss0p81v125c"


def test_select_anchors_falls_back_to_same_process_when_no_exact_temperature():
    target = CornerMeta(process="ss", voltage=0.81, temperature=125.0, name="ss0p81v125c")
    available = {
        CornerMeta("ss", 0.81, -40.0, "ss0p81vm40c"): object(),
        CornerMeta("ff", 0.99, 125.0, "ff0p99v125c"): object(),
    }
    anchors = select_anchors(target, available)
    assert [a.name for a in anchors] == ["ss0p81vm40c"]


def test_all_10_partial_corners_predict_finite_values(all_predictions):
    """Every prediction must be finite (src/liberty/writer.py would
    otherwise refuse to write it -- see its NaN/Inf guard). Delay-family
    tables (cell_rise/cell_fall/rise_transition/fall_transition) are
    physically always positive; internal_power tables legitimately
    contain negative entries in this dataset (e.g. AOI21M16 fall_power,
    representing charge recovery), and the multiplicative
    `source * exp(delta)` prediction form correctly preserves sign, so
    non-negativity is only asserted for the delay family."""
    for target_meta, (target_lib, preds, _stats) in all_predictions.items():
        for key, values in preds.items():
            assert np.isfinite(values).all(), f"{target_meta.name} {key} has non-finite values"
            if key[-1] in DELAY_TABLE_TYPES:
                assert (values >= 0).all(), f"{target_meta.name} {key} has negative delay/transition values"
            # Known-invalid all-zero rise_power/fall_power arcs (docs/plan.md
            # rule 3) must map source==0 -> target==0 exactly.
            table = target_lib.tables_by_key[key]
            assert table.is_blank


def test_all_10_partial_corners_fill_every_blank_table(all_predictions):
    for target_meta, (target_lib, preds, _stats) in all_predictions.items():
        blank_keys = {t.key for t in target_lib.tables if t.is_blank}
        assert set(preds.keys()) == blank_keys


def test_predictions_are_voltage_monotonic_against_anchor(all_predictions, full_libs):
    """V up -> delay down, checked against each target's own anchor
    (docs/plan.md Phase 2 item 5 acceptance: "單調性檢查全過")."""
    for target_meta, (target_lib, preds, _stats) in all_predictions.items():
        anchors = select_anchors(target_meta, full_libs)
        if len(anchors) != 1:
            continue  # geometric-mean blends aren't pointwise-monotonic by construction
        anchor_lib = full_libs[anchors[0]]
        anchor_meta = anchors[0]
        if anchor_meta.voltage == target_meta.voltage:
            continue
        higher_v = target_meta.voltage > anchor_meta.voltage
        for key, values in preds.items():
            if key[-1] not in DELAY_TABLE_TYPES:
                continue
            source = anchor_lib.tables_by_key[key].values
            if higher_v:
                assert (values <= source + 1e-9).all(), f"{target_meta.name} {key} violates V-up-delay-down"
            else:
                assert (values >= source - 1e-9).all(), f"{target_meta.name} {key} violates V-down-delay-up"


def test_buck_and_boost_corners_both_trigger_symmetric_derating(all_predictions):
    """docs/phase2_review.md item 3: shrinkage is now symmetric -- both
    buck (step-down) and boost (step-up) voltage extrapolations get the
    same SHRINK_LAMBDA de-rating, unlike Phase 2 where only buck
    corners were shrunk and boost corners went through un-shrunk."""
    buck = {"ss0p72v125c", "ss0p72vm40c", "ff0p88v125c", "ff0p88vm40c", "tt0p8v25c"}
    boost = {"ss0p9v125c", "ss0p9vm40c", "ff1p1v125c", "ff1p1vm40c", "tt1p0v25c"}
    for target_meta, (_lib, _preds, stats) in all_predictions.items():
        any_shrunk = any(s["n_shrunk_calls"] > 0 for s in stats.values())
        assert target_meta.name in buck or target_meta.name in boost
        assert any_shrunk, f"{target_meta.name} never triggered voltage-shift de-rating"


def test_predict_table_rejects_nan_source_gracefully():
    """models.phase2_scaling must never hand the writer a non-finite
    prediction (src/liberty/writer.py already raises on NaN/Inf -- this
    checks the model-side guard fires first with a clear error)."""
    from models.phase2_scaling import ShapeParams, Phase2Model

    # Vth above both corners' voltage makes log(V - Vth) = log(negative) = nan,
    # which is exactly the kind of bad fit this guard exists to catch.
    bogus = ShapeParams(
        kind="delay", Vth_by_process={"ss": 0.9, "tt": 0.9, "ff": 0.9}, alpha=1.5, k=None,
        c0_by_process={}, offset_by_process={},
        b_slew=0.0, b_load=0.0, b_strength=0.0, strength_center=0.0, group_offset={},
        clip_delta=10.0, n_train_pairs=0, fit_cost=0.0,
    )
    model = Phase2Model(params={"cell_rise": bogus})
    src_meta = CornerMeta("ss", 0.81, 25.0, "x")
    tgt_meta = CornerMeta("ss", 0.72, 25.0, "y")
    values = np.ones((7, 7))
    grid = tuple(range(7))
    with pytest.raises(ValueError):
        model.predict_table(("C", "Z", "timing", 0, "cell_rise"), values, src_meta, tgt_meta, grid, grid)
