"""Tests for models.phase4_features: the Phase 4 (official 400-cell
training set) dataset pipeline. Covers docs/plan.md Phase 4 item 7's
required guarantees: split determinism, zero leakage, finite output
values, and alpha-inference feature completeness (100 alpha cells x 5
anchor corners, no missing values).
"""

from __future__ import annotations

import numpy as np
import pytest

from features.corners import parse_corner_filename
from liberty.parser import parse_file
from models.phase4_features import (
    ALPHA_TOPOLOGY,
    ANCHOR_CORNER_NAMES,
    BETA_ANCHOR_NAMES,
    BETA_TARGET_NAMES,
    BETA_TOPOLOGY,
    DELIVERY_CORNER_NAMES,
    FEATURE_NAMES,
    FINAL_ANCHOR_NAMES,
    FINAL_TARGET_NAMES,
    FINAL_TOPOLOGY,
    NEAREST_ANCHOR_BY_TARGET,
    PHASE4_CELL_SPLIT_SEED,
    PHASE4_DEV_SPLIT_SEED,
    STAGE_TOPOLOGIES,
    XTABLE_COMPANION_TABLE_TYPES,
    build_arc_attr_index,
    build_base_dataset,
    build_family_vocab_for_phase4,
    build_power_to_timing_arc_map,
    build_xtable_features,
    extract_raw_values,
    infer_nearest_anchor_by_target,
    make_label,
    reconstruct_predictions,
    score_breakdown,
    split_cells,
    split_dev,
    trainable_mask,
    xtable_feature_names,
)
from paths import ALPHA_FULL_DIR, training_set_files

# ---------------------------------------------------------------------------
# Fixtures: parse the real training-set / alpha corners once per test
# session (the .tlib files are ~19MB each -- 15 files -- so this is the
# single expensive shared cost; individual tests reuse these fixtures).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def training_libs():
    libs = {}
    for path in training_set_files():
        meta = parse_corner_filename(str(path))
        libs[meta.name] = parse_file(str(path))
    return libs


@pytest.fixture(scope="module")
def anchor_libs(training_libs):
    return {name: training_libs[name] for name in ANCHOR_CORNER_NAMES}


@pytest.fixture(scope="module")
def all_training_cells(anchor_libs):
    return sorted(anchor_libs[ANCHOR_CORNER_NAMES[0]].cells)


@pytest.fixture(scope="module")
def alpha_full_libs():
    libs = {}
    for path in sorted(ALPHA_FULL_DIR.glob("*.lib")):
        meta = parse_corner_filename(str(path))
        libs[meta.name] = parse_file(str(path))
    return libs


@pytest.fixture(scope="module")
def all_alpha_cells(alpha_full_libs):
    return sorted(alpha_full_libs[ANCHOR_CORNER_NAMES[0]].cells)


# ---------------------------------------------------------------------------
# Corner topology sanity
# ---------------------------------------------------------------------------


def test_training_set_has_400_cells_and_matches_alpha_zero_overlap(all_training_cells, all_alpha_cells):
    assert len(all_training_cells) == 400
    assert len(all_alpha_cells) == 100
    assert set(all_training_cells) & set(all_alpha_cells) == set()


def test_anchor_and_delivery_corner_names_present_in_training_set(training_libs):
    assert set(ANCHOR_CORNER_NAMES) <= set(training_libs)
    assert set(DELIVERY_CORNER_NAMES) <= set(training_libs)
    assert len(ANCHOR_CORNER_NAMES) == 5
    assert len(DELIVERY_CORNER_NAMES) == 10


def test_nearest_anchor_matches_process_and_temperature():
    """Every delivery corner's assigned nearest anchor must share its
    process and temperature exactly (Delta_T == 0, pure voltage shift) --
    docs/plan.md's foundational assumption for this label definition."""
    for target_name, anchor_name in NEAREST_ANCHOR_BY_TARGET.items():
        t_meta = parse_corner_filename(f"lib1_{target_name}_alpha_100.lib")
        a_meta = parse_corner_filename(f"lib1_{anchor_name}_alpha_100.lib")
        assert t_meta.process == a_meta.process
        assert t_meta.temperature == a_meta.temperature
        assert t_meta.voltage != a_meta.voltage


# ---------------------------------------------------------------------------
# Split determinism / zero leakage (pure functions, no I/O)
# ---------------------------------------------------------------------------


def test_split_cells_is_deterministic_for_same_seed():
    cells = [f"CELL{i}" for i in range(400)]
    train_a, val_a = split_cells(cells, seed=PHASE4_CELL_SPLIT_SEED)
    train_b, val_b = split_cells(cells, seed=PHASE4_CELL_SPLIT_SEED)
    assert train_a == train_b
    assert val_a == val_b


def test_split_cells_400_gives_320_80_with_no_overlap():
    cells = [f"CELL{i}" for i in range(400)]
    train, val = split_cells(cells, seed=PHASE4_CELL_SPLIT_SEED)
    assert len(train) == 320
    assert len(val) == 80
    assert set(train) & set(val) == set()
    assert set(train) | set(val) == set(cells)


def test_dev_split_is_subset_of_train_and_disjoint_from_val():
    cells = [f"CELL{i}" for i in range(400)]
    train, val = split_cells(cells, seed=PHASE4_CELL_SPLIT_SEED)
    dev_train, dev_val = split_dev(train, seed=PHASE4_DEV_SPLIT_SEED)
    assert len(dev_train) == 256
    assert len(dev_val) == 64
    assert set(dev_train) <= set(train)
    assert set(dev_val) <= set(train)
    assert set(dev_train) & set(dev_val) == set()
    # the 80% held-out validation cells must never appear in either half
    # of the dev split -- this is the "early stopping never touches the
    # 80 validation cells" guarantee.
    assert not (set(dev_train) & set(val))
    assert not (set(dev_val) & set(val))


def test_real_400_cell_split_matches_recorded_sizes(all_training_cells):
    train, val = split_cells(all_training_cells, seed=PHASE4_CELL_SPLIT_SEED)
    assert len(train) == 320
    assert len(val) == 80
    assert set(train) | set(val) == set(all_training_cells)


# ---------------------------------------------------------------------------
# build_base_dataset: zero leakage + completeness on the real data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def split_and_datasets(anchor_libs, all_training_cells):
    train_cells, val_cells = split_cells(all_training_cells, seed=PHASE4_CELL_SPLIT_SEED)
    arc_attr_index = build_arc_attr_index(anchor_libs[ANCHOR_CORNER_NAMES[0]])
    family_vocab = build_family_vocab_for_phase4(all_training_cells)
    ds_train = build_base_dataset(anchor_libs, train_cells, arc_attr_index, family_vocab)
    ds_val = build_base_dataset(anchor_libs, val_cells, arc_attr_index, family_vocab)
    return train_cells, val_cells, ds_train, ds_val


def test_build_base_dataset_has_zero_cell_leakage(split_and_datasets):
    train_cells, val_cells, ds_train, ds_val = split_and_datasets
    assert not (set(ds_train.cell.tolist()) & set(val_cells))
    assert not (set(ds_val.cell.tolist()) & set(train_cells))
    assert set(ds_train.cell.tolist()) <= set(train_cells)
    assert set(ds_val.cell.tolist()) <= set(val_cells)


def test_build_base_dataset_feature_matrix_shape_matches_feature_names(split_and_datasets):
    _tc, _vc, ds_train, _ds_val = split_and_datasets
    assert ds_train.X.shape[1] == len(FEATURE_NAMES)
    assert ds_train.anchor_values.shape == (ds_train.n, len(ANCHOR_CORNER_NAMES))
    assert len(ds_train.keys) == ds_train.n
    assert np.isfinite(ds_train.X).all()
    assert np.isfinite(ds_train.anchor_values).all()


def test_build_base_dataset_rows_are_49_row_blocks_per_key(split_and_datasets):
    _tc, _vc, ds_train, _ds_val = split_and_datasets
    assert ds_train.n % 49 == 0


# ---------------------------------------------------------------------------
# label / reconstruction round trip + the zero-power rule
# ---------------------------------------------------------------------------


def test_make_label_and_reconstruct_roundtrip_on_nonzero_values():
    rng = np.random.default_rng(0)
    nearest = rng.uniform(1e-6, 10.0, size=1000)
    target = rng.uniform(1e-6, 10.0, size=1000)
    label = make_label(nearest, target)
    recon = reconstruct_predictions(nearest, label)
    np.testing.assert_allclose(recon, target, rtol=1e-6)


def test_reconstruct_predictions_zero_anchor_forces_zero_regardless_of_prediction():
    nearest = np.array([0.0, 0.0, 1.0, 2.0])
    y_pred = np.array([5.0, -100.0, 0.5, -0.5])
    recon = reconstruct_predictions(nearest, y_pred)
    assert recon[0] == 0.0
    assert recon[1] == 0.0
    assert recon[2] != 0.0
    assert recon[3] != 0.0


def test_reconstruct_predictions_always_finite_even_for_extreme_model_output():
    nearest = np.array([1.0, 1e-8, 1e8])
    y_pred = np.array([1e6, -1e6, 1e9])  # deliberately extreme
    recon = reconstruct_predictions(nearest, y_pred)
    assert np.isfinite(recon).all()


def test_trainable_mask_excludes_zero_anchor_rows_only():
    nearest = np.array([0.0, 1.0, -2.0, 0.0, 3.0])
    mask = trainable_mask(nearest)
    np.testing.assert_array_equal(mask, [False, True, True, False, True])


# ---------------------------------------------------------------------------
# real all-zero-power-table consistency (docs/plan.md rule 3), checked
# against the actual training set: a key is either all-zero at *every*
# training corner or all-zero at *none* of them.
# ---------------------------------------------------------------------------


def test_zero_power_pattern_is_consistent_across_all_training_corners(training_libs):
    power_keys = set()
    for lib in training_libs.values():
        for t in lib.tables:
            if t.table_type in ("rise_power", "fall_power"):
                power_keys.add(t.key)
    assert power_keys, "expected at least one power table key"

    checked = 0
    for key in power_keys:
        flags = []
        for lib in training_libs.values():
            table = lib.tables_by_key.get(key)
            if table is None:
                continue
            flags.append(bool((table.values == 0).all()))
        checked += 1
        assert len(set(flags)) == 1, f"key {key!r} has inconsistent all-zero flag across corners: {flags}"
    assert checked == len(power_keys)


# ---------------------------------------------------------------------------
# alpha inference feature completeness (docs/plan.md Phase 4 item 7):
# 100 alpha cells x 5 anchor corners must produce a fully populated
# feature matrix with no missing anchor values -- exactly the input shape
# available at real delivery-inference time.
# ---------------------------------------------------------------------------


def test_alpha_inference_feature_matrix_has_no_missing_anchor_values(alpha_full_libs, all_alpha_cells):
    arc_attr_index = build_arc_attr_index(alpha_full_libs[ANCHOR_CORNER_NAMES[0]])
    family_vocab = build_family_vocab_for_phase4(all_alpha_cells)
    ds_alpha = build_base_dataset(alpha_full_libs, all_alpha_cells, arc_attr_index, family_vocab)

    # every alpha cell must contribute at least one row (no cell dropped
    # entirely because one of its keys was missing an anchor value).
    assert set(ds_alpha.cell.tolist()) == set(all_alpha_cells)
    assert np.isfinite(ds_alpha.X).all()
    assert np.isfinite(ds_alpha.anchor_values).all()

    # cross-check against the known total table-key count for the alpha
    # 100-cell set (docs/plan.md / CLAUDE.md: 5804 keys shared across all
    # 15 alpha corners).
    n_keys = len(alpha_full_libs[ANCHOR_CORNER_NAMES[0]].tables_by_key)
    assert ds_alpha.n == n_keys * 49


def test_extract_raw_values_matches_reconstructed_key_blocks(alpha_full_libs, all_alpha_cells):
    """extract_raw_values must return values in exactly the row order
    implied by ds.keys (contiguous 49-row blocks per key) -- sanity-check
    against a lib whose values equal its own anchor column."""
    arc_attr_index = build_arc_attr_index(alpha_full_libs[ANCHOR_CORNER_NAMES[0]])
    family_vocab = build_family_vocab_for_phase4(all_alpha_cells)
    ds_alpha = build_base_dataset(alpha_full_libs, all_alpha_cells, arc_attr_index, family_vocab)

    # Use one of the anchor corners itself as the "target" lib: its
    # extracted values must equal that anchor's own column exactly.
    probe_name = ANCHOR_CORNER_NAMES[0]
    probe_lib = alpha_full_libs[probe_name]
    extracted = extract_raw_values(probe_lib, ds_alpha.keys)
    col = ANCHOR_CORNER_NAMES.index(probe_name)
    np.testing.assert_allclose(extracted, ds_alpha.anchor_values[:, col])


# ---------------------------------------------------------------------------
# Beta/final-stage corner topology (2026-07-27 addition, docs/plan.md
# improvement round "beta 階段模擬"): a different PVT triplet may be the
# "known" (anchor) set at later contest stages. These tests cover (a) the
# topology definitions themselves and (b) the single easiest place a
# stage bug could creep in -- silently computing a response-signature
# feature (or a nearest-anchor lookup) against an alpha-stage corner name
# while nominally running in beta/final mode.
# ---------------------------------------------------------------------------


def test_stage_topologies_are_disjoint_anchor_target_5_and_10():
    for topo in STAGE_TOPOLOGIES.values():
        assert len(topo.anchor_names) == 5
        assert len(topo.target_names) == 10
        assert not (set(topo.anchor_names) & set(topo.target_names))


def test_beta_topology_matches_coordinator_spec():
    """Beta anchors = the 5 boost corners; targets = 5 standard-voltage +
    5 buck corners. Every real corner name appears in exactly one of
    (alpha anchors, alpha targets) = (beta anchors + beta targets) since
    beta simply swaps which triplet is "known"."""
    assert set(BETA_ANCHOR_NAMES) == {"ss0p9v125c", "ss0p9vm40c", "ff1p1v125c", "ff1p1vm40c", "tt1p0v25c"}
    assert set(BETA_TARGET_NAMES) == {
        "ss0p81v125c", "ss0p81vm40c", "ff0p99v125c", "ff0p99vm40c", "tt0p9v25c",
        "ss0p72v125c", "ss0p72vm40c", "ff0p88v125c", "ff0p88vm40c", "tt0p8v25c",
    }
    # every real corner (alpha anchors + alpha targets) appears exactly
    # once across beta's anchor+target sets
    all_alpha_corners = set(ANCHOR_CORNER_NAMES) | set(DELIVERY_CORNER_NAMES)
    assert set(BETA_ANCHOR_NAMES) | set(BETA_TARGET_NAMES) == all_alpha_corners


def test_beta_buck_targets_map_to_the_boost_anchor_two_steps_away():
    """The buck targets (ss0p72*, ff0p88*, tt0p8v25c) are two voltage
    steps from their nearest anchor in beta mode (boost -> nominal ->
    buck), but the same-process-same-temperature rule still resolves to
    the boost anchor (there is no nominal anchor available in beta) --
    docs/plan.md improvement-round coordinator note, explicit spec."""
    assert BETA_TOPOLOGY.nearest_anchor_by_target["ss0p72v125c"] == "ss0p9v125c"
    assert BETA_TOPOLOGY.nearest_anchor_by_target["ss0p72vm40c"] == "ss0p9vm40c"
    assert BETA_TOPOLOGY.nearest_anchor_by_target["ff0p88v125c"] == "ff1p1v125c"
    assert BETA_TOPOLOGY.nearest_anchor_by_target["ff0p88vm40c"] == "ff1p1vm40c"
    assert BETA_TOPOLOGY.nearest_anchor_by_target["tt0p8v25c"] == "tt1p0v25c"
    # and the standard-voltage targets map to the same-process-temp boost anchor too
    assert BETA_TOPOLOGY.nearest_anchor_by_target["ss0p81v125c"] == "ss0p9v125c"
    assert BETA_TOPOLOGY.nearest_anchor_by_target["ff0p99vm40c"] == "ff1p1vm40c"
    assert BETA_TOPOLOGY.nearest_anchor_by_target["tt0p9v25c"] == "tt1p0v25c"


def test_final_topology_matches_coordinator_spec():
    assert set(FINAL_ANCHOR_NAMES) == {"ss0p72v125c", "ss0p72vm40c", "ff0p88v125c", "ff0p88vm40c", "tt0p8v25c"}
    all_alpha_corners = set(ANCHOR_CORNER_NAMES) | set(DELIVERY_CORNER_NAMES)
    assert set(FINAL_ANCHOR_NAMES) | set(FINAL_TARGET_NAMES) == all_alpha_corners


def test_infer_nearest_anchor_by_target_reproduces_hand_written_alpha_mapping():
    assert infer_nearest_anchor_by_target(ANCHOR_CORNER_NAMES, DELIVERY_CORNER_NAMES) == NEAREST_ANCHOR_BY_TARGET


# --- the critical zero-leakage guarantee: build_base_dataset in beta mode
# must NEVER read any of the 10 beta target corners' values. -----------------


@pytest.fixture(scope="module")
def beta_anchor_libs_only(training_libs):
    """A dict containing ONLY the 5 beta anchor libs -- structurally
    incapable of supplying a beta target corner's values. If
    build_base_dataset ever tried to look up a target corner (e.g. by
    accidentally falling back to the alpha-stage ANCHOR_CORNER_NAMES
    global instead of the `anchor_names` parameter), this dict would
    raise KeyError immediately rather than silently succeeding."""
    return {name: training_libs[name] for name in BETA_ANCHOR_NAMES}


def test_build_base_dataset_beta_mode_never_reads_target_corner_values(
    beta_anchor_libs_only, all_training_cells
):
    # Sanity: confirm the restricted dict really is missing every target
    # corner (i.e. this test would actually catch a leak, not vacuously pass).
    assert not (set(beta_anchor_libs_only) & set(BETA_TARGET_NAMES))
    assert set(beta_anchor_libs_only) == set(BETA_ANCHOR_NAMES)

    train_cells, _val_cells = split_cells(all_training_cells, seed=PHASE4_CELL_SPLIT_SEED)
    arc_attr_index = build_arc_attr_index(beta_anchor_libs_only[BETA_ANCHOR_NAMES[0]])
    family_vocab = build_family_vocab_for_phase4(all_training_cells)

    # If this ever tried to read a target-corner value it would KeyError
    # (the dict structurally cannot answer that lookup) -- reaching this
    # line at all is the leak-proof.
    ds_beta = build_base_dataset(
        beta_anchor_libs_only, train_cells, arc_attr_index, family_vocab, anchor_names=BETA_ANCHOR_NAMES
    )
    assert ds_beta.anchor_names == BETA_ANCHOR_NAMES
    assert ds_beta.anchor_values.shape == (ds_beta.n, len(BETA_ANCHOR_NAMES))
    assert np.isfinite(ds_beta.X).all()
    assert np.isfinite(ds_beta.anchor_values).all()


def test_beta_mode_nearest_anchor_and_sensitivity_features_use_beta_roles_not_alpha(
    beta_anchor_libs_only, all_training_cells
):
    """The response-signature features (lever 1) must resolve their
    ff_hot/ff_cold/ss_hot/ss_cold/tt_mid roles from the BETA anchor set,
    not silently fall back to alpha's ff0p99v125c-shaped indices. Cross-
    check: recompute the ff-temperature-sensitivity feature
    (log_ratio_ff_hot_cold, column index 5 of NUMERIC_FEATURE_NAMES,
    right after the 5 log_anchor_* columns) directly from
    ds_beta.anchor_values using BETA's own ff_hot/ff_cold columns, and
    confirm it matches the feature matrix exactly -- if the code had
    accidentally used alpha's role indices (which don't even exist in a
    5-column beta anchor_values array in the same order), this would
    either crash or silently mismatch."""
    train_cells, _val_cells = split_cells(all_training_cells, seed=PHASE4_CELL_SPLIT_SEED)
    arc_attr_index = build_arc_attr_index(beta_anchor_libs_only[BETA_ANCHOR_NAMES[0]])
    family_vocab = build_family_vocab_for_phase4(all_training_cells)
    ds_beta = build_base_dataset(
        beta_anchor_libs_only, train_cells, arc_attr_index, family_vocab, anchor_names=BETA_ANCHOR_NAMES
    )

    ff_hot_col = BETA_ANCHOR_NAMES.index("ff1p1v125c")
    ff_cold_col = BETA_ANCHOR_NAMES.index("ff1p1vm40c")
    log_anchor = np.log(np.abs(ds_beta.anchor_values) + 1e-30)
    expected_ff_sens = (log_anchor[:, ff_hot_col] - log_anchor[:, ff_cold_col]).astype(np.float32)

    # NUMERIC_FEATURE_NAMES layout: 5 log_anchor_* then log_ratio_ff_hot_cold
    ff_sens_feature_col = len(BETA_ANCHOR_NAMES)  # index 5
    np.testing.assert_allclose(ds_beta.X[:, ff_sens_feature_col], expected_ff_sens, atol=1e-4)

    # nearest_anchor() must also resolve against ds_beta's own anchor_names,
    # using BETA_TOPOLOGY's mapping (never the alpha NEAREST_ANCHOR_BY_TARGET).
    nearest = ds_beta.nearest_anchor("ss0p72v125c", BETA_TOPOLOGY.nearest_anchor_by_target)
    expected_col = BETA_ANCHOR_NAMES.index("ss0p9v125c")
    np.testing.assert_allclose(nearest, ds_beta.anchor_values[:, expected_col])


# ---------------------------------------------------------------------------
# Cross-table (power<->delay) features (2026-07-29 addition, docs/plan.md
# improvement round "跨表格特徵"). Covers: arc-alignment correctness (not
# just position-based arc_index matching), the fallback path for
# unmatched arcs, and -- the critical guarantee -- that
# build_xtable_features only ever reads from `anchor_libs` (never a
# target-corner lib), reusing the same `anchor_libs` fixture (restricted
# to the 5 alpha ANCHOR_CORNER_NAMES) the zero-leakage tests above rely on.
# ---------------------------------------------------------------------------


def test_build_power_to_timing_arc_map_matches_by_related_pin_not_position(anchor_libs):
    """Arc structure is shared across every corner (same rationale as
    build_arc_attr_index), so this only needs one anchor lib. Sanity: the
    map must be internally consistent -- every mapped (power -> timing)
    pair shares the same cell/pin, and matched + unmatched arcs account
    for every internal_power arc in the lib."""
    lib = anchor_libs[ANCHOR_CORNER_NAMES[0]]
    mapping, n_matched, n_unmatched = build_power_to_timing_arc_map(lib)

    assert n_matched > 0
    assert n_matched == len(mapping)

    n_power_arcs = 0
    for cell in lib.cells.values():
        for pin in cell.pins.values():
            n_power_arcs += sum(1 for a in pin.arcs if a.group_type == "internal_power")
    assert n_matched + n_unmatched == n_power_arcs

    for power_prefix, timing_prefix in mapping.items():
        p_cell, p_pin, p_group, _p_idx = power_prefix
        t_cell, t_pin, t_group, _t_idx = timing_prefix
        assert p_group == "internal_power"
        assert t_group == "timing"
        assert p_cell == t_cell
        assert p_pin == t_pin

    # related_pin correctness spot-check: for every mapped pair, the
    # matched timing arc's related_pin must equal the power arc's
    # related_pin (the actual matching criterion -- NOT arc_index, which
    # is independently assigned per group_type and need not line up).
    checked = 0
    for cell in lib.cells.values():
        for pin in cell.pins.values():
            by_index = {(a.group_type, a.arc_index): a for a in pin.arcs}
            for a in pin.arcs:
                if a.group_type != "internal_power":
                    continue
                prefix = (cell.name, pin.name, "internal_power", a.arc_index)
                if prefix not in mapping:
                    continue
                _tc, _tp, _tg, t_idx = mapping[prefix]
                matched_timing_arc = by_index[("timing", t_idx)]
                assert matched_timing_arc.related_pin == a.related_pin
                checked += 1
    assert checked == n_matched


def test_build_xtable_features_only_populates_power_rows_and_is_always_finite(anchor_libs, all_training_cells):
    """Structural zero-leakage guarantee: build_xtable_features only ever
    reads `anchor_libs` (here restricted to the 5 alpha anchors, via the
    same fixture the beta zero-leakage test uses) -- it cannot read a
    target-corner value regardless of feature_mode. Also checks the
    'delay 表預測不變' contract: non-power rows get all-zero xtable
    columns and xtable_has_match == 0."""
    small_cells = all_training_cells[:15]
    arc_attr_index = build_arc_attr_index(anchor_libs[ANCHOR_CORNER_NAMES[0]])
    family_vocab = build_family_vocab_for_phase4(all_training_cells)
    ds = build_base_dataset(anchor_libs, small_cells, arc_attr_index, family_vocab)

    mapping, n_matched_arcs, n_unmatched_arcs = build_power_to_timing_arc_map(anchor_libs[ANCHOR_CORNER_NAMES[0]])
    extra, names, n_matched_rows, n_fallback_rows = build_xtable_features(
        ds, anchor_libs, ANCHOR_CORNER_NAMES, mapping
    )

    assert names == xtable_feature_names(ANCHOR_CORNER_NAMES)
    assert extra.shape == (ds.n, len(names))
    assert np.isfinite(extra).all()
    assert n_matched_rows + n_fallback_rows == ds.n
    assert n_matched_rows % 49 == 0 and n_fallback_rows % 49 == 0

    has_match = extra[:, -1]
    assert set(np.unique(has_match).tolist()) <= {0.0, 1.0}

    power_mask = np.isin(ds.table_type, list(XTABLE_COMPANION_TABLE_TYPES))
    # every row with has_match==1 must be a power-table row
    assert np.all(power_mask[has_match == 1.0])
    # every non-power row has has_match==0 AND every xtable column exactly 0
    non_power_rows = ~power_mask
    assert np.all(has_match[non_power_rows] == 0.0)
    assert np.all(extra[non_power_rows, :] == 0.0)
    # fallback (unmatched-arc) power rows also get exactly 0, never NaN/garbage
    fallback_power_rows = power_mask & (has_match == 0.0)
    if fallback_power_rows.any():
        assert np.all(extra[fallback_power_rows, :] == 0.0)


def test_xtable_features_use_companion_table_not_the_power_table_itself(anchor_libs, all_training_cells):
    """Cross-check one matched fall_power row's xtable columns against a
    hand-computed value from the companion cell_fall table -- proves the
    function reads the DELAY table's values (not accidentally echoing the
    power table's own anchor values, which would defeat the point of a
    cross-table feature)."""
    small_cells = all_training_cells[:15]
    arc_attr_index = build_arc_attr_index(anchor_libs[ANCHOR_CORNER_NAMES[0]])
    family_vocab = build_family_vocab_for_phase4(all_training_cells)
    ds = build_base_dataset(anchor_libs, small_cells, arc_attr_index, family_vocab)
    mapping, _m, _u = build_power_to_timing_arc_map(anchor_libs[ANCHOR_CORNER_NAMES[0]])
    extra, names, _mr, _fr = build_xtable_features(ds, anchor_libs, ANCHOR_CORNER_NAMES, mapping)

    # find a matched fall_power key block
    n_keys = len(ds.keys) // 49
    found = False
    for k in range(n_keys):
        i = k * 49
        key = ds.keys[i]
        if key[-1] != "fall_power" or extra[i, -1] != 1.0:
            continue
        cell_name, pin_name, group_type, arc_index, _tt = key
        timing_prefix = mapping[(cell_name, pin_name, group_type, arc_index)]
        cell_fall_key = timing_prefix + ("cell_fall",)
        probe_lib = anchor_libs[ANCHOR_CORNER_NAMES[0]]
        cell_fall_table = probe_lib.tables_by_key[cell_fall_key]

        expected_log_anchor0 = np.log(np.abs(cell_fall_table.values.ravel()[0]) + 1e-30)
        # first xtable block ("delay1" = cell_fall for fall_power) first
        # column (anchor 0's log value) must equal the companion table's
        # own value at grid point 0 -- and must differ from the power
        # table's own log_anchor (base feature, first 5 columns of ds.X).
        actual = extra[i, 0]
        np.testing.assert_allclose(actual, expected_log_anchor0, atol=1e-4)
        found = True
        break
    assert found, "no matched fall_power row found in this cell subset -- test fixture needs a larger slice"
