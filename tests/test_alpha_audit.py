"""Unit tests for scoring.audits section 3 (alpha-composition reweighted
score audit, 2026-08-09 direction F)."""

import numpy as np
import pytest

from scoring.audits import (
    SUBGROUP_BULK_FP,
    SUBGROUP_FLIP,
    SUBGROUP_NEAR_ZERO,
    SUBGROUP_OTHER,
    assign_subgroups,
    reweighted_pooled_score,
    subgroup_stats,
)
from scoring.scorer import score_from_errors


def test_assign_subgroups_labels():
    y_true = np.array([1.0, -2e-5, 5e-5, 0.5, 3e-5])
    anchor = np.array([1.0, 3e-5, 8e-5, 0.4, 2e-5])
    table_type = np.array(
        ["fall_power", "fall_power", "fall_power", "cell_rise", "cell_rise"]
    )
    got = assign_subgroups(y_true, anchor, table_type)
    # fall_power, |y| >= 1e-4, sign-consistent -> bulk
    assert got[0] == SUBGROUP_BULK_FP
    # fall_power, sign differs from anchor -> flip (even though near zero)
    assert got[1] == SUBGROUP_FLIP
    # fall_power, tiny and sign-consistent -> near_zero
    assert got[2] == SUBGROUP_NEAR_ZERO
    # non-fall_power is always "other", regardless of magnitude
    assert got[3] == SUBGROUP_OTHER
    assert got[4] == SUBGROUP_OTHER


def test_reweighted_score_identity_when_multipliers_are_one():
    rng = np.random.default_rng(0)
    errs = rng.uniform(0, 1, size=1000)
    groups = np.where(rng.uniform(size=1000) < 0.3, "a", "b").astype("<U8")
    assert reweighted_pooled_score(errs, groups, {}) == pytest.approx(
        score_from_errors(errs), abs=1e-9
    )


def test_reweighted_score_zero_multiplier_equals_exclusion():
    errs = np.array([0.1, 0.2, 0.9, 1.0, 0.05])
    groups = np.array(["keep", "keep", "drop", "drop", "keep"])
    got = reweighted_pooled_score(errs, groups, {"drop": 0.0})
    want = score_from_errors(errs[groups == "keep"])
    assert got == pytest.approx(want, abs=1e-9)


def test_reweighted_score_halving_a_group_matches_hand_computation():
    errs = np.array([1.0, 1.0, 0.0, 0.0])  # group g: both fail; group h: both perfect
    groups = np.array(["g", "g", "h", "h"])
    # w_g=0.5 e2_g=1, w_h=0.5 e2_h=0; halve g: (0.25*1)/(0.25+0.5)=1/3
    got = reweighted_pooled_score(errs, groups, {"g": 0.5})
    want = 100.0 - 100.0 * np.sqrt(1.0 / 3.0)
    assert got == pytest.approx(want, abs=1e-9)


def test_subgroup_stats_mass_sums_to_total():
    rng = np.random.default_rng(1)
    errs = rng.uniform(0, 1, size=500)
    groups = np.where(rng.uniform(size=500) < 0.5, "a", "b").astype("<U8")
    stats = subgroup_stats(errs, groups)
    total_e2 = float(np.mean(errs**2))
    assert sum(s.e2_mass for s in stats) == pytest.approx(total_e2, rel=1e-12)
    assert sum(s.share for s in stats) == pytest.approx(1.0, rel=1e-12)
    assert sum(s.n_points for s in stats) == 500


# ---------------------------------------------------------------------------
# Per-cell composition helpers (2026-08-11, drive-matched audit -- these
# let prevalence be measured on a cell subset, which is what makes the
# drive-bucket rate table possible).
# ---------------------------------------------------------------------------


def _fake_lib(tables_by_cell):
    """Minimal LibertyFile stand-in: {cell: [(table_type, 7x7 array), ...]}."""
    from liberty.parser import LibertyFile, ValueTable

    tables_by_key = {}
    for cell, entries in tables_by_cell.items():
        for i, (table_type, values) in enumerate(entries):
            key = (cell, "Z", "internal_power", i, table_type)
            tables_by_key[key] = ValueTable(
                table_type=table_type, index_1=(), index_2=(), is_blank=False,
                values=np.asarray(values, dtype=float), row_spans=[], key=key,
            )
    return LibertyFile(path=None, text="", library_name="fake", cells={c: None for c in tables_by_cell},
                       tables=list(tables_by_key.values()), tables_by_key=tables_by_key)


def test_per_cell_composition_counts_near_zero_and_mixed_sign_per_cell():
    from scoring.audits import per_cell_fall_power_composition

    clean = np.full((7, 7), 1e-2)
    two_near_zero = np.full((7, 7), 1e-2)
    two_near_zero[0, 0] = 1e-5
    two_near_zero[0, 1] = -1e-5   # negative and tiny: near-zero AND mixed-sign
    lib = _fake_lib({"A": [("fall_power", clean)], "B": [("fall_power", two_near_zero)]})
    got = per_cell_fall_power_composition(lib)

    assert got["A"].n_near_zero == 0
    assert got["A"].n_points == 49
    assert got["A"].n_mixed_tables == 0
    assert got["A"].n_tables == 1
    assert got["B"].n_near_zero == 2
    assert got["B"].n_mixed_tables == 1


def test_per_cell_composition_ignores_other_table_types_and_all_zero_tables():
    from scoring.audits import per_cell_fall_power_composition

    lib = _fake_lib({
        "A": [("cell_rise", np.full((7, 7), 1e-5))],      # wrong table type
        "B": [("fall_power", np.zeros((7, 7)))],           # rule-3 invalid
        "C": [("fall_power", np.full((7, 7), 1e-2))],
    })
    got = per_cell_fall_power_composition(lib)
    assert set(got) == {"C"}


def test_aggregate_over_all_cells_matches_the_whole_lib_measurement():
    """The subset aggregator must agree with the pooled function it
    generalizes when the subset is 'everything'."""
    from scoring.audits import (aggregate_composition, measure_fall_power_composition,
                                per_cell_fall_power_composition)

    rng = np.random.default_rng(7)
    tables = {}
    for i in range(6):
        v = rng.normal(0, 1e-3, size=(7, 7))
        tables[f"C{i}"] = [("fall_power", v), ("rise_power", v)]
    lib = _fake_lib(tables)

    want = measure_fall_power_composition(lib)
    got = aggregate_composition(per_cell_fall_power_composition(lib), sorted(tables))
    assert got == pytest.approx(want, abs=1e-12)


def test_aggregate_weights_cells_by_table_count_not_equally():
    """A cell with more tables must carry more weight -- averaging
    per-cell shares would let a 1-table cell outvote a 3-table one."""
    from scoring.audits import aggregate_composition, per_cell_fall_power_composition

    dirty = np.full((7, 7), 1e-5)          # all 49 points near-zero
    clean = np.full((7, 7), 1e-2)
    lib = _fake_lib({
        "SMALL": [("fall_power", dirty)],                                    # 1 table, all dirty
        "BIG": [("fall_power", clean), ("fall_power", clean), ("fall_power", clean)],
    })
    nz_share, mixed_share = aggregate_composition(per_cell_fall_power_composition(lib),
                                                  ["SMALL", "BIG"])
    assert nz_share == pytest.approx(49 / (49 * 4))   # 1 of 4 tables, not 1/2
    assert mixed_share == 0.0


def test_aggregate_rejects_a_subset_with_no_valid_tables():
    from scoring.audits import aggregate_composition, per_cell_fall_power_composition

    lib = _fake_lib({"A": [("fall_power", np.full((7, 7), 1e-2))]})
    per_cell = per_cell_fall_power_composition(lib)
    with pytest.raises(ValueError):
        aggregate_composition(per_cell, ["NOT_A_CELL"])


def test_composition_multiplier_direction_a_cleaner_target_scores_higher():
    """The one place the audit can be silently inverted. A target
    population half as pathological as the scored one must get m<1 (its
    pathological points carry less weight -> higher score), never m>1."""
    from scoring.audits import composition_multiplier

    m = composition_multiplier(k=1.0, expected=0.01, base=0.02)
    assert m == pytest.approx(0.5)
    assert m < 1.0

    # k scales it linearly, and a dirtier target must exceed 1.
    assert composition_multiplier(k=0.5, expected=0.01, base=0.02) == pytest.approx(0.25)
    assert composition_multiplier(k=1.0, expected=0.04, base=0.02) == pytest.approx(2.0)


def test_composition_multiplier_applied_to_errors_moves_the_score_the_right_way():
    """End-to-end direction check through the scorer: down-weighting a
    failing subgroup must raise the pooled score."""
    from scoring.audits import composition_multiplier, reweighted_pooled_score

    errs = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    groups = np.array(["patho"] * 2 + ["bulk"] * 4)
    baseline = score_from_errors(errs)
    m = composition_multiplier(k=1.0, expected=0.005, base=0.02)  # target 4x cleaner
    got = reweighted_pooled_score(errs, groups, {"patho": m})
    assert m < 1.0
    assert got > baseline


def test_composition_multiplier_rejects_a_zero_base():
    from scoring.audits import composition_multiplier

    with pytest.raises(ValueError):
        composition_multiplier(k=1.0, expected=0.01, base=0.0)
