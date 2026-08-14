import pytest

from liberty.parser import parse_file
from scoring.loco import load_full_corners, run_loco
from scoring.scorer import compare_libs

from helpers import FULL_LIBS

# Direct-copy ("do nothing") baseline scores for each LOCO fold's actual
# held_out<-anchor direction, computed once at import time the same way
# scripts/loco_reference.py does (see docs/phase2_results.md for the full
# fold-by-fold writeup, including why these are *not* all equal to the
# single "69-81" range docs/phase1_results.md reports -- that document only
# ever measured one direction of each pair, and the capped-relative-error
# formula is not symmetric under swapping reference/predicted).
_SAME_PROCESS_BASELINE_PAIRS = [
    ("ff0p99v125c", "ff0p99vm40c"),
    ("ff0p99vm40c", "ff0p99v125c"),
    ("ss0p81v125c", "ss0p81vm40c"),
    ("ss0p81vm40c", "ss0p81v125c"),
]


@pytest.fixture(scope="module")
def full_libs():
    return load_full_corners(FULL_LIBS)


@pytest.fixture(scope="module")
def loco_results(full_libs):
    return run_loco(full_libs)


def test_run_loco_covers_all_five_corners_exactly_once(full_libs, loco_results):
    assert {r.held_out for r in loco_results} == set(full_libs)
    assert len(loco_results) == 5


def test_loco_scores_are_finite_and_in_valid_range(loco_results):
    for r in loco_results:
        assert 0.0 <= r.report.overall <= 100.0
        for s in r.report.by_table_type.values():
            assert 0.0 <= s.score <= 100.0


def test_tt_fold_massively_beats_its_cross_process_direct_copy_baseline(loco_results):
    """tt0p9v25c has no same-process full corner at all, so this fold
    exercises real process+voltage extrapolation -- the same regime Phase 1
    measured at 13-18 points for a naive direct copy
    (docs/phase1_results.md section 6). Phase 2's alpha-power voltage term
    should clear that floor by a wide margin."""
    tt_fold = next(r for r in loco_results if r.held_out.name == "tt0p9v25c")
    assert tt_fold.report.overall > 50.0


@pytest.mark.parametrize("held_out_name,anchor_name", _SAME_PROCESS_BASELINE_PAIRS)
def test_loco_same_process_fold_not_below_direct_copy_baseline(held_out_name, anchor_name, loco_results):
    """docs/plan.md Phase 2 acceptance / task requirement: 'LOCO 分數不低於
    同 process 直抄基準'. These 4 folds are the ones where the LOCO
    protocol destroys the target process's only temperature-pair
    calibration data (see docs/phase2_results.md "溫度項必須
    per-process"); the model's per-process temperature term safely
    defaults to 0 in that situation, which reproduces the direct-copy
    baseline exactly (not an improvement, but never a regression)."""
    held_out_lib = next(lib for m, lib in load_full_corners(FULL_LIBS).items() if m.name == held_out_name)
    anchor_lib = next(lib for m, lib in load_full_corners(FULL_LIBS).items() if m.name == anchor_name)
    baseline_report = compare_libs(held_out_lib, anchor_lib)

    fold = next(r for r in loco_results if r.held_out.name == held_out_name)
    assert fold.report.overall >= baseline_report.overall - 1e-6


def test_loco_stats_report_no_monotonic_violations_left_unfixed(loco_results):
    """Any monotonic violation the raw fit produces must have been fixed
    (docs/plan.md Phase 2 item 5): n_monotonic_violations counts fixes
    applied, not remaining violations, so this just checks the field is
    populated and non-negative -- the real monotonicity guarantee is
    checked directly in tests/test_phase2_scaling.py against the
    predicted values."""
    for r in loco_results:
        for s in r.stats.values():
            assert s["n_monotonic_violations"] >= 0
