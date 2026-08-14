import numpy as np
import pytest

from features.corners import parse_corner_filename
from liberty.parser import parse_file
from scoring.ensemble import (
    SS0P72_TARGETS,
    _filter_lib_cells,
    run_bounds_perturbation,
    run_cell_jackknife,
    run_ensemble,
)

from helpers import FULL_DIR, PARTIAL_DIR

# Deliberately small counts here: this is a smoke test for the mechanism
# (each fit_phase2_model call costs several seconds, see
# docs/phase25_results.md for the real ~20-run ensemble numbers used in
# the writeup). n_jackknife=20 is the docs/phase2_review.md-specified
# count for the actual report, not for every pytest run.
N_JACKKNIFE_SMOKE = 3


@pytest.fixture(scope="module")
def full_libs():
    return {parse_corner_filename(p): parse_file(str(p)) for p in sorted(FULL_DIR.glob("*.lib"))}


@pytest.fixture(scope="module")
def partial_libs():
    out = {}
    for path in sorted(PARTIAL_DIR.glob("*.lib")):
        meta = parse_corner_filename(path)
        out[meta.name] = (meta, parse_file(str(path)))
    return out


def test_filter_lib_cells_drops_only_requested_cells(full_libs):
    any_lib = next(iter(full_libs.values()))
    all_cells = set(any_lib.cells)
    keep = set(sorted(all_cells)[:5])
    filtered = _filter_lib_cells(any_lib, keep)
    assert set(filtered.cells) == keep
    assert all(key[0] in keep for key in filtered.tables_by_key)
    assert all(t.key[0] in keep for t in filtered.tables)
    # original untouched
    assert set(any_lib.cells) == all_cells


def test_cell_jackknife_produces_one_ratio_per_run(full_libs, partial_libs):
    results = run_cell_jackknife(full_libs, partial_libs, n_runs=N_JACKKNIFE_SMOKE, seed=0)
    assert len(results) == N_JACKKNIFE_SMOKE
    for r in results:
        assert np.isfinite(r.p50_ratio)
        assert r.p50_ratio > 0


def test_cell_jackknife_is_reproducible_given_a_seed(full_libs, partial_libs):
    a = run_cell_jackknife(full_libs, partial_libs, n_runs=2, seed=42)
    b = run_cell_jackknife(full_libs, partial_libs, n_runs=2, seed=42)
    assert [r.p50_ratio for r in a] == [r.p50_ratio for r in b]


def test_bounds_perturbation_produces_one_ratio_per_variant(full_libs, partial_libs):
    results = run_bounds_perturbation(full_libs, partial_libs)
    assert len(results) == 8  # _BOUNDS_VARIANTS in scoring.ensemble
    for r in results:
        assert np.isfinite(r.p50_ratio)
        assert r.p50_ratio > 0


def test_run_ensemble_reports_p5_p50_p95_for_both_sources(full_libs, partial_libs):
    report = run_ensemble(full_libs, partial_libs, n_jackknife=N_JACKKNIFE_SMOKE)
    jk_lo, jk_mid, jk_hi = report.jackknife_p5_p50_p95
    assert jk_lo <= jk_mid <= jk_hi
    b_lo, b_mid, b_hi = report.bounds_p5_p50_p95
    assert b_lo <= b_mid <= b_hi
    c_lo, c_mid, c_hi = report.combined_p5_p50_p95
    assert c_lo <= c_mid <= c_hi
    lines = report.summary_lines()
    assert len(lines) == 3
