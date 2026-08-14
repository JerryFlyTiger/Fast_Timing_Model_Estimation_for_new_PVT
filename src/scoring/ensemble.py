"""Phase 2.5 refit-ensemble diagnostics (docs/phase2_review.md item 4).

Two independent sources of prediction spread for the riskiest buck
corner, `ss0p72` (the largest step-down voltage extrapolation among the
10 delivered targets), used as an objective, data-driven basis for the
`SHRINK_LAMBDA` choice rather than picking it purely by decree:

1. **Cell-jackknife** (`run_cell_jackknife`): repeatedly drop a random
   20% of the ~100 cells from the training set, refit the whole Phase
   2.5 model on the rest, and re-predict ss0p72's delay tables against
   the *original* (un-dropped) anchor values -- this measures how
   sensitive the fitted Vth/alpha/c0/gain parameters are to which cells
   happen to be in the training pool, independent of the box-bound
   choice itself. Repeated ~20 times (docs/phase2_review.md item 4).
2. **Bounds-endpoint perturbation** (`run_bounds_perturbation`): refit
   on the full (unfiltered) cell set with `VTH_BOUNDS`/`ALPHA_BOUNDS`
   nudged to nearby alternatives, one bound-endpoint at a time. This
   measures how sensitive the delivered scaling factor is to exactly
   where the physical-prior box sits, independent of sampling noise.

Both report the resulting distribution of the ss0p72 delay
predicted/anchor scaling factor (p5/p50/p95) across runs, both
individually and pooled (`run_ensemble`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from features.corners import CornerMeta
from liberty.parser import LibertyFile
from models.phase2_scaling import (
    ALPHA_BOUNDS,
    DELAY_TABLE_TYPES,
    VTH_BOUNDS,
    fit_phase2_model,
    predict_corner,
    select_anchors,
)

SS0P72_TARGETS = ("ss0p72v125c", "ss0p72vm40c")


def _filter_lib_cells(lib: LibertyFile, keep_cells: set) -> LibertyFile:
    """Return a shadow `LibertyFile` (same pattern as
    `scoring.loco._wrap_predictions_as_lib`) exposing only the tables/
    cells in `keep_cells`, for building a cell-subsampled training set
    without touching the parser/writer or the original object."""
    cells = {name: c for name, c in lib.cells.items() if name in keep_cells}
    tables = [t for t in lib.tables if t.key[0] in keep_cells]
    tables_by_key = {k: v for k, v in lib.tables_by_key.items() if k[0] in keep_cells}
    return LibertyFile(
        path=lib.path, text=lib.text, library_name=lib.library_name,
        cells=cells, tables=tables, tables_by_key=tables_by_key,
    )


def _ss0p72_delay_ratio(
    full_libs: Mapping[CornerMeta, LibertyFile],
    partial_libs: Mapping[str, Tuple[CornerMeta, LibertyFile]],
    model,
) -> np.ndarray:
    """Predict both ss0p72 targets with `model` (against the *full,
    unfiltered* anchor -- only the model's fitted parameters vary across
    ensemble runs, not the anchor data) and return the pooled
    predicted/anchor ratio over the delay family, one float per grid
    point."""
    chunks = []
    for name in SS0P72_TARGETS:
        target_meta, target_lib = partial_libs[name]
        anchors = select_anchors(target_meta, full_libs)
        anchor_lib = full_libs[anchors[0]]
        preds = predict_corner(model, target_lib, target_meta, anchors, full_libs)
        for key, values in preds.items():
            if key[-1] not in DELAY_TABLE_TYPES:
                continue
            anchor_table = anchor_lib.tables_by_key.get(key)
            if anchor_table is None or anchor_table.values is None:
                continue
            mask = anchor_table.values != 0
            if not mask.any():
                continue
            chunks.append(values[mask] / anchor_table.values[mask])
    return np.concatenate(chunks)


@dataclass
class EnsembleRunResult:
    label: str
    p50_ratio: float  # this run's own median ss0p72 delay scaling factor


@dataclass
class EnsembleReport:
    jackknife_runs: List[EnsembleRunResult]
    bounds_runs: List[EnsembleRunResult]

    def _percentiles(self, runs: List[EnsembleRunResult]) -> Tuple[float, float, float]:
        vals = np.array([r.p50_ratio for r in runs])
        return (
            float(np.percentile(vals, 5)),
            float(np.percentile(vals, 50)),
            float(np.percentile(vals, 95)),
        )

    @property
    def jackknife_p5_p50_p95(self) -> Tuple[float, float, float]:
        return self._percentiles(self.jackknife_runs)

    @property
    def bounds_p5_p50_p95(self) -> Tuple[float, float, float]:
        return self._percentiles(self.bounds_runs)

    @property
    def combined_p5_p50_p95(self) -> Tuple[float, float, float]:
        return self._percentiles(self.jackknife_runs + self.bounds_runs)

    def summary_lines(self) -> List[str]:
        jk_lo, jk_mid, jk_hi = self.jackknife_p5_p50_p95
        b_lo, b_mid, b_hi = self.bounds_p5_p50_p95
        c_lo, c_mid, c_hi = self.combined_p5_p50_p95
        lines = [
            f"cell-jackknife ({len(self.jackknife_runs)} runs): "
            f"p5={jk_lo:.4f} p50={jk_mid:.4f} p95={jk_hi:.4f}",
            f"bounds-perturbation ({len(self.bounds_runs)} runs): "
            f"p5={b_lo:.4f} p50={b_mid:.4f} p95={b_hi:.4f}",
            f"combined ({len(self.jackknife_runs) + len(self.bounds_runs)} runs): "
            f"p5={c_lo:.4f} p50={c_mid:.4f} p95={c_hi:.4f}",
        ]
        return lines


def run_cell_jackknife(
    full_libs: Mapping[CornerMeta, LibertyFile],
    partial_libs: Mapping[str, Tuple[CornerMeta, LibertyFile]],
    *,
    n_runs: int = 20,
    drop_frac: float = 0.2,
    seed: int = 0,
) -> List[EnsembleRunResult]:
    """docs/phase2_review.md item 4: repeatedly drop a random `drop_frac`
    of cells from the training pool, refit, and re-predict ss0p72 --
    both partial targets (`SS0P72_TARGETS`) share the same cell set
    across all 5 full corners, so one dropped-cell set is applied
    uniformly to every corner in `full_libs`."""
    all_cells = sorted(set.intersection(*(set(lib.cells) for lib in full_libs.values())))
    rng = np.random.default_rng(seed)
    n_keep = int(round(len(all_cells) * (1.0 - drop_frac)))

    results = []
    for i in range(n_runs):
        keep = set(rng.choice(all_cells, size=n_keep, replace=False))
        filtered = {meta: _filter_lib_cells(lib, keep) for meta, lib in full_libs.items()}
        model = fit_phase2_model(filtered)
        ratio = _ss0p72_delay_ratio(full_libs, partial_libs, model)
        results.append(EnsembleRunResult(label=f"jackknife[{i}]", p50_ratio=float(np.median(ratio))))
    return results


# Small, deliberately asymmetric perturbations to each bound's two
# endpoints (docs/phase2_review.md item 4: "bounds 端點擾動"). Each
# variant nudges exactly one endpoint of one bound, so the resulting
# spread attributes cleanly to "how much does moving *this* edge of the
# physical-prior box change the delivered scaling factor".
_BOUNDS_VARIANTS: List[Tuple[str, Tuple[float, float], Tuple[float, float]]] = [
    ("vth_lo-0.03", (VTH_BOUNDS[0] - 0.03, VTH_BOUNDS[1]), ALPHA_BOUNDS),
    ("vth_lo+0.03", (VTH_BOUNDS[0] + 0.03, VTH_BOUNDS[1]), ALPHA_BOUNDS),
    ("vth_hi-0.03", (VTH_BOUNDS[0], VTH_BOUNDS[1] - 0.03), ALPHA_BOUNDS),
    ("vth_hi+0.03", (VTH_BOUNDS[0], VTH_BOUNDS[1] + 0.03), ALPHA_BOUNDS),
    ("alpha_lo-0.1", VTH_BOUNDS, (ALPHA_BOUNDS[0] - 0.1, ALPHA_BOUNDS[1])),
    ("alpha_lo+0.1", VTH_BOUNDS, (ALPHA_BOUNDS[0] + 0.1, ALPHA_BOUNDS[1])),
    ("alpha_hi-0.1", VTH_BOUNDS, (ALPHA_BOUNDS[0], ALPHA_BOUNDS[1] - 0.1)),
    ("alpha_hi+0.1", VTH_BOUNDS, (ALPHA_BOUNDS[0], ALPHA_BOUNDS[1] + 0.1)),
]


def run_bounds_perturbation(
    full_libs: Mapping[CornerMeta, LibertyFile],
    partial_libs: Mapping[str, Tuple[CornerMeta, LibertyFile]],
    *,
    variants: Sequence[Tuple[str, Tuple[float, float], Tuple[float, float]]] = _BOUNDS_VARIANTS,
) -> List[EnsembleRunResult]:
    """docs/phase2_review.md item 4: refit on the full (unfiltered) cell
    set with each `(vth_bounds, alpha_bounds)` variant in turn, and
    re-predict ss0p72."""
    results = []
    for label, vth_bounds, alpha_bounds in variants:
        model = fit_phase2_model(full_libs, vth_bounds=vth_bounds, alpha_bounds=alpha_bounds)
        ratio = _ss0p72_delay_ratio(full_libs, partial_libs, model)
        results.append(EnsembleRunResult(label=label, p50_ratio=float(np.median(ratio))))
    return results


def run_ensemble(
    full_libs: Mapping[CornerMeta, LibertyFile],
    partial_libs: Mapping[str, Tuple[CornerMeta, LibertyFile]],
    *,
    n_jackknife: int = 20,
    drop_frac: float = 0.2,
    seed: int = 0,
) -> EnsembleReport:
    jackknife_runs = run_cell_jackknife(full_libs, partial_libs, n_runs=n_jackknife, drop_frac=drop_frac, seed=seed)
    bounds_runs = run_bounds_perturbation(full_libs, partial_libs)
    return EnsembleReport(jackknife_runs=jackknife_runs, bounds_runs=bounds_runs)
