"""Scoring for predicted Liberty table values.

    Score = 100 - 100 * sqrt( (1/n) * sum_i( min(1, |y_i - yhat_i| / |y_i|)^2 ) )

The relative error is taken against the *true* value `y_i` (QA A4), and
capped at 1 (100% relative error) per point so a single wild outlier
cannot blow the score past a "fail" (=1) contribution -- see
docs/plan.md section 2.

Convention for `y_i == 0` (not specified by the contest formula, so we
define it here): a true value of exactly 0 most commonly represents an
invalid/known-zero internal_power arc. If the prediction is also exactly
0, the point error is defined as 0 (perfect). Otherwise it is defined as
1 (the same as a saturated/failed point), since relative error is
undefined at y_i == 0 and any nonzero prediction is an unbounded
relative miss. This is our own convention, not part of the contest spec.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np

from liberty.parser import TABLE_KINDS, LibertyFile, parse_file


def point_errors(y_true, y_pred) -> np.ndarray:
    """Vectorized per-point capped relative error, per the convention
    documented in this module's docstring for y_true == 0."""
    y = np.asarray(y_true, dtype=float)
    yhat = np.asarray(y_pred, dtype=float)
    if y.shape != yhat.shape:
        raise ValueError(f"shape mismatch: y_true {y.shape} vs y_pred {yhat.shape}")
    if not np.isfinite(yhat).all():
        raise ValueError("y_pred contains NaN/Inf; bad predictions must fail loudly")

    err = np.empty_like(y)
    zero_mask = y == 0
    nz = ~zero_mask

    err[nz] = np.minimum(1.0, np.abs(y[nz] - yhat[nz]) / np.abs(y[nz]))
    err[zero_mask] = np.where(yhat[zero_mask] == 0, 0.0, 1.0)
    return err


def score_from_errors(errors: Iterable[float]) -> float:
    errors = np.asarray(list(errors) if not isinstance(errors, np.ndarray) else errors, dtype=float)
    if errors.size == 0:
        raise ValueError("cannot score an empty set of points")
    rms = np.sqrt(np.mean(errors**2))
    return 100.0 - 100.0 * rms


def score_arrays(y_true, y_pred) -> float:
    """Score comparing two same-shape arrays of values directly."""
    return score_from_errors(point_errors(y_true, y_pred).ravel())


@dataclass
class ScoreReport:
    overall: float
    n_points: int
    by_table_type: Dict[str, "TableTypeScore"] = field(default_factory=dict)

    def summary_lines(self) -> list:
        lines = [f"overall: {self.overall:.4f}  (n={self.n_points})"]
        for table_type in TABLE_KINDS:
            if table_type in self.by_table_type:
                s = self.by_table_type[table_type]
                lines.append(f"  {table_type:16s}: {s.score:.4f}  (n={s.n_points})")
        return lines

    def __str__(self) -> str:
        return "\n".join(self.summary_lines())


@dataclass
class TableTypeScore:
    score: float
    n_points: int


def _as_lib(lib_or_path) -> LibertyFile:
    if isinstance(lib_or_path, LibertyFile):
        return lib_or_path
    return parse_file(lib_or_path)


def compare_libs(reference, predicted, *, keys: Optional[Iterable] = None) -> ScoreReport:
    """Compare two Liberty files (paths or already-parsed LibertyFile
    objects) at table-value granularity.

    Only keys present (and non-blank) in both files are scored. Pass
    `keys` to restrict scoring to a specific subset of table keys (e.g.
    only the tables that were blank in some template).

    Returns a ScoreReport with the overall pooled score (all points from
    all tables combined into a single RMS) and a breakdown by table type
    (cell_rise, cell_fall, rise_transition, fall_transition, rise_power,
    fall_power).
    """
    ref = _as_lib(reference)
    pred = _as_lib(predicted)

    if keys is None:
        keys = set(ref.tables_by_key) & set(pred.tables_by_key)
    else:
        keys = set(keys)

    errors_by_type = defaultdict(list)
    for key in keys:
        ref_table = ref.tables_by_key.get(key)
        pred_table = pred.tables_by_key.get(key)
        if ref_table is None or pred_table is None:
            continue
        if ref_table.values is None or pred_table.values is None:
            continue  # cannot score a blank table against anything
        errs = point_errors(ref_table.values, pred_table.values)
        errors_by_type[key[-1]].append(errs.ravel())

    by_table_type = {}
    all_errors = []
    for table_type, chunks in errors_by_type.items():
        flat = np.concatenate(chunks)
        by_table_type[table_type] = TableTypeScore(
            score=score_from_errors(flat), n_points=flat.size
        )
        all_errors.append(flat)

    if not all_errors:
        raise ValueError("no overlapping non-blank table keys to score")

    pooled = np.concatenate(all_errors)
    return ScoreReport(
        overall=score_from_errors(pooled), n_points=pooled.size, by_table_type=by_table_type
    )


def compare_lib_files(reference_path: str, predicted_path: str, **kwargs) -> ScoreReport:
    return compare_libs(parse_file(reference_path), parse_file(predicted_path), **kwargs)
