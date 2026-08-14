"""Phase 2.5 physical audits (docs/phase2_review.md item 3, "待補" items).

These are checks on the *delivered* predictions (the 10 partial-corner
`output/*.lib` files), independent of and in addition to the pointwise
voltage-monotonicity / clip / shrink diagnostics `models.phase2_scaling`
already tracks:

1. **Cross-corner process-ordering inequalities**
   (`run_cross_corner_inequality_audit`): SS is always the slowest
   process and FF the fastest, regardless of voltage; TT sits in
   between. This physical ordering must still hold *after* boosting or
   bucking a process across the +-10% VDD gap -- a corner transfer that
   flips it is a strong, cheap signal of a broken extrapolation
   direction, independent of the scorer (which only ever compares a
   prediction to its own corner's ground truth, never across corners).

   Three checks (docs/phase2_review.md item 3's "待補" bullet), each
   compared pointwise on the delay family (cell_rise, cell_fall,
   rise_transition, fall_transition):

   a. predicted ss0p9 (v125c and vm40c) >= true tt0p9v25c -- SS boosted
      all the way up to TT's own nominal voltage must still not be
      faster than TT itself (SS is the slow corner at any V).
   b. predicted tt1p0v25c >= true ff0p99 -- TT boosted to 1.0V must
      still not be faster than FF (the fast corner) at either of FF's
      calibrated temperatures. FF's ground truth only exists at
      +-125C/-40C, not TT's target 25C, so the comparison uses whichever
      of the two FF readings is *slower* (the larger delay) as the
      threshold: if predicted TT delay exceeds even FF's slower/larger
      reading, it certainly exceeds FF's true (unobserved) 25C value
      too, whatever that interpolates to between the two extremes. This
      is the "conservative" (safe-for-our-claim, still a real test)
      choice named in docs/phase2_review.md item 3.
   c. predicted tt0p8v25c <= true ss0p81 -- TT bucked down to 0.8V must
      still not be slower than SS (the slow corner) at either of SS's
      calibrated temperatures. Symmetric reasoning to (b): use whichever
      of SS's two readings is *faster* (the smaller delay) as the
      threshold, so passing it implies passing against SS's true
      (unobserved) 25C value too.

   A check fails if its pointwise violation rate exceeds
   `VIOLATION_RATE_THRESHOLD` (1%, per docs/phase2_review.md item 3).

2. **Scaling-factor distribution report** (`scaling_factor_quantiles`,
   `check_delay_scaling_bands`, `check_power_k_band`): for every
   delivered corner, the quantiles (p1/p50/p99) of the pointwise
   predicted/anchor ratio (== `exp(Delta)`, always positive regardless
   of the source table's sign since the prediction is multiplicative --
   see `models.phase2_scaling`'s module docstring), checked against the
   architect's physical-midband estimate for the two most
   extrapolation-risky delivered corners (ss0p72, ff1p1) plus the power
   exponent `k`'s expected range.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from liberty.parser import LibertyFile, TableKey
from models.phase2_scaling import DELAY_TABLE_TYPES, POWER_TABLE_TYPES, ShapeParams

VIOLATION_RATE_THRESHOLD = 0.01  # docs/phase2_review.md item 3: ">1% 即審計失敗"

# docs/phase2_review.md item 3, "待補: 縮放因子分佈報告" band checks.
# Keyed by the corner-name *prefix* shared by both temperatures
# (ss0p72v125c/ss0p72vm40c -> "ss0p72"), since the review's band applies
# to the corner as a physical target, not to one particular temperature.
DELAY_SCALING_BANDS = {
    "ss0p72": (1.1, 2.0),   # inclusive both ends
    "ff1p1": (0.7, 1.0),    # upper bound exclusive, see check_delay_scaling_bands
}
POWER_K_BAND = (2.0, 3.5)  # inclusive both ends


# ---------------------------------------------------------------------------
# 1. Cross-corner process-ordering inequality audit
# ---------------------------------------------------------------------------


@dataclass
class InequalityResult:
    name: str
    n_points: int
    n_violations: int
    by_table_type: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # table_type -> (n_violations, n_points)
    max_violation_magnitude: float = 0.0  # worst observed (threshold - value)/|threshold|, 0 if no violations

    @property
    def violation_rate(self) -> float:
        return self.n_violations / self.n_points if self.n_points else 0.0

    @property
    def passed(self) -> bool:
        return self.violation_rate <= VIOLATION_RATE_THRESHOLD

    def summary_line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.name}: {self.n_violations}/{self.n_points} "
            f"({100 * self.violation_rate:.4f}%) violations, "
            f"worst_magnitude={self.max_violation_magnitude:.4f}"
        )


def _common_delay_keys(lib_a: LibertyFile, lib_b: LibertyFile) -> List[TableKey]:
    return [
        k for k in (set(lib_a.tables_by_key) & set(lib_b.tables_by_key))
        if k[-1] in DELAY_TABLE_TYPES
    ]


def _compare_pointwise(
    name: str,
    pred_lib: LibertyFile,
    threshold_tables: Mapping[TableKey, np.ndarray],
    *,
    direction: str,  # ">=" or "<="
) -> InequalityResult:
    """`direction == ">="` checks pred >= threshold (a "predicted must not
    be faster" audit); `"<="` checks pred <= threshold ("predicted must
    not be slower"). `threshold_tables` supplies, per key, the value(s)
    the corresponding predicted table must satisfy the inequality
    against."""
    n_points = 0
    n_violations = 0
    by_table_type: Dict[str, List[int]] = {}
    worst = 0.0

    for key, threshold in threshold_tables.items():
        pred_table = pred_lib.tables_by_key.get(key)
        if pred_table is None or pred_table.values is None:
            continue
        pred = pred_table.values
        if direction == ">=":
            ok = pred >= threshold - 1e-9
            margin = np.where(~ok, (threshold - pred) / np.maximum(np.abs(threshold), 1e-30), 0.0)
        else:
            ok = pred <= threshold + 1e-9
            margin = np.where(~ok, (pred - threshold) / np.maximum(np.abs(threshold), 1e-30), 0.0)

        viol = int(np.sum(~ok))
        n_points += ok.size
        n_violations += viol
        worst = max(worst, float(np.max(margin)) if margin.size else 0.0)

        table_type = key[-1]
        acc = by_table_type.setdefault(table_type, [0, 0])
        acc[0] += viol
        acc[1] += ok.size

    return InequalityResult(
        name=name,
        n_points=n_points,
        n_violations=n_violations,
        by_table_type={t: tuple(v) for t, v in by_table_type.items()},
        max_violation_magnitude=worst,
    )


def _extreme_across_libs(
    libs: List[LibertyFile], keys: List[TableKey], *, extreme: str  # "max" or "min"
) -> Dict[TableKey, np.ndarray]:
    """Pointwise max (or min) of the same table across several ground-
    truth libs, restricted to `keys` present with non-blank values in
    every lib supplied."""
    op = np.maximum if extreme == "max" else np.minimum
    out: Dict[TableKey, np.ndarray] = {}
    for key in keys:
        arrays = []
        for lib in libs:
            t = lib.tables_by_key.get(key)
            if t is None or t.values is None:
                arrays = []
                break
            arrays.append(t.values)
        if not arrays:
            continue
        result = arrays[0]
        for a in arrays[1:]:
            result = op(result, a)
        out[key] = result
    return out


def run_cross_corner_inequality_audit(
    predicted: Mapping[str, LibertyFile],
    truth: Mapping[str, LibertyFile],
) -> List[InequalityResult]:
    """Run all three cross-corner process-ordering checks (see module
    docstring). `predicted` maps delivered partial-corner names (e.g.
    "ss0p9v125c") to their filled `output/*.lib` LibertyFile; `truth`
    maps full-corner names (e.g. "tt0p9v25c") to their ground-truth
    LibertyFile. Missing entries in either mapping simply skip that
    check (callers should assert on `len(results)` if completeness
    matters)."""
    results: List[InequalityResult] = []

    # (a) predicted ss0p9{v125c,vm40c} >= true tt0p9v25c
    if "tt0p9v25c" in truth:
        tt_truth = truth["tt0p9v25c"]
        for name in ("ss0p9v125c", "ss0p9vm40c"):
            if name not in predicted:
                continue
            pred_lib = predicted[name]
            keys = _common_delay_keys(pred_lib, tt_truth)
            threshold = {k: tt_truth.tables_by_key[k].values for k in keys}
            results.append(
                _compare_pointwise(f"{name} >= tt0p9v25c", pred_lib, threshold, direction=">=")
            )

    # (b) predicted tt1p0v25c >= max(true ff0p99v125c, true ff0p99vm40c)
    if "tt1p0v25c" in predicted and "ff0p99v125c" in truth and "ff0p99vm40c" in truth:
        pred_lib = predicted["tt1p0v25c"]
        ff_libs = [truth["ff0p99v125c"], truth["ff0p99vm40c"]]
        keys = [k for k in _common_delay_keys(pred_lib, ff_libs[0]) if k in ff_libs[1].tables_by_key]
        threshold = _extreme_across_libs(ff_libs, keys, extreme="max")
        results.append(
            _compare_pointwise("tt1p0v25c >= max(ff0p99v125c, ff0p99vm40c)", pred_lib, threshold, direction=">=")
        )

    # (c) predicted tt0p8v25c <= min(true ss0p81v125c, true ss0p81vm40c)
    if "tt0p8v25c" in predicted and "ss0p81v125c" in truth and "ss0p81vm40c" in truth:
        pred_lib = predicted["tt0p8v25c"]
        ss_libs = [truth["ss0p81v125c"], truth["ss0p81vm40c"]]
        keys = [k for k in _common_delay_keys(pred_lib, ss_libs[0]) if k in ss_libs[1].tables_by_key]
        threshold = _extreme_across_libs(ss_libs, keys, extreme="min")
        results.append(
            _compare_pointwise("tt0p8v25c <= min(ss0p81v125c, ss0p81vm40c)", pred_lib, threshold, direction="<=")
        )

    return results


# ---------------------------------------------------------------------------
# 2. Scaling-factor distribution report
# ---------------------------------------------------------------------------


@dataclass
class ScalingFactorQuantiles:
    corner: str
    table_type: str
    n_points: int
    p1: float
    p50: float
    p99: float


def scaling_factor_quantiles(
    corner_name: str, predicted_lib: LibertyFile, anchor_lib: LibertyFile
) -> List[ScalingFactorQuantiles]:
    """Per table_type quantiles (p1/p50/p99) of the pointwise
    predicted/anchor ratio for one delivered corner, restricted to
    points where the anchor is nonzero (a zero anchor is the known-
    invalid all-zero power arc convention, docs/plan.md rule 3; the
    ratio there is an undefined 0/0, correctly predicted as 0/0 -> 0,
    and excluded from the distribution rather than reported as some
    arbitrary placeholder)."""
    out: List[ScalingFactorQuantiles] = []
    by_type: Dict[str, List[float]] = {}
    keys = set(predicted_lib.tables_by_key) & set(anchor_lib.tables_by_key)
    for key in keys:
        table_type = key[-1]
        if table_type not in DELAY_TABLE_TYPES and table_type not in POWER_TABLE_TYPES:
            continue
        pred_table = predicted_lib.tables_by_key[key]
        anchor_table = anchor_lib.tables_by_key[key]
        if pred_table.values is None or anchor_table.values is None:
            continue
        anchor_vals = anchor_table.values
        pred_vals = pred_table.values
        mask = anchor_vals != 0
        if not mask.any():
            continue
        ratio = pred_vals[mask] / anchor_vals[mask]
        by_type.setdefault(table_type, []).append(ratio)

    for table_type, chunks in sorted(by_type.items()):
        flat = np.concatenate(chunks)
        out.append(
            ScalingFactorQuantiles(
                corner=corner_name,
                table_type=table_type,
                n_points=flat.size,
                p1=float(np.percentile(flat, 1)),
                p50=float(np.percentile(flat, 50)),
                p99=float(np.percentile(flat, 99)),
            )
        )
    return out


@dataclass
class BandCheck:
    name: str
    value: float
    lo: float
    hi: float
    hi_inclusive: bool

    @property
    def passed(self) -> bool:
        if self.hi_inclusive:
            return self.lo <= self.value <= self.hi
        return self.lo <= self.value < self.hi

    def summary_line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        hi_bracket = "]" if self.hi_inclusive else ")"
        return f"[{status}] {self.name}: {self.value:.4f} in [{self.lo}, {self.hi}{hi_bracket}"


def _pooled_ratio(corner_prefix: str, predicted: Mapping[str, LibertyFile], anchor_libs: Mapping[str, LibertyFile], anchor_of: Mapping[str, str], table_types) -> np.ndarray:
    chunks = []
    for name, pred_lib in predicted.items():
        if not name.startswith(corner_prefix):
            continue
        anchor_lib = anchor_libs[anchor_of[name]]
        for key in set(pred_lib.tables_by_key) & set(anchor_lib.tables_by_key):
            if key[-1] not in table_types:
                continue
            pred_table = pred_lib.tables_by_key[key]
            anchor_table = anchor_lib.tables_by_key[key]
            if pred_table.values is None or anchor_table.values is None:
                continue
            mask = anchor_table.values != 0
            if not mask.any():
                continue
            chunks.append(pred_table.values[mask] / anchor_table.values[mask])
    if not chunks:
        raise ValueError(f"no points found for corner prefix {corner_prefix!r}")
    return np.concatenate(chunks)


def check_delay_scaling_bands(
    predicted: Mapping[str, LibertyFile],
    anchor_libs: Mapping[str, LibertyFile],
    anchor_of: Mapping[str, str],
) -> List[BandCheck]:
    """docs/phase2_review.md item 3: delay scaling factor p50 band checks
    for the two riskiest delivered corners, pooling both temperatures of
    each (e.g. ss0p72v125c + ss0p72vm40c) since the review's band is
    named at the corner-group level ("ss0p72"), not per temperature."""
    checks = []
    for prefix, (lo, hi) in DELAY_SCALING_BANDS.items():
        ratio = _pooled_ratio(prefix, predicted, anchor_libs, anchor_of, DELAY_TABLE_TYPES)
        hi_inclusive = not (prefix == "ff1p1")  # spec: ff1p1 in [0.7, 1.0) -- upper exclusive
        checks.append(BandCheck(name=f"{prefix} delay scaling p50", value=float(np.percentile(ratio, 50)), lo=lo, hi=hi, hi_inclusive=hi_inclusive))
    return checks


def check_power_k_band(model_params: Mapping[str, ShapeParams]) -> List[BandCheck]:
    """docs/phase2_review.md item 3: fitted power exponent k in [2, 3.5]."""
    checks = []
    lo, hi = POWER_K_BAND
    for table_type in POWER_TABLE_TYPES:
        p = model_params[table_type]
        checks.append(BandCheck(name=f"{table_type} k", value=p.k, lo=lo, hi=hi, hi_inclusive=True))
    return checks


# ---------------------------------------------------------------------------
# 3. Alpha-composition reweighted score audit (2026-08-09, direction F --
#    docs/recheck_20260809.md section 5)
#
# The acceptance CV measures held-out TRAINING cells, but the official
# deliverable is scored on the ALPHA cells, whose fall_power pathology
# (near-zero / sign-flip points, the dominant error mass) is ~2x rarer
# -- measured at the 5 anchor corners where both populations have real
# values. These helpers reweight a validation run's per-point errors to
# the alpha population's composition: pathological-group point shares are
# scaled by measured alpha/train prevalence ratios, everything else keeps
# its conditional error unchanged (stated assumption: composition shifts,
# conditional per-group error does not).
# ---------------------------------------------------------------------------

NEAR_ZERO_THRESHOLD = 1e-4  # |y| below this (and sign-consistent) == the
                             # "near-zero" pathological group, matching
                             # every prior diagnosis (docs/closure_report.md)

SUBGROUP_FLIP = "fall_power:flip"
SUBGROUP_NEAR_ZERO = "fall_power:near_zero"
SUBGROUP_BULK_FP = "fall_power:bulk"
SUBGROUP_OTHER = "other_tables"


def assign_subgroups(
    y_true: np.ndarray, nearest_anchor: np.ndarray, table_type: np.ndarray
) -> np.ndarray:
    """Per-point subgroup label (the four SUBGROUP_* constants) used by
    the alpha-composition audit. Flip/near-zero are only distinguished
    within fall_power -- the one table type where they carry material
    error mass (docs/recheck_20260809.md section 1)."""
    y = np.asarray(y_true, dtype=float)
    a = np.asarray(nearest_anchor, dtype=float)
    is_fp = np.asarray(table_type) == "fall_power"
    flip = is_fp & (y * a < 0)
    near_zero = is_fp & ~flip & (np.abs(y) < NEAR_ZERO_THRESHOLD) & (y != 0)
    out = np.full(y.shape, SUBGROUP_OTHER, dtype="<U24")
    out[is_fp] = SUBGROUP_BULK_FP
    out[near_zero] = SUBGROUP_NEAR_ZERO
    out[flip] = SUBGROUP_FLIP
    return out


@dataclass
class SubgroupStat:
    name: str
    n_points: int
    share: float          # fraction of all pooled points
    score: float          # contest score of this subgroup alone
    mean_sq_err: float    # mean of capped squared error within the group
    e2_mass: float        # share * mean_sq_err == contribution to pooled e2


def subgroup_stats(errors: np.ndarray, subgroups: np.ndarray) -> List[SubgroupStat]:
    errors = np.asarray(errors, dtype=float)
    n_total = errors.size
    out = []
    for name in sorted(set(subgroups.tolist())):
        mask = subgroups == name
        e2 = float(np.mean(errors[mask] ** 2))
        share = mask.sum() / n_total
        out.append(SubgroupStat(
            name=name, n_points=int(mask.sum()), share=float(share),
            score=100.0 - 100.0 * float(np.sqrt(e2)), mean_sq_err=e2,
            e2_mass=float(share * e2),
        ))
    return out


def reweighted_pooled_score(
    errors: np.ndarray,
    group_ids: np.ndarray,
    multipliers: Mapping[str, float],
) -> float:
    """Pooled contest score after scaling each group's point-population
    weight by `multipliers[group_id]` (default 1.0 for groups not
    listed), renormalizing so weights still sum to 1:

        e2 = sum_g(w_g * m_g * e2_g) / sum_g(w_g * m_g)

    This models a target population whose *composition* differs from the
    measured one (fewer/more points per group) while each group's
    conditional error distribution is unchanged."""
    errors = np.asarray(errors, dtype=float)
    n_total = errors.size
    num = 0.0
    den = 0.0
    for name in set(group_ids.tolist()):
        mask = group_ids == name
        w = mask.sum() / n_total
        m = float(multipliers.get(name, 1.0))
        num += w * m * float(np.mean(errors[mask] ** 2))
        den += w * m
    return 100.0 - 100.0 * float(np.sqrt(num / den))


@dataclass(frozen=True)
class CellComposition:
    """Raw pathology-prevalence counts for one cell's fall_power tables.
    Kept as counts rather than shares so an arbitrary cell subset can be
    aggregated by summation (a mean of per-cell shares would weight a
    2-table cell like a 20-table one)."""

    n_near_zero: int
    n_points: int
    n_mixed_tables: int
    n_tables: int


def per_cell_fall_power_composition(lib: LibertyFile) -> Dict[str, CellComposition]:
    """`measure_fall_power_composition` split per cell, so prevalence can
    be measured on a *subset* of a lib's cells (a drive-strength bucket,
    or the 80 held-out validation cells) rather than the whole file.

    Same exclusions as the pooled version: all-zero (rule-3 invalid)
    fall_power tables count toward neither numerator nor denominator.
    Cells whose fall_power tables are all invalid are omitted entirely
    (rather than returned with a zero denominator)."""
    acc: Dict[str, List[int]] = {}
    for key, t in lib.tables_by_key.items():
        if key[-1] != "fall_power" or t.values is None:
            continue
        v = t.values.ravel()
        if np.all(v == 0):
            continue
        cell = key[0]
        a = acc.setdefault(cell, [0, 0, 0, 0])
        a[0] += int(np.sum((np.abs(v) < NEAR_ZERO_THRESHOLD) & (v != 0)))
        a[1] += v.size
        a[2] += int(v.min() < 0 < v.max())
        a[3] += 1
    return {c: CellComposition(*a) for c, a in acc.items()}


def aggregate_composition(
    per_cell: Mapping[str, CellComposition], cells
) -> Tuple[float, float]:
    """(near_zero_point_share, mixed_sign_table_share) over `cells` --
    the same two proxies `measure_fall_power_composition` returns, but
    for a chosen subset. Cells absent from `per_cell` (all-invalid
    fall_power) are skipped."""
    n_nz = n_pts = n_mixed = n_tables = 0
    for c in cells:
        comp = per_cell.get(c)
        if comp is None:
            continue
        n_nz += comp.n_near_zero
        n_pts += comp.n_points
        n_mixed += comp.n_mixed_tables
        n_tables += comp.n_tables
    if n_pts == 0 or n_tables == 0:
        raise ValueError("no valid fall_power tables among the requested cells")
    return n_nz / n_pts, n_mixed / n_tables


def composition_multiplier(k: float, expected: float, base: float) -> float:
    """Population-reweighting multiplier for one (corner, subgroup):
    how many times more (or less) prevalent a pathology is in the target
    population than in the population actually scored.

        m = k * expected / base

    `expected` is the target population's drive-matched expected
    prevalence at this corner, `k` the residual population factor
    calibrated where the target's truth is observable, and `base` the
    prevalence directly measured on the scored cells.

    Trivial arithmetic, but it is the one place the whole audit can be
    silently inverted -- `base / (k * expected)` also yields plausible
    multipliers and a plausible score, just the wrong way round -- so it
    is a named, tested function rather than an inline expression.
    A cleaner target population (expected < base) must give m < 1."""
    if base <= 0:
        raise ValueError(f"base prevalence must be positive, got {base}")
    return k * expected / base


def measure_fall_power_composition(lib: LibertyFile) -> Tuple[float, float]:
    """(near_zero_point_share, mixed_sign_table_share) of a lib's
    fall_power tables -- the two observable pathology-prevalence proxies
    used to form alpha/train composition ratios at the anchor corners
    (where both populations have real values). All-zero (rule-3 invalid)
    tables are excluded from both denominators."""
    n_pts = n_nz = n_tables = n_mixed = 0
    for key, t in lib.tables_by_key.items():
        if key[-1] != "fall_power" or t.values is None:
            continue
        v = t.values.ravel()
        if np.all(v == 0):
            continue
        n_tables += 1
        n_pts += v.size
        n_nz += int(np.sum((np.abs(v) < NEAR_ZERO_THRESHOLD) & (v != 0)))
        if v.min() < 0 < v.max():
            n_mixed += 1
    if n_pts == 0:
        raise ValueError("lib has no non-blank, non-all-zero fall_power tables")
    return n_nz / n_pts, n_mixed / n_tables
