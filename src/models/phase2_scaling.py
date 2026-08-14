"""Phase 2 / 2.5 physical scaling model (docs/plan.md Phase 2 design,
docs/phase2_review.md Phase 2.5 corrections).

Additive log-space decomposition::

    log tau_target(entry) = log tau_source(entry) + Delta(P, V, T; theta)

``entry`` is one (cell, pin, arc, table_type, grid point) triple. We never
need a free per-entry baseline parameter: `log tau_source` is copied
directly from an already-known full-corner value ("anchor"), and only the
*shift* `Delta` is modeled/fit. This module only ever multiplies an
existing table by `exp(Delta)` -- see `Phase2Model.predict_table` -- so a
true-zero source point (the known-invalid rise_power/fall_power arcs,
docs/plan.md rule 3) maps to exactly zero regardless of `Delta`, with no
special-casing needed.

Model shape
-----------
Delay family (cell_rise, cell_fall, rise_transition, fall_transition)::

    g(V, T, P) = log(V) - alpha * log(V - Vth[P]) + c0[P] * T (+ offset[P])
    Delta_global = g(V_target, T_target, P_target) - g(V_source, T_source, P_source)

Power family (rise_power, fall_power)::

    g(V, T, P) = k * log(V) + c0[P] * T (+ offset[P])

Phase 2.5 changes (docs/phase2_review.md "修正方向", in priority order)
------------------------------------------------------------------------
Phase 2's single shared `Vth`/`alpha` pinned to the *top* of their box
bounds (Vth=0.5, alpha=2.0) for every table_type -- the architect review
found this meant the delivered scaling factors were "prior box edge x
lambda", not a data fit (docs/phase2_review.md item 4). The fixes below
are almost entirely about giving the voltage-shape fit enough degrees of
freedom to stop needing to hide the whole cross-process gap inside one
scalar Vth pinned at its ceiling:

1. **Per-process Vth** (`Vth_by_process`, one value each for ss/tt/ff)
   with the physical ordering constraint Vth_ss >= Vth_tt >= Vth_ff
   (threshold voltage rises as a process gets slower), fit jointly with
   a single shared `alpha` via `scipy.optimize.minimize(method="SLSQP")`
   -- box bounds per Vth (still `VTH_BOUNDS = (0.25, 0.5)`, narrower
   `ALPHA_BOUNDS = (1.1, 1.5)`) plus the two linear inequality
   constraints `Vth_tt - Vth_ff >= 0` and `Vth_ss - Vth_tt >= 0`. SLSQP
   was chosen over a softplus/increment reparameterization (the other
   option docs/phase2_review.md item 1 names) because it keeps every
   parameter's own box bound exact and legible, rather than folding the
   upper-bound headroom into a nonlinear increment transform.
2. **Explicit per-process offset** (`offset_by_process`): even with 3
   independent Vth's, the alpha-power law's *shape* is still shared
   across processes, so a residual multiplicative process gap can
   remain. `offset[P]` is a plain additive constant (in log space) fit
   from the same cross-process pairs, gauge-fixed at
   `offset["tt"] = 0` (see `_fit_process_offset` docstring for why tt is
   the reference). **This offset is applied only when a caller
   explicitly opts in via `use_process_offset=True`** on
   `predict_table`/`predict_corner` -- see the parameter's docstring.
   `scripts/phase2_predict.py` (the real deliverable) never passes it,
   so it is always `False` there; every one of the 10 real partial-corner
   targets uses a same-process anchor anyway (docs/phase2_results.md
   S5.1), so `offset[P_target] - offset[P_source] = offset[P] - offset[P]
   = 0` identically regardless of the flag -- the flag is a belt-and-
   braces guarantee, not just a mathematical coincidence.
   `scoring/loco.py::run_loco` *does* pass `use_process_offset=True`,
   since LOCO's cross-process folds (all of them but the two same-
   process pairs) are exactly the case this term exists to help.
3. Temperature term `c0[P]` is unchanged from Phase 2: a linear,
   per-process coefficient fit only from that process's own same-process
   pair (docs/phase2_results.md "溫度項必須 per-process"); missing
   processes safely default to `c0 = 0`.
4. Buck-corner shrinkage becomes **symmetric voltage-shift de-rating**:
   `SHRINK_LAMBDA` now defaults to 0.65 (was 0.85, buck-only) and applies
   to *both* directions -- boost corners (ss0p9, ff1p1, tt1p0) previously
   went through un-shrunk, which docs/phase2_review.md item 4 identified
   as the reason their delivered scaling factors (0.667x, 0.741x) sat
   well outside the physical mid-band estimate (0.83-0.88x, 0.88-0.93x).
   See `predict_table`'s `shrunk` computation.

Background carried over from Phase 2 (still true, unaffected by the
above): SS only has full-corner data at V=0.81, FF only at V=0.99, TT
only at V=0.90 -- no full corner varies V within one process, so all
voltage-shape signal (Vth/alpha) necessarily comes from cross-process
pairs (docs/plan.md: "沒有任何兩個 corner 只差電壓"; docs/phase2_results.md
S4.1). The alpha-power law's physical-prior shape (now split across 3
independent Vth's) is what lets those cross-process pairs stand in for
missing same-process voltage variation at all.

Sensitivity modulation (docs/plan.md Phase 2 item 3) and the power
family's independent V^k model are unchanged from Phase 2 -- see
docs/phase2_results.md for their design rationale.

Robustness (docs/plan.md Phase 2 item 5, all mandatory; item 4 above is
the only behavioral change in Phase 2.5):

1. Gain clip: `gain` is clamped to [GAIN_MIN, GAIN_MAX] (both positive),
   which also *structurally* protects voltage-monotonicity: the
   alpha-power law's V-term is monotonically decreasing in V for any
   `alpha >= 1` and `V > Vth` (both guaranteed by the fit bounds), so a
   strictly positive gain can only rescale that monotonic shape, never
   flip its sign.
2. Delta clip: `|Delta|` (after gain) is capped at
   `CLIP_RANGE_FACTOR * max(|observed cross-corner log-ratio|)` for that
   table_type, i.e. 1.5x the widest cross-corner swing actually seen
   among the training corners.
3. Symmetric voltage-shift shrinkage (item 4 above).
4. Explicit monotonicity enforcement on the delay family: after all of
   the above, if the target voltage is higher (lower) than the source's,
   the predicted value is clipped to be at most (at least) the source
   value, guaranteeing "V up -> delay down" holds pointwise even if the
   fitted shape's sign guarantee were ever violated by a modeling bug.

Call `predict_table(..., stats=some_dict)` to accumulate counts of how
often each of these mechanisms actually fired -- see
`docs/phase25_results.md` for the observed rates, fitted parameters, and
whether Vth/alpha are still pinned at their box bounds after the Phase
2.5 changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares, lsq_linear, minimize

from features.align import align_table_to_grid
from features.cellinfo import parse_cell_name
from features.corners import CornerMeta
from liberty.parser import LibertyFile, TableKey

DELAY_TABLE_TYPES = ("cell_rise", "cell_fall", "rise_transition", "fall_transition")
POWER_TABLE_TYPES = ("rise_power", "fall_power")

# ---- robustness knobs (docs/plan.md Phase 2 item 5, docs/phase2_review.md item 3) ----
GAIN_MIN = 0.2
GAIN_MAX = 5.0
CLIP_RANGE_FACTOR = 1.5
# Symmetric voltage-shift de-rating lambda (docs/phase2_review.md item 3:
# "lambda_buck -> 0.6-0.7, boost 對稱 de-rating"). Applied identically to
# both step-down (buck) and step-up (boost) voltage extrapolations --
# see predict_table's `shrunk` computation. Configurable per fit/predict
# call via Phase2Model.shrink_lambda / fit_phase2_model(shrink_lambda=...).
SHRINK_LAMBDA = 0.65

# alpha-power law parameter bounds (docs/plan.md Phase 2 item 1,
# docs/phase2_review.md item 1: alpha narrowed from [1.0, 2.0] to
# [1.1, 1.5] now that per-process Vth carries more of the cross-process
# gap, so alpha no longer needs as much headroom to compensate).
VTH_BOUNDS = (0.25, 0.5)
ALPHA_BOUNDS = (1.1, 1.5)

# Ascending Vth ordering constraint (docs/phase2_review.md item 1:
# "Vth_ss >= Vth_tt >= Vth_ff" -- threshold voltage rises as a process
# gets physically slower).
PROCESS_ORDER = ("ff", "tt", "ss")

# The process whose offset is gauge-fixed to 0 -- see
# `_fit_process_offset`'s docstring for why "tt" is used.
OFFSET_REFERENCE_PROCESS = "tt"

# Generous enough to fit the real T sensitivity in this data. Empirically
# (docs/phase2_results.md "溫度項擬合") SS needs c0 ~ -0.003 to reproduce
# its strong temperature inversion (~65% delay swing over -40..125C at
# V=0.81) while FF needs only c0 ~ -0.0002 (~4% swing at V=0.99) -- very
# different per-process magnitudes, which is exactly why the coefficient
# below is per-process (see the module docstring).
C0_BOUND = 0.05  # per degree C, per process

# power model: V^k bounds (k ~ 2 for a P ~ C*V^2*f style prior)
K_BOUNDS = (0.2, 6.0)
PC0_BOUND = 0.05  # per degree C, per process

# Stage-2 sensitivity-gain ridge weights (smaller = looser). The 3 smooth
# global terms (slew/load/strength) get a light prior; the per-family
# offsets (many groups, each with much less data) get a much stronger
# pull back to 0 (gain -> 1).
SMOOTH_TERM_RIDGE = 1.0
FAMILY_RIDGE = 30.0

# Tolerance for reporting a fitted scalar as "pinned at its box bound" --
# see `boundary_hit_report`.
BOUNDARY_TOL = 1e-6


def _c0_lookup(process, p: "ShapeParams"):
    """`process` may be a scalar string or an array of them (vectorized
    training-row fits use the array form)."""
    if isinstance(process, np.ndarray):
        return np.array([p.c0_by_process.get(proc, 0.0) for proc in process])
    return p.c0_by_process.get(process, 0.0)


def _Vth_lookup(process, p: "ShapeParams"):
    """Every delay-family fit always produces all 3 of PROCESS_ORDER's
    keys (see `_fit_voltage_shape_per_process`), so this should never
    hit the fallback in practice; the fallback (mean of whatever *is*
    fitted) only exists as a defensive default, mirroring `_c0_lookup`'s
    style, rather than crashing on an unexpected process label."""
    if isinstance(process, np.ndarray):
        return np.array([_Vth_lookup(proc, p) for proc in process])
    if process in p.Vth_by_process:
        return p.Vth_by_process[process]
    return float(np.mean(list(p.Vth_by_process.values())))


def _offset_lookup(process, p: "ShapeParams"):
    """Missing process -> 0 (no offset correction), same safe-default
    convention as `_c0_lookup`."""
    if isinstance(process, np.ndarray):
        return np.array([p.offset_by_process.get(proc, 0.0) for proc in process])
    return p.offset_by_process.get(process, 0.0)


def _g(V, T, process, p: "ShapeParams", *, use_offset: bool = False):
    c0 = _c0_lookup(process, p)
    if p.kind == "delay":
        Vth = _Vth_lookup(process, p)
        val = np.log(V) - p.alpha * np.log(V - Vth) + c0 * T
    else:
        val = p.k * np.log(V) + c0 * T
    if use_offset:
        val = val + _offset_lookup(process, p)
    return val


@dataclass
class ShapeParams:
    kind: str  # "delay" | "power"
    Vth_by_process: Dict[str, float]  # {} for power
    alpha: Optional[float]  # None for power
    k: Optional[float]  # None for delay
    c0_by_process: Dict[str, float]
    offset_by_process: Dict[str, float]  # docs/phase2_review.md item 1; LOCO-only, see module docstring
    b_slew: float
    b_load: float
    b_strength: float
    strength_center: float
    group_offset: Dict[str, float]
    clip_delta: float
    n_train_pairs: int
    fit_cost: float


def new_stats() -> dict:
    return {
        "n_points": 0,
        "n_calls": 0,
        "n_gain_clipped": 0,
        "n_delta_clipped": 0,
        "n_monotonic_violations": 0,
        "n_shrunk_calls": 0,
    }


@dataclass
class Phase2Model:
    params: Dict[str, ShapeParams]
    shrink_lambda: float = SHRINK_LAMBDA

    def predict_table(
        self,
        key: TableKey,
        source_values,
        source_meta: CornerMeta,
        target_meta: CornerMeta,
        source_index_2,
        target_index_2,
        *,
        stats: Optional[dict] = None,
        use_process_offset: bool = False,
    ) -> np.ndarray:
        """Predict one blank table's values.

        `use_process_offset` gates the explicit per-process offset term
        (docs/phase2_review.md item 1) -- default False, which is what
        `scripts/phase2_predict.py` (the real deliverable) always uses.
        `scoring.loco.run_loco` passes True. See the module docstring's
        "Explicit per-process offset" section for why this is safe to
        leave off for delivery (every real target has a same-process
        anchor, so the offset term is 0 either way) and necessary for
        LOCO's cross-process folds.
        """
        table_type = key[-1]
        p = self.params[table_type]
        values = align_table_to_grid(np.asarray(source_values, dtype=float), source_index_2, target_index_2)

        n_rows, n_cols = values.shape
        row_idx, col_idx = np.indices((n_rows, n_cols))
        slew = (row_idx - 3.0) / 3.0
        load = (col_idx - 3.0) / 3.0
        info = parse_cell_name(key[0])
        log_strength = np.log(info.drive_strength)

        delta_global = _g(
            target_meta.voltage, target_meta.temperature, target_meta.process, p, use_offset=use_process_offset
        ) - _g(
            source_meta.voltage, source_meta.temperature, source_meta.process, p, use_offset=use_process_offset
        )
        raw_gain = (
            1.0
            + p.b_slew * slew
            + p.b_load * load
            + p.b_strength * (log_strength - p.strength_center)
            + p.group_offset.get(info.family, 0.0)
        )
        gain = np.clip(raw_gain, GAIN_MIN, GAIN_MAX)

        delta_raw = delta_global * gain
        delta_clipped = np.clip(delta_raw, -p.clip_delta, p.clip_delta)

        # Symmetric voltage-shift de-rating (docs/phase2_review.md item 3):
        # any voltage change -- boost (V up) or buck (V down) -- pulls the
        # whole Delta (V and T terms combined) toward 0 by the same
        # `shrink_lambda`, not just the buck direction as in Phase 2. A
        # pure-temperature transfer (V_target == V_source, only possible
        # inside LOCO's same-process folds) is left un-shrunk: that is not
        # the risky +-10% VDD extrapolation this mechanism exists for.
        shrunk = not np.isclose(target_meta.voltage, source_meta.voltage, atol=1e-12)
        delta = delta_clipped * self.shrink_lambda if shrunk else delta_clipped

        predicted = values * np.exp(delta)
        predicted = np.where(values == 0.0, 0.0, predicted)

        n_mono_fixes = 0
        if table_type in DELAY_TABLE_TYPES:
            before = predicted
            predicted = _enforce_voltage_monotonic(predicted, values, target_meta.voltage, source_meta.voltage)
            n_mono_fixes = int(np.sum(before != predicted))

        if stats is not None:
            s = stats.setdefault(table_type, new_stats())
            s["n_points"] += predicted.size
            s["n_calls"] += 1
            s["n_gain_clipped"] += int(np.sum(raw_gain != gain))
            s["n_delta_clipped"] += int(np.sum(delta_raw != delta_clipped))
            s["n_monotonic_violations"] += n_mono_fixes
            s["n_shrunk_calls"] += int(shrunk)

        if not np.isfinite(predicted).all():
            raise ValueError(f"phase2 prediction for key {key!r} produced non-finite values")
        return predicted


def _enforce_voltage_monotonic(predicted, source, v_target, v_source):
    """V up -> delay down (docs/plan.md Phase 2 item 5). Clip the
    predicted delay-family value against the source (anchor) value in
    the direction implied by the voltage change."""
    if v_target > v_source + 1e-12:
        return np.minimum(predicted, source)
    if v_target < v_source - 1e-12:
        return np.maximum(predicted, source)
    return predicted


def _build_training_rows(libs: Mapping[CornerMeta, LibertyFile], table_type: str) -> dict:
    """Cross-corner log-ratio training pairs for one table_type, pooled
    over every pair of corners in `libs` and every (cell, arc, grid
    point). index_2 is aligned via features.align before differencing
    (a documented no-op in this dataset, see features/align.py)."""
    metas = list(libs)
    r_chunks, Vi_chunks, Vj_chunks, Ti_chunks, Tj_chunks = [], [], [], [], []
    Pi_chunks, Pj_chunks = [], []
    slew_chunks, load_chunks, family_chunks, strength_chunks = [], [], [], []

    for a in range(len(metas)):
        for b in range(a + 1, len(metas)):
            mi, mj = metas[a], metas[b]
            lib_i, lib_j = libs[mi], libs[mj]
            for key, ti in lib_i.tables_by_key.items():
                if key[-1] != table_type or ti.values is None:
                    continue
                tj = lib_j.tables_by_key.get(key)
                if tj is None or tj.values is None:
                    continue
                vj_aligned = align_table_to_grid(tj.values, tj.index_2, ti.index_2)
                vi = ti.values
                mask = (vi > 0) & (vj_aligned > 0)
                if not mask.any():
                    continue
                rows, cols = np.nonzero(mask)
                n = rows.size
                info = parse_cell_name(key[0])

                r_chunks.append(np.log(vj_aligned[mask]) - np.log(vi[mask]))
                Vi_chunks.append(np.full(n, mi.voltage))
                Vj_chunks.append(np.full(n, mj.voltage))
                Ti_chunks.append(np.full(n, mi.temperature))
                Tj_chunks.append(np.full(n, mj.temperature))
                Pi_chunks.append(np.full(n, mi.process, dtype=object))
                Pj_chunks.append(np.full(n, mj.process, dtype=object))
                slew_chunks.append((rows - 3.0) / 3.0)
                load_chunks.append((cols - 3.0) / 3.0)
                family_chunks.append(np.full(n, info.family, dtype=object))
                strength_chunks.append(np.full(n, np.log(info.drive_strength)))

    if not r_chunks:
        raise ValueError(f"no usable cross-corner training pairs for table_type={table_type!r}")

    return dict(
        r=np.concatenate(r_chunks),
        V_i=np.concatenate(Vi_chunks),
        V_j=np.concatenate(Vj_chunks),
        T_i=np.concatenate(Ti_chunks),
        T_j=np.concatenate(Tj_chunks),
        P_i=np.concatenate(Pi_chunks),
        P_j=np.concatenate(Pj_chunks),
        slew=np.concatenate(slew_chunks),
        load=np.concatenate(load_chunks),
        family=np.concatenate(family_chunks),
        strength=np.concatenate(strength_chunks),
    )


def _fit_temperature_terms_by_process(rows: dict, bound: float) -> Dict[str, float]:
    """Fit one linear temperature coefficient `c0[process]` per process,
    each from *that process's own* same-process pair(s) only (docs/plan.md
    Phase 2 item 2: "SS/FF 各有 -40/125 兩點，溫度敏感度可直接從資料辨識").
    A process with no same-process pair in `rows` (either because it only
    ever has one corner -- tt -- or because a LOCO fold held out one of
    its two corners, destroying its only pair) is simply absent from the
    returned dict; `_g` treats a missing process as `c0 = 0` (no
    correction). Unchanged from Phase 2 -- see docs/phase2_results.md
    "溫度項必須 per-process" for the full rationale.
    """
    same_v = np.isclose(rows["V_i"], rows["V_j"])
    coeffs: Dict[str, float] = {}
    for process in sorted(set(rows["P_i"][same_v]) | set(rows["P_j"][same_v])):
        mask = same_v & (rows["P_i"] == process) & (rows["P_j"] == process)
        if not mask.any():
            continue
        r = rows["r"][mask]
        dT = rows["T_j"][mask] - rows["T_i"][mask]
        result = lsq_linear(dT.reshape(-1, 1), r, bounds=([-bound], [bound]))
        coeffs[process] = float(result.x[0])
    return coeffs


def _voltage_term_delay(rows: dict, Vth_by_process: Dict[str, float], alpha: float) -> np.ndarray:
    Vth_i = np.array([Vth_by_process[p] for p in rows["P_i"]])
    Vth_j = np.array([Vth_by_process[p] for p in rows["P_j"]])
    return (np.log(rows["V_j"]) - alpha * np.log(rows["V_j"] - Vth_j)) - (
        np.log(rows["V_i"]) - alpha * np.log(rows["V_i"] - Vth_i)
    )


def _fit_voltage_shape_per_process(
    rows: dict,
    c0_by_process: Dict[str, float],
    *,
    vth_bounds: Tuple[float, float] = VTH_BOUNDS,
    alpha_bounds: Tuple[float, float] = ALPHA_BOUNDS,
):
    """Fit (Vth_ff, Vth_tt, Vth_ss, alpha) jointly against the residual
    left after subtracting the (already-identified, see
    `_fit_temperature_terms_by_process`) per-process temperature term,
    using every available pair -- same-process pairs contribute nothing
    here (V_i == V_j collapses the alpha-power difference to 0) but
    cross-process pairs are what actually carries voltage-shape signal,
    since no full corner varies V within a process (docs/plan.md: "沒有
    任何兩個 corner 只差電壓") -- see the module docstring's §4.1-style
    background.

    docs/phase2_review.md item 1: Vth is now **per-process** (was a
    single shared scalar in Phase 2) subject to the physical ordering
    `Vth_ff <= Vth_tt <= Vth_ss`, fit jointly with a single shared
    `alpha` via `scipy.optimize.minimize(method="SLSQP")`, which natively
    supports the box bounds on all 4 parameters *and* the two linear
    inequality constraints in one solve (no softplus/increment
    reparameterization needed).

    If a process is entirely absent from `rows` (only possible for "tt"
    -- the LOCO fold that holds out tt0p9v25c has zero tt rows in its
    training set, since tt is the only process with just one full-corner
    voltage point among all 5), that process's Vth has zero gradient
    contribution and SLSQP simply leaves it at its `x0` starting value
    (the bounds midpoint) -- a reasonable, ordering-consistent default,
    analogous to the `c0`/`offset` "safe default when missing" pattern
    used elsewhere in this module. This only matters for LOCO's tt fold:
    the real 5-corner delivery fit always has data for all 3 processes.
    """
    c0_i = np.array([c0_by_process.get(p, 0.0) for p in rows["P_i"]])
    c0_j = np.array([c0_by_process.get(p, 0.0) for p in rows["P_j"]])
    t_term = c0_j * rows["T_j"] - c0_i * rows["T_i"]
    target = rows["r"] - t_term

    def unpack(theta):
        return {"ff": theta[0], "tt": theta[1], "ss": theta[2]}, theta[3]

    def resid(theta):
        Vth, alpha = unpack(theta)
        Vth_i = np.array([Vth[p] for p in rows["P_i"]])
        Vth_j = np.array([Vth[p] for p in rows["P_j"]])
        pred = (np.log(rows["V_j"]) - alpha * np.log(rows["V_j"] - Vth_j)) - (
            np.log(rows["V_i"]) - alpha * np.log(rows["V_i"] - Vth_i)
        )
        return pred - target

    # SLSQP's internal convergence/line-search behavior is sensitive to the
    # objective's absolute magnitude (unlike least_squares, which works
    # with raw per-point residual vectors). This dataset has ~1e4-1e5
    # training rows, so the *unnormalized* sum-of-squares cost sits around
    # 1e5-1e6, which was empirically found to make SLSQP declare
    # "converged" after a handful of iterations without actually moving
    # away from `x0` (its internal QP step-size heuristics are tuned
    # around an O(1) objective). Dividing by `n` (mean squared residual,
    # same fix as several other floating point issues in this class of
    # problem) restores normal convergence -- verified by checking the
    # fit is a real minimum (positive-cost-decrease from `x0`, and stable
    # across many random restarts within the constraint region).
    n = target.size

    def cost(theta):
        r = resid(theta)
        return 0.5 * float(np.dot(r, r)) / n

    mid_v = 0.5 * (vth_bounds[0] + vth_bounds[1])
    mid_a = 0.5 * (alpha_bounds[0] + alpha_bounds[1])
    x0 = [mid_v, mid_v, mid_v, mid_a]
    bnds = [vth_bounds, vth_bounds, vth_bounds, alpha_bounds]
    constraints = [
        {"type": "ineq", "fun": lambda th: th[1] - th[0]},  # Vth_tt - Vth_ff >= 0
        {"type": "ineq", "fun": lambda th: th[2] - th[1]},  # Vth_ss - Vth_tt >= 0
    ]
    result = minimize(
        cost, x0, method="SLSQP", bounds=bnds, constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-16},
    )
    if not result.success:
        # SLSQP occasionally reports a non-success status (observed:
        # "Inequality constraints incompatible") even though it has, in
        # fact, landed on a feasible point with lower cost than the
        # starting guess -- its internal line-search heuristics are
        # tuned around the default bound geometry and can misfire near a
        # widened bound (seen when vth_bounds/alpha_bounds are perturbed
        # away from the module defaults, e.g. scoring/ensemble.py's
        # bounds-perturbation sweep) without the final iterate actually
        # being invalid. Retry once from a different starting point
        # before accepting or giving up.
        retry_x0 = [vth_bounds[0], mid_v, vth_bounds[1], mid_a]
        retry = minimize(
            cost, retry_x0, method="SLSQP", bounds=bnds, constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-16},
        )
        if retry.success:
            result = retry
        else:
            ordering_ok = result.x[1] - result.x[0] >= -1e-6 and result.x[2] - result.x[1] >= -1e-6
            bounds_ok = all(lo - 1e-9 <= v <= hi + 1e-9 for v, (lo, hi) in zip(result.x, bnds))
            improved = cost(result.x) <= cost(x0) + 1e-9
            if not (ordering_ok and bounds_ok and improved):
                raise RuntimeError(f"per-process Vth/alpha fit did not converge: {result.message}")
    # Report `fit_cost` in the same (raw, unnormalized) convention
    # `scipy.optimize.least_squares`'s `.cost` uses elsewhere in this
    # module (0.5 * sum(residual**2)), rather than the mean-scaled value
    # actually driving the optimizer above.
    result.fun = 0.5 * float(np.dot(resid(result.x), resid(result.x)))
    Vth_by_process, alpha = unpack(result.x)
    Vth_by_process = {k: float(v) for k, v in Vth_by_process.items()}
    return Vth_by_process, float(alpha), result


def _fit_process_offset(
    rows: dict, target: np.ndarray, *, reference: str = OFFSET_REFERENCE_PROCESS
) -> Dict[str, float]:
    """Solve for one additive per-process offset (docs/phase2_review.md
    item 1, "explicit process offset term to absorb residual process
    gaps") from `target` -- the residual left after subtracting the
    voltage-shape and temperature terms -- gauge-fixed at
    `offset[reference] = 0` (offsets are only identifiable as pairwise
    differences: nothing in the training data separates an absolute
    per-process level from the shared `log tau_source` baseline, so one
    process must be pinned as the zero point).

    `reference="tt"` is a deliberate choice, not an arbitrary one: tt is
    the only process that can be *entirely absent* from a fold's
    training rows (the LOCO fold holding out tt0p9v25c -- tt's only
    full-corner data point -- leaves zero tt rows). Fixing the reference
    process's offset to 0 needs no data for it at all, so `offset["tt"]`
    is trivially well-defined (0) in that fold, while `offset["ss"]` and
    `offset["ff"]` remain solvable from their own direct ss-ff cross
    pairs regardless of whether tt data exists. Choosing "ss" or "ff" as
    the reference instead would have left the *other* one undefined
    whenever tt-anchored pairs were needed to solve for it in a fold
    with a different missing process -- "tt" is the only choice that is
    always safe.
    """
    others = [p for p in PROCESS_ORDER if p != reference]
    n = target.size
    X = np.zeros((n, len(others)))
    for col, proc in enumerate(others):
        X[:, col] += (rows["P_j"] == proc).astype(float)
        X[:, col] -= (rows["P_i"] == proc).astype(float)
    coef, *_ = np.linalg.lstsq(X, target, rcond=None)
    offset = {reference: 0.0}
    offset.update({proc: float(c) for proc, c in zip(others, coef)})
    return offset


def _fit_power_shape(rows: dict):
    c0_by_process = _fit_temperature_terms_by_process(rows, PC0_BOUND)
    c0_i = np.array([c0_by_process.get(p, 0.0) for p in rows["P_i"]])
    c0_j = np.array([c0_by_process.get(p, 0.0) for p in rows["P_j"]])
    t_term = c0_j * rows["T_j"] - c0_i * rows["T_i"]
    target_for_k = rows["r"] - t_term

    def resid(theta):
        (k,) = theta
        pred = k * (np.log(rows["V_j"]) - np.log(rows["V_i"]))
        return pred - target_for_k

    x0 = [2.0]
    result = least_squares(resid, x0, bounds=([K_BOUNDS[0]], [K_BOUNDS[1]]))
    (k,) = result.x

    v_term = k * (np.log(rows["V_j"]) - np.log(rows["V_i"]))
    residual = rows["r"] - t_term - v_term
    offset_by_process = _fit_process_offset(rows, residual)

    return float(k), c0_by_process, offset_by_process, result


def _fit_sensitivity_gain(rows: dict, shape_params: ShapeParams, families: List[str]):
    """Closed-form ridge regression for the delay-family gain modulation
    (docs/plan.md Phase 2 item 3). Solves for `b_slew, b_load,
    b_strength, group_offset[family]` such that
    `r - delta_global ~= delta_global * (b_slew*slew + b_load*load +
    b_strength*strength_centered + group_offset[family])`, i.e.
    `gain = 1 + (...)`, ridge-shrunk toward the `gain == 1` baseline.

    Fit *only* on same-process rows (V_i == V_j, so `delta_global` is
    pure temperature signal with no V-shape component, and the explicit
    process offset is inert here too since it cancels for a same-process
    pair regardless of the `use_offset` flag). Unchanged from Phase 2 --
    see docs/phase2_results.md "gain 修正必須只用 same-process 資料".
    """
    same_v = np.isclose(rows["V_i"], rows["V_j"])
    delta_global_full = _g(rows["V_j"], rows["T_j"], rows["P_j"], shape_params) - _g(
        rows["V_i"], rows["T_i"], rows["P_i"], shape_params
    )

    delta_global = delta_global_full[same_v]
    y = rows["r"][same_v] - delta_global
    slew = rows["slew"][same_v]
    load = rows["load"][same_v]
    strength = rows["strength"][same_v]
    family = rows["family"][same_v]
    n = y.size

    if n == 0:
        return 0.0, 0.0, 0.0, float(rows["strength"].mean()), {}

    strength_center = float(strength.mean())
    strength_centered = strength - strength_center

    fam_index = {f: i for i, f in enumerate(families)}
    fam_col = np.array([fam_index.get(f, -1) for f in family])

    ncols = 3 + len(families)
    X = np.zeros((n, ncols))
    X[:, 0] = delta_global * slew
    X[:, 1] = delta_global * load
    X[:, 2] = delta_global * strength_centered
    valid = fam_col >= 0
    X[valid, 3 + fam_col[valid]] = delta_global[valid]

    reg_diag = np.concatenate([np.full(3, SMOOTH_TERM_RIDGE), np.full(len(families), FAMILY_RIDGE)])
    X_aug = np.vstack([X, np.diag(reg_diag)])
    y_aug = np.concatenate([y, np.zeros(ncols)])
    coef, *_ = np.linalg.lstsq(X_aug, y_aug, rcond=None)

    b_slew, b_load, b_strength = coef[0], coef[1], coef[2]
    group_offset = {f: float(coef[3 + i]) for i, f in enumerate(families)}
    return b_slew, b_load, b_strength, strength_center, group_offset


def fit_phase2_model(
    libs: Mapping[CornerMeta, LibertyFile],
    *,
    shrink_lambda: float = SHRINK_LAMBDA,
    vth_bounds: Tuple[float, float] = VTH_BOUNDS,
    alpha_bounds: Tuple[float, float] = ALPHA_BOUNDS,
) -> Phase2Model:
    """Fit a Phase2Model from >= 2 full (fully-populated) corner
    LibertyFiles, keyed by their CornerMeta.

    `vth_bounds`/`alpha_bounds` default to the module constants but can
    be overridden per call -- used by the Phase 2.5 bounds-endpoint
    perturbation ensemble (docs/phase2_review.md item 4,
    `scoring/ensemble.py`) to probe how sensitive the delivered scaling
    factor is to exactly where the physical-prior box sits.
    """
    if len(libs) < 2:
        raise ValueError("need at least 2 full corners to fit a Phase 2 model")

    all_families = sorted({parse_cell_name(name).family for lib in libs.values() for name in lib.cells})

    params: Dict[str, ShapeParams] = {}

    for table_type in DELAY_TABLE_TYPES:
        rows = _build_training_rows(libs, table_type)
        c0_by_process = _fit_temperature_terms_by_process(rows, C0_BOUND)
        Vth_by_process, alpha, fit_result = _fit_voltage_shape_per_process(
            rows, c0_by_process, vth_bounds=vth_bounds, alpha_bounds=alpha_bounds
        )
        v_term = _voltage_term_delay(rows, Vth_by_process, alpha)
        c0_i = np.array([c0_by_process.get(p, 0.0) for p in rows["P_i"]])
        c0_j = np.array([c0_by_process.get(p, 0.0) for p in rows["P_j"]])
        t_term = c0_j * rows["T_j"] - c0_i * rows["T_i"]
        offset_by_process = _fit_process_offset(rows, rows["r"] - t_term - v_term)

        shape = ShapeParams(
            kind="delay", Vth_by_process=Vth_by_process, alpha=alpha, k=None,
            c0_by_process=c0_by_process, offset_by_process=offset_by_process,
            b_slew=0.0, b_load=0.0, b_strength=0.0, strength_center=0.0, group_offset={},
            clip_delta=np.inf, n_train_pairs=int(rows["r"].size), fit_cost=float(fit_result.fun),
        )
        b_slew, b_load, b_strength, strength_center, group_offset = _fit_sensitivity_gain(
            rows, shape, all_families
        )
        clip_delta = CLIP_RANGE_FACTOR * float(np.max(np.abs(rows["r"])))
        params[table_type] = ShapeParams(
            kind="delay", Vth_by_process=Vth_by_process, alpha=alpha, k=None,
            c0_by_process=c0_by_process, offset_by_process=offset_by_process,
            b_slew=b_slew, b_load=b_load, b_strength=b_strength, strength_center=strength_center,
            group_offset=group_offset, clip_delta=clip_delta,
            n_train_pairs=int(rows["r"].size), fit_cost=float(fit_result.fun),
        )

    for table_type in POWER_TABLE_TYPES:
        rows = _build_training_rows(libs, table_type)
        k, c0_by_process, offset_by_process, fit_result = _fit_power_shape(rows)
        clip_delta = CLIP_RANGE_FACTOR * float(np.max(np.abs(rows["r"])))
        params[table_type] = ShapeParams(
            kind="power", Vth_by_process={}, alpha=None, k=k,
            c0_by_process=c0_by_process, offset_by_process=offset_by_process,
            b_slew=0.0, b_load=0.0, b_strength=0.0, strength_center=0.0, group_offset={},
            clip_delta=clip_delta, n_train_pairs=int(rows["r"].size), fit_cost=float(fit_result.cost),
        )

    return Phase2Model(params=params, shrink_lambda=shrink_lambda)


def _at_bound(value: float, bounds: Tuple[float, float], tol: float = BOUNDARY_TOL) -> Optional[str]:
    lo, hi = bounds
    if value <= lo + tol:
        return "lower"
    if value >= hi - tol:
        return "upper"
    return None


def boundary_hit_report(model: Phase2Model) -> Dict[str, dict]:
    """For each table_type, report which fitted shape parameters are
    pinned at their box bounds (docs/phase2_review.md item 1: "擬合後回
    報：各 process 的 Vth、alpha 是否仍頂界（若仍頂界，如實記錄並回報，
    不要放寬盒子硬讓它不頂界）"). Returns, per table_type:

    - delay: `{"alpha_at_bound": "lower"|"upper"|None,
               "Vth_at_bound": {process: "lower"|"upper"|None}}`
    - power: `{"k_at_bound": "lower"|"upper"|None}`
    """
    report: Dict[str, dict] = {}
    for table_type, p in model.params.items():
        if p.kind == "delay":
            report[table_type] = {
                "alpha_at_bound": _at_bound(p.alpha, ALPHA_BOUNDS),
                "Vth_at_bound": {proc: _at_bound(v, VTH_BOUNDS) for proc, v in p.Vth_by_process.items()},
            }
        else:
            report[table_type] = {"k_at_bound": _at_bound(p.k, K_BOUNDS)}
    return report


def select_anchors(target: CornerMeta, available: Mapping[CornerMeta, LibertyFile]) -> List[CornerMeta]:
    """Prefer, in order: (1) a same-process, same-temperature anchor --
    Delta_T == 0, so this is a pure (and best-supported) voltage-only
    transfer, which is what every one of the 10 real partial-corner
    targets has (docs/phase2_results.md); (2) any same-process anchor
    (Phase 1 finding: cross-process direct-copy scores ~13-18 vs ~69-85
    same-process, docs/phase1_results.md section 6) -- needed only inside
    LOCO folds that hold out one of SS's/FF's two temperature points;
    (3) every available corner (blended via geometric mean, see
    `predict_corner`) when the target's process has no full corner at all
    -- only the tt0p9v25c LOCO fold hits this, since tt has just one
    temperature point among the full corners.

    Tier (1) is deliberately exclusive of tier (2): blending in a
    same-process-but-different-temperature anchor would needlessly import
    that anchor's temperature-extrapolation noise into an otherwise exact
    (Delta_T == 0) prediction."""
    exact = [m for m in available if m.process == target.process and m.temperature == target.temperature]
    if exact:
        return exact
    same_process = [m for m in available if m.process == target.process]
    return same_process if same_process else list(available)


def _geomean(arrays: List[np.ndarray]) -> np.ndarray:
    """Blend several anchors' predictions for the same point via a
    geometric mean of magnitudes (stable across the many orders of
    magnitude these tables span), with sign handled separately: internal
    power tables legitimately contain negative entries in this dataset
    (docs/phase2_results.md), so this cannot just assume positivity like
    a plain log-space average would."""
    stacked = np.stack(arrays, axis=0)
    nonzero = stacked != 0
    count = nonzero.sum(axis=0)
    abs_log_sum = np.where(nonzero, np.log(np.abs(np.where(nonzero, stacked, 1.0))), 0.0).sum(axis=0)
    mean_abs_log = np.divide(abs_log_sum, count, out=np.zeros_like(abs_log_sum), where=count > 0)
    magnitude = np.exp(mean_abs_log)
    sign = np.sign(stacked.sum(axis=0))
    result = magnitude * sign
    return np.where(count == 0, 0.0, result)


def predict_corner(
    model: Phase2Model,
    target_lib: LibertyFile,
    target_meta: CornerMeta,
    anchors: List[CornerMeta],
    anchor_libs: Mapping[CornerMeta, LibertyFile],
    *,
    keys: Optional[Iterable[TableKey]] = None,
    stats: Optional[dict] = None,
    use_process_offset: bool = False,
) -> Dict[TableKey, np.ndarray]:
    """Predict every table named in `keys` (default: every blank table in
    `target_lib`) for `target_meta`, blending across `anchors` (>1 anchor
    only occurs for the tt0p9v25c LOCO fold -- see `select_anchors`) via
    a per-point geometric mean in linear space.

    `use_process_offset` is forwarded to `Phase2Model.predict_table` --
    see its docstring and the module docstring's "Explicit per-process
    offset" section. Default False (delivery path); `scoring.loco.run_loco`
    passes True.
    """
    if keys is None:
        keys = [t.key for t in target_lib.tables if t.is_blank]

    predictions: Dict[TableKey, np.ndarray] = {}
    for key in keys:
        target_table = target_lib.tables_by_key[key]
        dst_index_2 = target_table.index_2
        preds = []
        for anchor_meta in anchors:
            anchor_lib = anchor_libs[anchor_meta]
            src_table = anchor_lib.tables_by_key.get(key)
            if src_table is None or src_table.values is None:
                continue
            preds.append(
                model.predict_table(
                    key, src_table.values, anchor_meta, target_meta,
                    src_table.index_2, dst_index_2, stats=stats,
                    use_process_offset=use_process_offset,
                )
            )
        if not preds:
            raise KeyError(f"no anchor supplies usable source values for blank table {key!r}")
        predictions[key] = preds[0] if len(preds) == 1 else _geomean(preds)
    return predictions
