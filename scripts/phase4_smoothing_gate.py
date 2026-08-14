"""Ceiling for the "cross-corner smoothing" idea (docs/round_20260810.md
section 6, the last direction still listed as untried).

The 10 target corners are predicted independently, one model run each. If a
grid point's relative error is *rough* along the corner axis it looks like
independent noise, and averaging along that axis should cancel it. This
measures whether it actually can, with no retraining, from an existing
error dump.

Score = 100 - 100*sqrt(mean(min(1, |y-yhat|/|y|)^2))  -- see scoring/scorer.py.

Let q_c = (yhat_c - y_c)/y_c be the signed relative error of one grid point
at corner c. A linear smoother projects the prediction onto a smooth basis B
in corner space, giving

    new residual = P_B q  -  (the label's own out-of-basis component)
                   ^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                   the gain            the bias it pays for it

Every "ceiling" printed here keeps only the first term, i.e. it assumes the
smoother reproduces the truth for free. That is the generous half of the
trade; the bias half is measured separately at the end so the two can be
compared.

The global-constant row is the *best observed* basis, not a proven bound on
the family. In plain L2 a coarser basis always shrinks the residual more, so
the constant would dominate -- but min(1,x)^2 is convex then flat, i.e. not
convex, and that argument does not survive the cap. What is measured: six
bases x two topologies, and the global constant wins all twelve. Treat it as
a well-probed empirical ceiling, not a proof.

S6 (global window, process dummies) is algebraically the same projection as
S2 (per-process constant) reached by a different code path, so the two rows
printing identical numbers is a standing check on group_projection.

Three things make the honest answer much smaller than the seductive one:

  * min(1,x)^2 is convex then flat, so averaging is not guaranteed to help.
    Rows containing a saturated (clipped) sign-flip corner get *worse* when
    smoothed -- the catastrophic corner contaminates its healthy siblings.
    The "smoothable points only" block excludes flip/near-zero corners from
    the average itself, which is the strongest honest form of the idea.
  * the error mass lives where smoothing cannot go: 84% of it is in the
    sign-flip and near-zero subgroups, where the log-space ratio is
    undefined (negative) or unstable (denominator -> 0).
  * only 10 target corners exist, as 3 process groups of at most 2 voltages
    x 2 temperatures. A basis rich enough to not bias the strongly
    nonlinear V-dependence ({1,V,T,V*T} on 4 points) is the identity, i.e.
    no smoothing at all.

Usage:
    python3 scripts/phase4_smoothing_gate.py DUMP.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from features.corners import parse_corner_filename
from models.phase4_features import STAGE_TOPOLOGIES
from scoring.audits import (
    SUBGROUP_BULK_FP,
    SUBGROUP_FLIP,
    SUBGROUP_NEAR_ZERO,
    SUBGROUP_OTHER,
    assign_subgroups,
)
from scoring.scorer import point_errors, score_from_errors


def corner_meta(short_name: str):
    """CornerMeta for a bare corner label, via the project's own parser
    (which expects a full lib filename)."""
    return parse_corner_filename(f"lib1_{short_name}_alpha_100.lib")


def score(e: np.ndarray) -> float:
    return score_from_errors(e.ravel())


def align(d) -> tuple:
    """Reshape a flat dump into (points x corners) matrices.

    The dump stores no grid coordinate, only (corner, cell, table_type), so
    row i of one corner block and row i of another are assumed to be the same
    (cell, pin, arc, row, col). Two independent checks stand behind that:

    1. every corner block lists the same (cell, table_type) sequence -- this
       is necessary but NOT sufficient, since one (cell, table_type) spans
       many arcs x 49 grid points and a permutation inside such a run would
       pass;
    2. the stage topology says which targets share a nearest anchor; each
       such pair MUST then carry a bitwise-identical `nearest_anchor`
       column. This runs at full grid-point resolution and does catch a
       permutation inside a run, which check 1 cannot. The expected pairs
       come from the topology rather than from the data, so a misalignment
       cannot hide by producing fewer matching pairs.

    Check 2 covers the anchor column only. A permutation that moved y_true
    and y_pred while leaving `nearest_anchor` alone would pass both checks:
    the dump's anchor column and its value columns are written by separate
    code paths. What rules that out is upstream, not here -- extract_raw_values
    (models/phase4_features.py) asserts each key owns one contiguous 49-point
    run, and all 10 corners of a run share one ds_val key order. So alignment
    is verified here for the anchor column and inherited by contract for the
    values; if that contract ever changes, this script will not notice.
    """
    corner = d["corner"]
    corners = sorted(set(corner.tolist()))
    idx = {c: np.flatnonzero(corner == c) for c in corners}
    n0 = idx[corners[0]].size

    def key(sel):
        return np.char.add(np.char.add(d["cell"][sel], "|"), d["table_type"][sel])

    ref = key(idx[corners[0]])
    for c in corners[1:]:
        if idx[c].size != n0:
            raise SystemExit(f"corner {c} has {idx[c].size} rows, expected {n0}")
        if not np.array_equal(key(idx[c]), ref):
            raise SystemExit(f"corner {c} row order differs from {corners[0]}")

    def stack(name):
        return np.stack([d[name][idx[c]] for c in corners], axis=1)

    anchors = stack("nearest_anchor")
    stage = str(d["meta_stage"][0])
    if stage not in STAGE_TOPOLOGIES:
        # phase4_final_validate.py labels an --anchors/--targets run "custom";
        # there is no topology to look the anchor-sharing pairs up in.
        raise SystemExit(
            f"dump was produced with stage {stage!r}, which has no registered "
            f"topology (known: {sorted(STAGE_TOPOLOGIES)}). The grid-point "
            "alignment check needs one.")
    topo = STAGE_TOPOLOGIES[stage]
    if sorted(topo.target_names) != corners:
        # e.g. a --corners smoke-test subset. Not corruption, but the
        # per-process bases below need the full 10-corner window to mean
        # anything, so this is refused rather than silently narrowed.
        raise SystemExit(
            f"dump holds {len(corners)} targets {corners}, but stage {stage!r} "
            f"has {len(topo.target_names)}: {sorted(topo.target_names)}. This "
            "gate needs the complete target set.")
    expected = [(i, j)
                for i in range(len(corners)) for j in range(i + 1, len(corners))
                if topo.nearest_anchor_by_target[corners[i]]
                == topo.nearest_anchor_by_target[corners[j]]]
    for i, j in expected:
        if not np.array_equal(anchors[:, i], anchors[:, j]):
            n_bad = int((anchors[:, i] != anchors[:, j]).sum())
            raise SystemExit(
                f"{corners[i]} and {corners[j]} share anchor "
                f"{topo.nearest_anchor_by_target[corners[i]]} but their anchor "
                f"columns differ at {n_bad}/{n0} grid points -- rows are "
                "misaligned across corner blocks")
    if not expected:
        print("WARNING: topology has no two targets sharing an anchor; "
              "grid-point alignment rests on the (cell, table_type) check alone")
    else:
        print(f"alignment: {len(expected)} anchor-sharing corner pairs agree "
              f"bitwise on all {n0} grid points")

    return corners, stack("y_true"), stack("y_pred"), anchors, \
        stack("table_type"), n0


def group_projection(q: np.ndarray, groups: np.ndarray, cols: list) -> np.ndarray:
    """Least-squares projection of each row of q onto `cols`, done
    independently within each group of corners."""
    out = np.empty_like(q)
    for g in sorted(set(groups.tolist())):
        m = groups == g
        X = np.column_stack([c[m] for c in cols])
        # drop columns that are constant inside this group (e.g. temperature
        # within tt, which has a single temperature) to keep pinv well posed
        keep = [i for i in range(X.shape[1]) if i == 0 or np.ptp(X[:, i]) > 0]
        X = X[:, keep]
        out[:, m] = q[:, m] @ (X @ np.linalg.pinv(X)).T
    # pinv's internal SVD emits benign over/underflow warnings on these
    # badly-scaled design matrices; the guarantee that matters is that the
    # fitted values stay finite, so assert it instead of trusting it.
    if not np.isfinite(out).all():
        raise SystemExit("projection produced non-finite values")
    return out


def masked_group_mean(q: np.ndarray, groups: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-row group mean of q using only the corners selected by `w`."""
    out = np.zeros_like(q)
    for g in sorted(set(groups.tolist())):
        m = groups == g
        num = (q[:, m] * w[:, m]).sum(axis=1, keepdims=True)
        den = w[:, m].sum(axis=1, keepdims=True)
        out[:, m] = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    # same invariant as group_projection: this path feeds the headline number,
    # and a NaN here would propagate through np.minimum straight into the
    # pooled score rather than being capped at 1.
    if not np.isfinite(out).all():
        raise SystemExit("masked group mean produced non-finite values")
    return out


def main(dump_path: str) -> None:
    d = np.load(dump_path, allow_pickle=True)
    corners, T, P, A, TT, npts = align(d)
    C = len(corners)
    print(f"dump={Path(dump_path).name}  config={d['meta_config'][0]} "
          f"stage={d['meta_stage'][0]} seeds={d['meta_n_seeds'][0]}")
    print(f"{npts} grid points x {C} corners, row order verified identical")
    print(f"targets: {', '.join(corners)}")

    meta = [corner_meta(c) for c in corners]
    proc = np.array([m.process for m in meta])
    volt = np.array([m.voltage for m in meta])
    temp = np.array([m.temperature for m in meta])
    one = np.ones(C)

    e0 = point_errors(T.ravel(), P.ravel()).reshape(T.shape)
    base = score(e0)
    print(f"\nbaseline pooled (recomputed from dump): {base:.4f}")

    # y_true == 0 has no relative error to smooth (the scorer defines those
    # points as 0 or 1 outright), so they are frozen at baseline on output.
    #
    # They are filled with q = 0 rather than dropped, and that value is not a
    # placeholder: a zero true value here comes with a zero anchor, so the
    # reconstruction predicts 0 too and the point is already exact (its e0 is
    # 0, hence the 0.00% of e^2 mass printed below). Feeding 0 into the
    # naive-table fit is therefore feeding the correct value, not a fabricated
    # one. The headline number does not depend on this either way -- it comes
    # from masked_group_mean, which excludes these points from the average.
    zero = T == 0
    tot0 = (e0 ** 2).sum()
    if (e0[zero] != 0).any():
        raise SystemExit("a y_true==0 point has nonzero error; the q=0 fill "
                         "above is no longer justified")
    print(f"y_true==0: {zero.sum()} points ({zero.mean():.5%}), "
          f"{(e0[zero] ** 2).sum() / tot0:.2%} of e^2 mass -- frozen")
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(zero, 0.0, (P - T) / np.where(zero, 1.0, T))

    # --- the seductive number ----------------------------------------------
    qc = np.clip(q, -1.0, 1.0)          # as the score sees it
    qbar = qc.mean(axis=1, keepdims=True)
    ss_tot = (qc ** 2).sum()
    ss_sys = (qbar ** 2).sum() * C
    print("\nvariance split of the capped relative error:")
    print(f"  systematic (per-point mean over all {C} corners): {ss_sys / ss_tot:6.1%}")
    print(f"  rough      (the only part smoothing can remove):  {1 - ss_sys / ss_tot:6.1%}")

    # --- naive ceilings: every point smoothed, truth assumed free ----------
    allc = np.full(C, "all")
    variants = {
        "S1  global constant": (allc, [one]),
        "S2  per-process constant": (proc, [one]),
        "S3  per-process {1, V, T}": (proc, [one, volt, temp]),
        # Same full 10-corner averaging window as S1, adding covariates rather
        # than narrowing the window. These exist so the docstring's "the global
        # constant won every basis we tried" is reproducible from the repo
        # instead of being a number remembered from a review.
        "S4  global {1, V}": (allc, [one, volt]),
        "S5  global {1, V, T}": (allc, [one, volt, temp]),
        "S6  global {1, process}": (allc, [one] + [(proc == p).astype(float)
                                                   for p in sorted(set(proc))[1:]]),
    }
    print(f"\nceiling with every point smoothed (truth assumed free):")
    print(f"  {'variant':30s} {'pooled':>9s} {'delta':>8s} {'oracle-gated':>13s} {'delta':>8s}")
    print(f"  {'baseline':30s} {base:9.4f} {0.0:+8.4f} {base:13.4f} {0.0:+8.4f}")
    best = ("", -np.inf, e0)
    for name, (groups, cols) in variants.items():
        qs = group_projection(q, groups, cols)
        es = np.where(zero, e0, np.minimum(1.0, np.abs(qs)))
        eg = np.minimum(es, e0)     # oracle decides per point where to smooth
        print(f"  {name:30s} {score(es):9.4f} {score(es) - base:+8.4f} "
              f"{score(eg):13.4f} {score(eg) - base:+8.4f}")
        if score(eg) > best[1]:
            best = (name, score(eg), eg)

    # --- where the mass is --------------------------------------------------
    sub = assign_subgroups(T.ravel(), A.ravel(), TT.ravel()).reshape(T.shape)
    flip = sub == SUBGROUP_FLIP
    nearzero = sub == SUBGROUP_NEAR_ZERO
    print(f"\ne^2 mass by subgroup, and what {best[0].strip()} (oracle-gated) removes:")
    print(f"  {'subgroup':20s} {'weight':>9s} {'e2 share':>9s} {'removed':>9s}")
    for lbl, tag in (("sign flip", SUBGROUP_FLIP), ("near zero", SUBGROUP_NEAR_ZERO),
                     ("fall_power bulk", SUBGROUP_BULK_FP),
                     ("other five tables", SUBGROUP_OTHER)):
        m = sub == tag
        b, a = (e0[m] ** 2).sum(), (best[2][m] ** 2).sum()
        print(f"  {lbl:20s} {m.mean():9.5f} {b / tot0:9.1%} {(b - a) / tot0:+9.2%}")

    # --- the honest strongest form -----------------------------------------
    # Log-space smoothing is undefined at a sign flip and unstable near zero,
    # and a saturated corner drags its healthy siblings up. So exclude those
    # corners from the average itself, not just from the result.
    smoothable = ~(flip | nearzero | zero)
    print(f"\nceiling with only smoothable points moved, and flip/near-zero "
          f"corners\nexcluded from the average itself "
          f"({smoothable.mean():.1%} of points, "
          f"{(e0[smoothable] ** 2).sum() / tot0:.1%} of e^2 mass):")
    w = smoothable.astype(float)
    for name, (groups, _cols) in list(variants.items())[:2]:
        qs = masked_group_mean(q, groups, w)
        es = np.where(smoothable, np.minimum(1.0, np.abs(qs)), e0)
        eg = np.minimum(es, e0)
        print(f"  {name:30s} {score(es):9.4f} {score(es) - base:+8.4f} "
              f"{score(eg):13.4f} {score(eg) - base:+8.4f}")

    # --- the bias every ceiling above assumed away -------------------------
    # A smoother acts on the model's log-ratio label; the label's own
    # out-of-basis component is injected as error into every smoothed point.
    rows = smoothable.all(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = T[rows] / A[rows]
    rows_ok = np.isfinite(ratio).all(axis=1) & (ratio > 0).all(axis=1)
    ll = np.log(ratio[rows_ok])
    print(f"\nbias injected by smoothing -- the other half of the trade "
          f"({rows_ok.sum()} clean rows):")
    for name, (groups, cols) in variants.items():
        resid = group_projection(ll, groups, cols) - ll
        r = float(np.sqrt((resid ** 2).mean()))
        print(f"  {name:30s} RMS {r:7.4f} log units -> {np.expm1(r):7.1%} "
              f"relative error injected")
    eb = e0[smoothable]
    print(f"  for scale, the current error on those same points is "
          f"p50 {np.percentile(eb, 50):.3%}, mean {eb.mean():.3%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", help=".npz produced by phase4_final_validate.py --dump-errors")
    main(ap.parse_args().dump)
