"""End-to-end sign-flip prediction oracle for fall_power.

Mechanism under test: the fall_power zero-crossing contour moves along
index_2 (columns) by a highly regular per-corner-pair shift when voltage
changes (PLAN.md direction B). Flip points are the grid points the
contour sweeps. If we predict the target contour position from the
anchor's (possibly VIRTUAL, off-grid extrapolated) crossing + the
population median shift, we can predict each point's target sign.

Measures, per corner pair and pooled:
  - flip recall / false-flip rate vs decision margin
  - net pooled-score delta from applying a surgical sign-correction
    decision layer on top of the existing model (error model:
    corrected true-flip -> err = |exp(dr)-1| with dr from sibling ratio
    stats; false-flip -> err 1.0; missed flip -> stays 1.0).

Zero training; population medians are corner-level statistics (400
cells), leakage-free enough for a feasibility bound.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from liberty.parser import parse_file  # noqa: E402
from models.phase4_features import NEAREST_ANCHOR_BY_TARGET  # noqa: E402


def libpath(corner):
    for d in glob.glob("testcase/training_set/base_nom_*"):
        p = os.path.join(d, f"lib1_{corner}_base_400.tlib")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(corner)


def row_crossings(row, lo=-8.0, hi=14.0):
    """First zero crossing position (fractional column) of a 7-value row,
    linearly interpolated; if none in-grid, extrapolate from both edges
    and return the nearest virtual crossing within [lo, hi]. Returns
    (pos, low_side_sign) or (None, sign) if effectively single-sign."""
    s = np.sign(row)
    # in-grid crossing
    for j in range(6):
        if s[j] != 0 and s[j + 1] != 0 and s[j] != s[j + 1]:
            t = row[j] / (row[j] - row[j + 1])
            return j + t, s[j]
    # virtual crossing beyond right edge
    cands = []
    if row[6] != row[5]:
        x = 6.0 - row[6] / (row[6] - row[5]) * 1.0
        # crossing must lie beyond the right edge to be consistent
        if 6.0 < x <= hi:
            cands.append((x, s[6] if s[6] != 0 else 1.0))
    if row[0] != row[1]:
        x = 0.0 + row[0] / (row[0] - row[1]) * 1.0
        x = -x  # position measured leftwards
        if lo <= x < 0.0:
            cands.append((x, -s[0] if s[0] != 0 else 1.0))
            # low side of a left-virtual crossing is the (off-grid) side;
            # in-grid is entirely the "high" side => low_side_sign = -s[0]
    if cands:
        cands.sort(key=lambda c: min(abs(c[0] - 6.0), abs(c[0])))
        return cands[0]
    return None, s[3] if s[3] != 0 else 1.0


def main():
    pairs = sorted(NEAREST_ANCHOR_BY_TARGET.items())
    anchors = {}
    MARGINS = [0.0, 0.5, 0.75, 1.0, 1.5]

    pooled = {m: dict(fixed=0, missed=0, false=0, keep_ok=0) for m in MARGINS}
    tot_points_all = 0
    flip_dr = []  # |log ratio| stats of true flips for value-error model
    shift_stats = {}

    for tgt, anc in pairs:
        if anc not in anchors:
            anchors[anc] = parse_file(libpath(anc))
        alib, tlib = anchors[anc], parse_file(libpath(tgt))

        # ---- pass 1: learn per-pair median contour shift (rows with a
        # real crossing in both anchor and target) ----
        shifts = []
        tables = []
        for key, ttab in tlib.tables_by_key.items():
            if key[-1] != "fall_power" or ttab.values is None:
                continue
            atab = alib.tables_by_key.get(key)
            if atab is None or atab.values is None:
                continue
            A, Y = atab.values, ttab.values
            tables.append((key, A, Y))
            for i in range(7):
                pa, _ = row_crossings(A[i])
                py, _ = row_crossings(Y[i])
                if pa is not None and py is not None and 0 <= pa <= 6 and 0 <= py <= 6:
                    shifts.append(py - pa)
        delta = float(np.median(shifts)) if shifts else 0.0
        iqr = (np.percentile(shifts, 75) - np.percentile(shifts, 25)) if shifts else 0.0
        shift_stats[tgt] = (delta, iqr, len(shifts))

        # ---- pass 2: per-point sign prediction & bookkeeping ----
        stats = {m: dict(fixed=0, missed=0, false=0, keep_ok=0) for m in MARGINS}
        n_points = 0
        for key, A, Y in tables:
            n_points += Y.size
            for i in range(7):
                pa, lows = row_crossings(A[i])
                arow, yrow = A[i], Y[i]
                for j in range(7):
                    if arow[j] == 0 or yrow[j] == 0:
                        continue
                    true_flip = arow[j] * yrow[j] < 0
                    if true_flip:
                        flip_dr.append(np.log(abs(yrow[j]) / abs(arow[j])))
                    if pa is None:
                        pred_shift_flip = False
                        dist = np.inf
                    else:
                        p_new = pa + delta
                        # sign at j before/after shift
                        sign_before = lows if j < pa else -lows
                        sign_after = lows if j < p_new else -lows
                        pred_shift_flip = sign_before != sign_after
                        dist = abs(j - p_new)
                    # anchor sign at j must equal sign_before by
                    # construction when pa in-grid; trust arow[j] anyway
                    for m in MARGINS:
                        decide_flip = pred_shift_flip and dist >= m
                        if decide_flip and true_flip:
                            stats[m]["fixed"] += 1
                        elif decide_flip and not true_flip:
                            stats[m]["false"] += 1
                        elif not decide_flip and true_flip:
                            stats[m]["missed"] += 1
                        else:
                            stats[m]["keep_ok"] += 1
        tot_points_all += n_points
        for m in MARGINS:
            for k in stats[m]:
                pooled[m][k] += stats[m][k]
        s = stats[0.75]
        print(f"{tgt:14s} shift {delta:+.2f} (IQR {iqr:.2f}, n={len(shifts):5d})  "
              f"[margin .75] fixed {s['fixed']:5d}  missed {s['missed']:5d}  "
              f"false {s['false']:5d}")

    # ---- value-error model for corrected flips ----
    flip_dr = np.array(flip_dr)
    print(f"\ntrue-flip |log ratio| median {np.median(np.abs(flip_dr)):.2f}  "
          f"log-ratio std {flip_dr.std():.2f}")
    # corrected flip predicted with sibling/population median ratio:
    med = np.median(flip_dr)
    errs_corrected = np.minimum(1.0, np.abs(np.exp(med - flip_dr) - 1.0))
    e2_corr = float(np.mean(errs_corrected**2))
    print(f"corrected-flip value error model: score "
          f"{100 - 100*np.sqrt(e2_corr):.1f} (e2 {e2_corr:.3f})")

    # ---- pooled score deltas ----
    # NOTE tot_points_all counts fall_power points only; pooled score uses
    # ALL table types: fall_power share ~= 1/6 of pooled (measured 16.59%).
    FP_SHARE = 0.1659
    e2_total = (1 - 96.45 / 100) ** 2
    print(f"\n{'margin':>6s} {'recall':>7s} {'falseFlip%':>10s} {'pooled delta':>12s} {'-> score':>9s}")
    for m in MARGINS:
        s = pooled[m]
        nfp = s["fixed"] + s["missed"]
        nkeep = s["false"] + s["keep_ok"]
        share = 1.0 / tot_points_all * FP_SHARE  # sanity ~1
        # per fall_power point masses -> pooled masses
        gain = s["fixed"] / tot_points_all * (1.0 - e2_corr) * FP_SHARE
        # false flip: err was e_keep (assume current model err ~0.05 on
        # kept-sign points, conservative) -> becomes 1.0
        loss = s["false"] / tot_points_all * (1.0 - 0.05**2) * FP_SHARE
        e2_new = e2_total - gain + loss
        print(f"{m:6.2f} {s['fixed']/max(1,nfp):7.1%} "
              f"{s['false']/max(1,nkeep):10.3%} {(-gain+loss)*1e4:11.3f}e-4 "
              f"{100-100*np.sqrt(max(e2_new,0)):9.2f}")
    print(f"\n(fall_power points scanned: {tot_points_all}, "
          f"true flips pooled: {pooled[0.0]['fixed']+pooled[0.0]['missed']})")


if __name__ == "__main__":
    main()
