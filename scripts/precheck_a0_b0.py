"""A0/B0 pre-checks on the official training set (no training involved).

Q1 (direction A): at fall_power near-zero / sign-flip points, is the
    reconstruction fall = S - rise (S = rise+fall predicted separately)
    better than the current direct prediction (near-zero group score 67,
    flip group score 0)?  Reconstruction abs error ~ eps*sqrt(S^2+rise^2).
Q2 (direction B): does the zero-crossing position of fall_power tables
    drift regularly with voltage (SS 0.72/0.81/0.9, m40c)?
Q3 (info ceiling): dispersion of log-ratio target/anchor at near-zero
    points vs bulk points.
"""
import sys, numpy as np
from pathlib import Path

REPO = Path("/Users/jerrychen/My_Projects/Fast_Timing_Model_Esti_for_new_ PVT")
sys.path.insert(0, str(REPO / "src"))
from liberty.parser import parse_file  # noqa: E402

T = REPO / "testcase" / "training_set"
FILES = {
    "ss0p72vm40c": T / "base_nom_0p8v" / "lib1_ss0p72vm40c_base_400.tlib",
    "ss0p81vm40c": T / "base_nom_0p9v" / "lib1_ss0p81vm40c_base_400.tlib",
    "ss0p9vm40c":  T / "base_nom_1p0v" / "lib1_ss0p9vm40c_base_400.tlib",
    "ff0p99vm40c": T / "base_nom_0p9v" / "lib1_ff0p99vm40c_base_400.tlib",
    "ff1p1vm40c":  T / "base_nom_1p0v" / "lib1_ff1p1vm40c_base_400.tlib",
}
NZ = 1e-4

def load(corner):
    lf = parse_file(str(FILES[corner]))
    fall, rise = {}, {}
    for key, tb in lf.tables_by_key.items():
        if tb.values is None:
            continue
        if key[4] == "fall_power":
            fall[key[:4]] = tb.values
        elif key[4] == "rise_power":
            rise[key[:4]] = tb.values
    return fall, rise

def group_score(e):
    return 100.0 * (1.0 - np.sqrt(np.mean(np.square(np.minimum(e, 1.0)))))

def pair_report(anchor, target):
    fa, ra = load(anchor)
    ft, rt = load(target)
    keys = [k for k in ft if k in fa and k in rt and k in ra]
    ya, yt, rr = [], [], []
    for k in keys:
        a, t, r = fa[k], ft[k], rt[k]
        if not a.any() or not t.any():   # all-zero: invalid table, excluded
            continue
        ya.append(a.ravel()); yt.append(t.ravel()); rr.append(r.ravel())
    ya = np.concatenate(ya); yt = np.concatenate(yt); rr = np.concatenate(rr)
    n = yt.size
    zero_t = yt == 0
    flip = (ya * yt < 0)
    nz = (np.abs(yt) < NZ) & ~flip & ~zero_t
    bulk = ~flip & ~nz & ~zero_t
    print(f"\n=== {anchor} -> {target}  (n={n}, tables kept={n//49}) ===")
    print(f"exact-zero targets: {zero_t.sum()} ({100*zero_t.mean():.3f}%)")
    print(f"sign-flip: {flip.sum()} ({100*flip.mean():.3f}%)   "
          f"near-zero(|y|<1e-4, sign-ok): {nz.sum()} ({100*nz.mean():.3f}%)   "
          f"bulk: {100*bulk.mean():.1f}%")

    # Q1: reconstruction feasibility.  fall_hat = S_hat - rise_hat
    for name, mask in (("near-zero", nz), ("sign-flip", flip & ~zero_t)):
        if not mask.any():
            continue
        S = yt[mask] + rr[mask]
        R = rr[mask]
        y = np.abs(yt[mask])
        ratio = np.abs(R) / np.maximum(y, 1e-15)
        q = np.percentile(ratio, [10, 50, 90])
        print(f"  [{name}] |rise|/|fall| P10/P50/P90 = {q[0]:.1f} / {q[1]:.1f} / {q[2]:.1f}")
        for eps in (0.005, 0.01, 0.02):
            e = eps * np.sqrt(S**2 + R**2) / np.maximum(y, 1e-15)
            print(f"    eps={eps*100:.1f}%: reconstruction group score = "
                  f"{group_score(e):.1f}  (fail-capped frac {np.mean(e>=1)*100:.0f}%)")

    # Q3: log-ratio dispersion, bulk vs near-zero
    for name, mask in (("bulk", bulk), ("near-zero", nz)):
        m = mask & (np.abs(ya) > 0)
        lr = np.log(np.abs(yt[m]) / np.abs(ya[m]))
        print(f"  [Q3 {name}] log|t/a|: std={lr.std():.3f}  IQR={np.subtract(*np.percentile(lr,[75,25]))*-1:.3f}")
        # score if we predicted with the median ratio (naive baseline)
        pred = np.abs(ya[m]) * np.exp(np.median(lr))
        pred = np.where(yt[m] < 0, -pred, pred)  # oracle sign, isolates magnitude info
        e = np.abs(yt[m] - pred) / np.abs(yt[m])
        print(f"           naive median-ratio predictor (oracle sign) group score = {group_score(e):.1f}")

def crossings(vals):
    """Row-wise sign-change crossing positions (col units, linear interp)."""
    out = {}
    for i in range(7):
        row = vals[i]
        s = np.sign(row)
        idx = np.where(s[:-1] * s[1:] < 0)[0]
        if len(idx) == 1:
            j = idx[0]
            frac = row[j] / (row[j] - row[j + 1])
            out[i] = j + frac
    return out

def contour_report():
    tabs = {c: load(c)[0] for c in ("ss0p72vm40c", "ss0p81vm40c", "ss0p9vm40c")}
    common = set(tabs["ss0p72vm40c"]) & set(tabs["ss0p81vm40c"]) & set(tabs["ss0p9vm40c"])
    has_cross = 0
    drifts_dn, drifts_up = [], []   # 0.81->0.72, 0.81->0.9
    for k in common:
        v72, v81, v90 = tabs["ss0p72vm40c"][k], tabs["ss0p81vm40c"][k], tabs["ss0p9vm40c"][k]
        if not (v81.any() and v72.any() and v90.any()):
            continue
        c72, c81, c90 = crossings(v72), crossings(v81), crossings(v90)
        if c81:
            has_cross += 1
        for row in set(c81) & set(c72):
            drifts_dn.append(c72[row] - c81[row])
        for row in set(c81) & set(c90):
            drifts_up.append(c90[row] - c81[row])
    print(f"\n=== B0: zero-crossing contour, SS m40c 0.81 vs 0.72/0.9 ===")
    print(f"tables with >=1 single-crossing row at 0.81V: {has_cross}")
    for name, d in (("0.81->0.72", drifts_dn), ("0.81->0.90", drifts_up)):
        if d:
            d = np.array(d)
            q = np.percentile(d, [10, 25, 50, 75, 90])
            print(f"  drift {name} (col units, n={len(d)}): "
                  f"P10={q[0]:+.2f} P25={q[1]:+.2f} P50={q[2]:+.2f} P75={q[3]:+.2f} P90={q[4]:+.2f}")

pair_report("ss0p81vm40c", "ss0p72vm40c")
pair_report("ff0p99vm40c", "ff1p1vm40c")
contour_report()
