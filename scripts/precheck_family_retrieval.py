"""Family-consistency analysis for fall_power near-zero / sign-flip points.

Question: is the anchor->target ratio at pathological points consistent
WITHIN a cell function-family (across drive strengths)? If yes, a
training-set sibling lookup can supply the "cell individuality" that the
pointwise MLP provably cannot extract from anchor features alone.

Zero training. Pure measurement on the official 400-cell training set,
leave-one-out within family to simulate the alpha situation (alpha cell
absent, siblings present).
"""
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "src")
from liberty.parser import parse_file  # noqa: E402

ROOT = "testcase/training_set"
PAIRS = [
    # (anchor lib, target lib, tag)
    (f"{ROOT}/base_nom_0p9v/lib1_ss0p81vm40c_base_400.tlib",
     f"{ROOT}/base_nom_0p8v/lib1_ss0p72vm40c_base_400.tlib", "ss m40c 0.81->0.72 (buck)"),
    (f"{ROOT}/base_nom_0p9v/lib1_ff0p99vm40c_base_400.tlib",
     f"{ROOT}/base_nom_1p0v/lib1_ff1p1vm40c_base_400.tlib", "ff m40c 0.99->1.1 (boost)"),
]

NEAR_ZERO = 1e-4

def family_of(cell: str) -> str:
    return re.sub(r"M[0-9]+B?$", "", cell)

def drive_of(cell: str) -> float:
    m = re.search(r"M([0-9]+)B?$", cell)
    return float(m.group(1)) if m else 1.0

def score_from_errs(errs):
    errs = np.asarray(errs, float)
    return 100.0 - 100.0 * np.sqrt(np.mean(np.minimum(1.0, errs) ** 2))

for anchor_path, target_path, tag in PAIRS:
    print(f"\n================ {tag} ================")
    alib = parse_file(anchor_path)
    tlib = parse_file(target_path)

    # rows: (cell, family, drive, subkey, r, is_nz, is_flip, y, a)
    # subkey identifies the position within the family-shared structure:
    # (pin, group_type, arc_index, table_type, grid_idx)
    recs = []
    for key, ttab in tlib.tables_by_key.items():
        if key[-1] != "fall_power" or ttab.values is None:
            continue
        atab = alib.tables_by_key.get(key)
        if atab is None or atab.values is None:
            continue
        cell = key[0]
        fam, drv = family_of(cell), drive_of(cell)
        y = ttab.values.ravel()
        a = atab.values.ravel()
        for g in range(49):
            if a[g] == 0.0:
                continue
            flip = y[g] * a[g] < 0
            nz = (abs(y[g]) < NEAR_ZERO) and not flip and y[g] != 0.0
            if not (flip or nz):
                continue
            subkey = key[1:] + (g,)
            r = np.log(abs(y[g]) / abs(a[g])) if y[g] != 0 else None
            recs.append((cell, fam, drv, subkey, r, nz, flip, y[g], a[g]))

    nz_recs = [x for x in recs if x[5]]
    flip_recs = [x for x in recs if x[6]]
    print(f"near-zero pts: {len(nz_recs)}, sign-flip pts: {len(flip_recs)}")

    # ---------- near-zero: variance decomposition + LOO ----------
    groups = defaultdict(list)
    for cell, fam, drv, subkey, r, nz, flip, y, a in nz_recs:
        groups[(fam, subkey)].append((cell, drv, r, y, a))

    rs_all = np.array([x[4] for x in nz_recs])
    print(f"overall log-ratio std (across cells): {rs_all.std():.3f}")

    within_devs, loo_errs, base_errs = [], [], []
    med = np.median(rs_all)
    n_multi = 0
    for (fam, subkey), members in groups.items():
        if len(members) < 2:
            continue
        n_multi += len(members)
        rs = np.array([m[2] for m in members])
        within_devs.extend(rs - rs.mean())
        for i, (cell, drv, r, y, a) in enumerate(members):
            others = np.array([m[2] for j, m in enumerate(members) if j != i])
            pred_r = np.mean(others)
            # relative error of value prediction a*exp(pred_r) vs y=a*exp(r), same sign
            loo_errs.append(abs(np.exp(pred_r - r) - 1.0))
            base_errs.append(abs(np.exp(med - r) - 1.0))
    within = np.array(within_devs)
    print(f"near-zero pts with >=2 family members at same subkey: {n_multi} "
          f"({100.0*n_multi/max(1,len(nz_recs)):.1f}% coverage)")
    if len(within):
        print(f"WITHIN-family log-ratio std: {np.sqrt(np.mean(within**2)):.3f}"
              f"   (vs across-cell {rs_all.std():.3f})")
        print(f"score on covered near-zero pts:  global-median baseline {score_from_errs(base_errs):6.2f}"
              f"   family-LOO {score_from_errs(loo_errs):6.2f}")

    # ---------- sign-flip: LOO majority vote on flip-ness ----------
    fgroups = defaultdict(list)
    # need flip status per (fam,subkey) for ALL points that are nz or flip or even bulk?
    # For sign prediction what matters: at this subkey, did the sibling flip too?
    allpts = defaultdict(list)
    for key, ttab in tlib.tables_by_key.items():
        if key[-1] != "fall_power" or ttab.values is None:
            continue
        atab = alib.tables_by_key.get(key)
        if atab is None or atab.values is None:
            continue
        cell = key[0]
        fam = family_of(cell)
        y = ttab.values.ravel(); a = atab.values.ravel()
        for g in range(49):
            if a[g] == 0.0:
                continue
            allpts[(fam, key[1:] + (g,))].append((cell, y[g] * a[g] < 0, y[g], a[g]))
    correct = wrong = uncovered = 0
    loo_flip_value_errs = []
    for cell, fam, drv, subkey, r, nz, flip, y, a in flip_recs:
        members = allpts[(fam, subkey)]
        others = [m for m in members if m[0] != cell]
        if not others:
            uncovered += 1
            continue
        vote = np.mean([o[1] for o in others])
        if vote > 0.5:
            correct += 1
            # value prediction from flipped siblings' ratio
            ors = [np.log(abs(o[2]) / abs(o[3])) for o in others if o[1] and o[2] != 0]
            if ors and y != 0:
                pred = -abs(a) * np.exp(np.mean(ors)) * (1 if a < 0 else -1)  # opposite sign of a... careful
                # simpler: predicted value = sign(y_pred)=-sign(a), magnitude |a|*exp(mean ors)
                predv = -np.sign(a) * abs(a) * np.exp(np.mean(ors))
                loo_flip_value_errs.append(abs(predv - y) / abs(y))
        else:
            wrong += 1
    print(f"sign-flip LOO majority vote: correct {correct}, wrong {wrong}, uncovered {uncovered}")
    if loo_flip_value_errs:
        print(f"  flip pts, sibling-ratio value pred score: {score_from_errs(loo_flip_value_errs):6.2f} "
          f"(current model: 0)")
    # false-positive check: how often would the vote flip a NON-flip point?
    fp = tn = 0
    for cell, fam, drv, subkey, r, nz, flip, y, a in nz_recs:
        members = allpts[(fam, subkey)]
        others = [m for m in members if m[0] != cell]
        if not others:
            continue
        vote = np.mean([o[1] for o in others])
        if vote > 0.5:
            fp += 1
        else:
            tn += 1
    print(f"false-positive flips on near-zero non-flip pts: {fp} vs {tn} correct-keep")

    # ---------- sibling count stats ----------
    fam_sizes = defaultdict(set)
    for key in tlib.tables_by_key:
        fam_sizes[family_of(key[0])].add(key[0])
    sizes = np.array([len(v) for v in fam_sizes.values()])
    print(f"family sizes in training: mean {sizes.mean():.1f}, "
          f"1-member families: {(sizes==1).sum()}/{len(sizes)}")
