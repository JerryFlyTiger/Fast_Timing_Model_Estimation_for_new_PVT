"""Alpha-composition reweighted score audit (2026-08-09, direction F --
docs/recheck_20260809.md section 5).

Reads the per-point validation dump produced by
`scripts/phase4_final_validate.py --dump-errors PATH` and answers:

  "The acceptance run scored held-out TRAINING cells. What score should
   the SAME model be expected to get on the official ALPHA population,
   whose fall_power pathology (near-zero / sign-flip points) is
   measurably rarer?"

Method (scoring.audits section 3 helpers):

1. Decompose the dump's per-point errors into subgroups
   {fall_power:flip, fall_power:near_zero, fall_power:bulk, other_tables}
   per corner, and report each group's share/score/e2 mass.
2. Measure the alpha/train pathology-prevalence ratios at ALL 5 anchor
   corners (the corners where both populations have real values):
   near-zero point share ratio and mixed-sign table share ratio, per
   process (ss/ff/tt).
3. Reweight: scale each corner's flip/near-zero group weights by its
   process's measured ratio (flip -> mixed-sign-table ratio, near_zero
   -> near-zero-share ratio), renormalize, and report the reweighted
   pooled score. A sensitivity band swaps the flip proxy for the
   near-zero proxy (and vice versa) to bracket proxy-choice uncertainty.
4. Also reports the direction-D dual metrics for free: pooled excluding
   flips, and pooled excluding flips + near-zero points.

Stated assumption (also in docs/recheck_20260809.md section 5): only the
*composition* differs between populations; each subgroup's conditional
error distribution transfers unchanged. Bulk conditional difficulty is
assumed population-independent.

Usage:
    python3 scripts/phase4_alpha_audit.py output/_phase4_cache/final_validate_errors.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from features.corners import parse_corner_filename
from liberty.parser import parse_file
from models.phase4_features import ANCHOR_CORNER_NAMES
from paths import ALPHA_FULL_DIR, training_set_files
from scoring.audits import (
    SUBGROUP_FLIP,
    SUBGROUP_NEAR_ZERO,
    assign_subgroups,
    measure_fall_power_composition,
    reweighted_pooled_score,
    subgroup_stats,
)
from scoring.scorer import point_errors, score_from_errors


def _process_of(corner_name: str) -> str:
    return parse_corner_filename(f"lib1_{corner_name}_alpha_100.lib").process


def measure_anchor_ratios():
    """Per-process alpha/train prevalence ratios measured at the 5 anchor
    corners: {process: (near_zero_share_ratio, mixed_table_share_ratio)}.
    Multi-anchor processes (ss/ff have 2 anchors each) pool by simple
    mean of per-anchor ratios."""
    train_paths = {parse_corner_filename(str(p)).name: p for p in training_set_files()}
    per_process: dict = {}
    print("anchor-corner pathology prevalence (train400 -> alpha100):")
    for anchor in ANCHOR_CORNER_NAMES:
        train_lib = parse_file(str(train_paths[anchor]))
        alpha_lib = parse_file(str(ALPHA_FULL_DIR / f"lib1_{anchor}_alpha_100.lib"))
        nz_t, mixed_t = measure_fall_power_composition(train_lib)
        nz_a, mixed_a = measure_fall_power_composition(alpha_lib)
        r_nz = nz_a / nz_t
        r_mixed = mixed_a / mixed_t
        print(f"  {anchor:14s} near-zero {100*nz_t:5.2f}% -> {100*nz_a:5.2f}%  (ratio {r_nz:.3f})   "
              f"mixed-sign tables {100*mixed_t:5.2f}% -> {100*mixed_a:5.2f}%  (ratio {r_mixed:.3f})")
        per_process.setdefault(_process_of(anchor), []).append((r_nz, r_mixed))
    out = {}
    for proc, pairs in per_process.items():
        r_nz = float(np.mean([p[0] for p in pairs]))
        r_mixed = float(np.mean([p[1] for p in pairs]))
        out[proc] = (r_nz, r_mixed)
        print(f"  process {proc}: near-zero ratio {r_nz:.3f}, mixed-table ratio {r_mixed:.3f}")
    return out


def main(dump_path: str) -> None:
    d = np.load(dump_path, allow_pickle=False)
    corner = d["corner"]
    table_type = d["table_type"]
    y_true = d["y_true"]
    y_pred = d["y_pred"]
    nearest = d["nearest_anchor"]
    print(f"dump: {dump_path}")
    print(f"  config={d['meta_config'][0]} stage={d['meta_stage'][0]} "
          f"fold={d['meta_fold'][0]} seeds={d['meta_n_seeds'][0]}  n={y_true.size}")

    errs = point_errors(y_true, y_pred)
    official = score_from_errors(errs)
    print(f"\nofficial pooled (this validation run, training-cell composition): {official:.4f}")

    sub = assign_subgroups(y_true, nearest, table_type)
    print("\nsubgroup decomposition (pooled across corners):")
    total_e2 = float(np.mean(errs ** 2))
    for s in subgroup_stats(errs, sub):
        print(f"  {s.name:24s} n={s.n_points:8d} share={100*s.share:7.3f}%  score={s.score:7.2f}  "
              f"e2_mass={s.e2_mass:.6f} ({100*s.e2_mass/total_e2:5.1f}% of total)")

    ratios = measure_anchor_ratios()

    # Per-(corner, subgroup) group ids so each corner's pathological
    # groups get its own process's measured ratio.
    group_ids = np.char.add(np.char.add(corner.astype("<U16"), "|"), sub)

    def multipliers(flip_proxy: str, nz_proxy: str):
        m = {}
        for c in np.unique(corner):
            proc = _process_of(str(c))
            r_nz, r_mixed = ratios[proc]
            proxy = {"nz": r_nz, "mixed": r_mixed}
            m[f"{c}|{SUBGROUP_FLIP}"] = proxy[flip_proxy]
            m[f"{c}|{SUBGROUP_NEAR_ZERO}"] = proxy[nz_proxy]
        return m

    central = reweighted_pooled_score(errs, group_ids, multipliers("mixed", "nz"))
    alt_a = reweighted_pooled_score(errs, group_ids, multipliers("nz", "nz"))
    alt_b = reweighted_pooled_score(errs, group_ids, multipliers("mixed", "mixed"))
    lo, hi = min(central, alt_a, alt_b), max(central, alt_a, alt_b)
    print("\n=== alpha-composition reweighted pooled score ===")
    print(f"  central (flip~mixed-table ratio, near-zero~near-zero ratio): {central:.4f}")
    print(f"  proxy sensitivity band: [{lo:.4f}, {hi:.4f}]")

    # Direction-D dual metrics on the raw (training-composition) run.
    keep_noflip = sub != SUBGROUP_FLIP
    keep_clean = keep_noflip & (sub != SUBGROUP_NEAR_ZERO)
    print("\ndirection-D dual metrics (training-cell composition):")
    print(f"  excluding sign-flip points:            {score_from_errors(errs[keep_noflip]):.4f}")
    print(f"  excluding sign-flip + near-zero points: {score_from_errors(errs[keep_clean]):.4f}")

    print("\nper-corner official vs reweighted:")
    for c in sorted(np.unique(corner).tolist()):
        mask = corner == c
        off_c = score_from_errors(errs[mask])
        rew_c = reweighted_pooled_score(errs[mask], group_ids[mask], multipliers("mixed", "nz"))
        print(f"  {c:16s} official {off_c:8.4f}   reweighted {rew_c:8.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", help=".npz produced by phase4_final_validate.py --dump-errors")
    main(ap.parse_args().dump)
