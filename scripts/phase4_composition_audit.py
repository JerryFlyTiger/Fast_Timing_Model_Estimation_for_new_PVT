"""Drive-matched composition audit -- what score does a validation run on
held-out TRAINING cells correspond to on an official test population?

This is the corrected method of docs/round_20260810.md section 2, which until now
existed only as prose in that document (the original computation was done
ad-hoc in a session transcript and was never committed). It supersedes
`scripts/phase4_alpha_audit.py`, whose direct anchor-ratio extrapolation
produced the retired 97.4344 -- that number is contaminated by drive
strength, because the test populations differ from train400 in drive
(median drive: train400=6, alpha=2, beta=4, final=8) and drive is itself
strongly correlated with fall_power pathology.

Method
------
The scored run measures the 80 held-out training cells. The official
population's target-corner values are blanked, so its pathology
prevalence there can never be observed directly -- only at the 5 corners
that stage gets fully populated. So prevalence is *modelled* and the
model is calibrated on those 5:

1. Rate table: measure pathology prevalence per (corner x drive bucket)
   on train400, the only population with real values at all 15 corners.
2. Drive-matched expectation for a test population at any corner:
       expected(corner) = sum_b  test_drive_share(b) * rate(corner, b)
   i.e. "what would this corner look like if the test population's
   pathology were entirely explained by its drive mix".
3. Residual population factor k, calibrated where truth exists:
       k = mean over the stage's 5 populated corners of
           true_test_prevalence(anchor) / expected(anchor)
   k < 1 means the population is cleaner than its drive mix alone
   predicts. Its spread across the 5 anchors is reported as the
   stability check: the whole method assumes corner effects and
   population effects separate, and a k that swung wildly per anchor
   would falsify that.
4. Reweight the dump's per-point errors, scaling each corner's
   pathological subgroup weights by
       m(corner) = k * expected(corner) / true_heldout80_prevalence(corner)
   (denominator measured directly on the same 80 cells the dump scored,
   from the training libs) and renormalizing -- `reweighted_pooled_score`.

Two observable proxies stand in for the two pathological subgroups, as
in the superseded script: sign flips ~ mixed-sign fall_power table
share, near-zero points ~ near-zero fall_power point share. The central
estimate pairs each subgroup with its natural proxy; swapping them gives
the sensitivity band.

Stated assumption (unchanged from the old audit, and empirically
supported by docs/round_20260810.md section 3's R^2=0.906 regression):
only *composition* differs between populations; each subgroup's
conditional error distribution transfers unchanged.

Usage:
    python3 scripts/phase4_composition_audit.py DUMP.npz --population alpha \
        --bucket-scheme stable
    python3 scripts/phase4_composition_sensitivity.py DUMP.npz --population alpha

`--bucket-scheme` is required, not defaulted: see the BUCKET_SCHEMES
comment below. This script answers "what does the audit give under THIS
choice"; the sensitivity script answers "how much does the choice
matter", and it is the second one that should be quoted when a single
number is wanted.

`--expect` is a computation-reproducibility gate (does today's code still
produce the same figure as the recorded run, for a stated scheme), NOT
evidence that the reconstruction recovered the original method. Under
`--bucket-scheme stable` it reproduces docs/round_20260810.md section 2's
97.3378 to 97.3355; under `none` the same code gives 97.4255. Both are
defensible readings of section 2's prose.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from features.cellinfo import parse_cell_name
from features.corners import parse_corner_filename
from liberty.parser import parse_file
from models.phase4_features import STAGE_TOPOLOGIES
from paths import stage_full_lib, training_set_files
from scoring.audits import (
    SUBGROUP_FLIP,
    SUBGROUP_NEAR_ZERO,
    aggregate_composition,
    assign_subgroups,
    composition_multiplier,
    per_cell_fall_power_composition,
    reweighted_pooled_score,
    subgroup_stats,
)
from scoring.scorer import point_errors, score_from_errors

# Drive-strength buckets. Coarser than exact drive values because
# train400's tail is thin (drives 10/18/20/24/26/32/36/40/48 have 2-4
# cells each -- a per-value rate there would be pure noise), and because
# beta100 contains a drive (14) that train400 does not have at all.
#
# The edges are NOT free, and -- importantly -- NOTHING SELECTS THEM.
# docs/round_20260810.md section 2 never recorded them, and
# `scripts/phase4_composition_sensitivity.py` measures the alpha answer
# moving over 97.298-97.526 across defensible choices: an
# implementation-choice spread (~0.23) larger than every model
# improvement this project has measured.
#
# A first attempt (2026-08-11) picked `stable` by the method's own stated
# self-check -- section 2 justifies separability with "k is stable across
# the stage's own 5 anchors" -- and reported that it reproduced the
# recorded 97.3378. A cold review killed that reasoning: the criterion was
# applied only to k(near-zero), while the central estimate applies
# k(mixed-table) to the sign-flip subgroup, which carries 63% (alpha) to
# 69% (beta) of all squared-error mass. Measured on alpha, the two
# proxies rank the schemes in OPPOSITE orders:
#
#   scheme      sd(k_nz)   sd(k_mixed)   central
#   stable        0.024       0.346       97.3355   <- best nz, worst mixed
#   occupancy     0.037       0.245       97.3206
#   fine          0.046       0.236       97.3512
#   coarse        0.033       0.313       97.3635
#   none          0.039       0.190       97.4255   <- best mixed
#
# So the stability criterion does not select a scheme; picking the
# near-zero proxy for it was what selected one. There is no principled
# default, which is why this script REQUIRES --bucket-scheme rather than
# silently supplying one: a number out of this audit is meaningless
# without the scheme printed next to it. Report the range from
# `phase4_composition_sensitivity.py` when a single caveat-free figure is
# wanted.
BUCKET_SCHEMES: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "stable": ((1, 2), (3, 6), (7, 14), (15, 10 ** 9)),
    "occupancy": ((1, 1), (2, 3), (4, 6), (8, 14), (16, 10 ** 9)),
    "fine": ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 10 ** 9)),
    "coarse": ((1, 3), (4, 8), (9, 10 ** 9)),
    "none": ((1, 10 ** 9),),
}


def make_bucket_of(scheme: str):
    edges = BUCKET_SCHEMES[scheme]

    def bucket_of(cell_name: str) -> str:
        d = parse_cell_name(cell_name).drive_strength
        for lo, hi in edges:
            if lo <= d <= hi:
                return f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
        raise ValueError(f"drive strength {d} of {cell_name!r} falls outside scheme {scheme!r}")

    return bucket_of, edges


def load_training_libs() -> Dict[str, "object"]:
    """corner name -> parsed training-set lib (all 15 corners)."""
    out = {}
    for p in training_set_files():
        out[parse_corner_filename(str(p)).name] = parse_file(str(p))
    assert len(out) == 15, f"expected 15 training corners, got {sorted(out)}"
    return out


def main(dump_path: str, population: str, expect: float | None, scheme: str) -> bool:
    bucket_of, _ = make_bucket_of(scheme)
    d = np.load(dump_path, allow_pickle=False)
    corner = d["corner"]
    table_type = d["table_type"]
    y_true = d["y_true"]
    y_pred = d["y_pred"]
    nearest = d["nearest_anchor"]
    dump_stage = str(d["meta_stage"][0])

    print(f"dump: {dump_path}")
    print(f"  config={d['meta_config'][0]} stage={dump_stage} "
          f"fold={d['meta_fold'][0]} seeds={d['meta_n_seeds'][0]}  n={y_true.size}")
    print(f"  population audited against: {population!r}")
    if dump_stage != population:
        print(f"  !! WARNING: dump was produced under the {dump_stage!r} topology but is being "
              f"reweighted to the {population!r} population. These are independent choices "
              f"(topology = which corners are known; population = which 100 cells are scored), "
              f"but a mismatch is almost always a mistake.")

    topology = STAGE_TOPOLOGIES[population]
    errs = point_errors(y_true, y_pred)
    official = score_from_errors(errs)
    print(f"\nmeasured pooled (this run, held-out training-cell composition): {official:.4f}")

    sub = assign_subgroups(y_true, nearest, table_type)
    total_e2 = float(np.mean(errs ** 2))
    print("\nsubgroup decomposition (pooled across corners):")
    for s in subgroup_stats(errs, sub):
        print(f"  {s.name:24s} n={s.n_points:8d} share={100*s.share:7.3f}%  score={s.score:7.2f}  "
              f"e2_mass={s.e2_mass:.7f} ({100*s.e2_mass/total_e2:5.1f}% of total)")

    # ---- 1. rate table: prevalence per (corner x drive bucket) on train400
    print("\nparsing 15 training-set corners...")
    train_libs = load_training_libs()
    train_comp = {c: per_cell_fall_power_composition(lib) for c, lib in train_libs.items()}
    any_corner = next(iter(train_libs))
    train_cells = sorted(train_libs[any_corner].cells)
    train_by_bucket: Dict[str, List[str]] = {}
    for c in train_cells:
        train_by_bucket.setdefault(bucket_of(c), []).append(c)

    # ---- 2. test population's drive mix, from any one of its populated libs
    pop_anchor_libs = {a: parse_file(str(stage_full_lib(population, a)))
                       for a in topology.anchor_names}
    pop_cells = sorted(next(iter(pop_anchor_libs.values())).cells)
    pop_bucket_share: Dict[str, float] = {}
    for c in pop_cells:
        pop_bucket_share[bucket_of(c)] = pop_bucket_share.get(bucket_of(c), 0.0) + 1.0 / len(pop_cells)

    print(f"\ndrive-bucket occupancy (scheme {scheme!r}; train400 cells / {population}100 share):")
    for label in sorted(set(train_by_bucket) | set(pop_bucket_share)):
        n_tr = len(train_by_bucket.get(label, []))
        share = pop_bucket_share.get(label, 0.0)
        print(f"  {label:>5s}: train n={n_tr:4d}   {population} share={100*share:6.2f}%")
        if share > 0 and n_tr == 0:
            raise SystemExit(f"bucket {label!r} is populated in {population}100 but empty in "
                             f"train400 -- the rate table cannot be evaluated there")

    def expected(corner_name: str) -> Tuple[float, float]:
        """Drive-matched expected (near_zero_share, mixed_share) for the
        test population at `corner_name`."""
        nz = mixed = 0.0
        for label, share in pop_bucket_share.items():
            r_nz, r_mixed = aggregate_composition(train_comp[corner_name], train_by_bucket[label])
            nz += share * r_nz
            mixed += share * r_mixed
        return nz, mixed

    # ---- 3. residual population factor k, calibrated on the 5 populated corners
    print(f"\nresidual population factor k ({population}100 actual / drive-matched expectation):")
    k_nz_list, k_mixed_list = [], []
    for a in topology.anchor_names:
        true_nz, true_mixed = aggregate_composition(
            per_cell_fall_power_composition(pop_anchor_libs[a]), pop_cells)
        exp_nz, exp_mixed = expected(a)
        k_nz_list.append(true_nz / exp_nz)
        k_mixed_list.append(true_mixed / exp_mixed)
        print(f"  {a:14s} near-zero {100*true_nz:6.3f}% vs expected {100*exp_nz:6.3f}%  "
              f"(k={k_nz_list[-1]:.3f})   mixed-tables {100*true_mixed:6.2f}% vs "
              f"{100*exp_mixed:6.2f}%  (k={k_mixed_list[-1]:.3f})")
    k_nz, k_mixed = float(np.mean(k_nz_list)), float(np.mean(k_mixed_list))
    sd_nz, sd_mixed = float(np.std(k_nz_list)), float(np.std(k_mixed_list))
    print(f"  k(near-zero) = {k_nz:.3f} +- {sd_nz:.3f}     "
          f"k(mixed-tables) = {k_mixed:.3f} +- {sd_mixed:.3f}   (sd across the 5 corners)")

    # ---- 4. reweight
    heldout_cells = sorted(set(d["cell"].tolist()))
    print(f"\nreweighting {len(heldout_cells)} held-out cells' errors to the "
          f"{population}100 composition, per delivery corner:")
    m_nz: Dict[str, float] = {}
    m_mixed: Dict[str, float] = {}
    for c in sorted(set(corner.tolist())):
        base_nz, base_mixed = aggregate_composition(train_comp[c], heldout_cells)
        exp_nz, exp_mixed = expected(c)
        m_nz[c] = composition_multiplier(k_nz, exp_nz, base_nz)
        m_mixed[c] = composition_multiplier(k_mixed, exp_mixed, base_mixed)
        print(f"  {c:14s} near-zero x{m_nz[c]:.3f}   mixed-tables x{m_mixed[c]:.3f}")

    group_ids = np.char.add(np.char.add(corner.astype("<U16"), "|"), sub)

    def multipliers(flip_proxy: str, nz_proxy: str) -> Dict[str, float]:
        src = {"nz": m_nz, "mixed": m_mixed}
        out = {}
        for c in set(corner.tolist()):
            out[f"{c}|{SUBGROUP_FLIP}"] = src[flip_proxy][c]
            out[f"{c}|{SUBGROUP_NEAR_ZERO}"] = src[nz_proxy][c]
        return out

    central = reweighted_pooled_score(errs, group_ids, multipliers("mixed", "nz"))
    alt_a = reweighted_pooled_score(errs, group_ids, multipliers("nz", "nz"))
    alt_b = reweighted_pooled_score(errs, group_ids, multipliers("mixed", "mixed"))
    lo, hi = min(central, alt_a, alt_b), max(central, alt_a, alt_b)

    print(f"\n=== {population}-composition reweighted pooled score "
          f"(drive-matched, scheme {scheme!r}) ===")
    print(f"  measured (training-cell composition): {official:.4f}")
    print(f"  central (flip~mixed-table, near-zero~near-zero): {central:.4f}")
    print(f"  proxy sensitivity band: [{lo:.4f}, {hi:.4f}]")
    print(f"  uplift vs measured: {central - official:+.4f}")
    print(f"  NOTE: this number is conditional on --bucket-scheme {scheme!r}. Nothing "
          f"selects that choice, and it moves the answer by ~0.23 -- run "
          f"scripts/phase4_composition_sensitivity.py for the full range.")

    print("\nper-corner measured vs reweighted:")
    for c in sorted(set(corner.tolist())):
        mask = corner == c
        print(f"  {c:14s} measured {score_from_errors(errs[mask]):8.4f}   "
              f"reweighted {reweighted_pooled_score(errs[mask], group_ids[mask], multipliers('mixed', 'nz')):8.4f}")

    if expect is not None:
        delta = central - expect
        ok = abs(delta) < 0.005
        print(f"\n  REGRESSION GATE: expected {expect:.4f}, got {central:.4f} "
              f"(delta {delta:+.4f}) -- {'PASS' if ok else 'FAIL'}")
        return ok
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", help=".npz produced by phase4_final_validate.py --dump-errors")
    ap.add_argument("--population", default="alpha", choices=sorted(STAGE_TOPOLOGIES),
                    help="which official 100-cell population to reweight to (default alpha)")
    ap.add_argument("--expect", type=float, default=None,
                    help="regression gate: fail (exit 1) if the central estimate differs "
                         "from this by >= 0.005")
    ap.add_argument("--bucket-scheme", required=True, choices=sorted(BUCKET_SCHEMES),
                    help="drive-strength bucketing. REQUIRED and deliberately without a "
                         "default: no criterion selects one (see this module's docstring), "
                         "the choice moves the answer by ~0.23, so it must be stated by "
                         "whoever quotes the number. 'none' disables drive matching "
                         "entirely, reproducing the superseded anchor-ratio audit.")
    a = ap.parse_args()
    sys.exit(0 if main(a.dump, a.population, a.expect, a.bucket_scheme) else 1)
