"""How much of the composition audit's answer is the *method*, and how
much is choices nobody wrote down?

docs/round_20260810.md section 2 describes the drive-matched reweighting in
prose but pins down neither the drive-bucket edges, nor how the residual
population factor k is aggregated over the stage's 5 calibration corners,
nor which population supplies the base prevalence in the multiplier's
denominator. `scripts/phase4_composition_audit.py` had to pick one of
each. This sweeps all three axes so the audit's real precision is a
measured quantity rather than an assumption.

This exists as a committed script, rather than as numbers quoted in a
document, because the round it belongs to was itself triggered by exactly
that failure mode: the original section 2 computation was done ad-hoc, was
never committed, and could not be re-derived afterwards.

Both proxies' k-stability is reported (`sd_nz`, `sd_mix`), not just the
near-zero one. That matters for choosing a scheme: the central estimate
applies k(mixed-table) to the sign-flip subgroup, which carries 63% (alpha)
to 69% (beta) of the total squared-error mass, so the mixed proxy's
stability is at least as load-bearing as the near-zero proxy's -- a point
missed when `stable` was first selected. See section 7.3.

Usage:
    python3 scripts/phase4_composition_sensitivity.py DUMP.npz --population alpha
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    per_cell_fall_power_composition,
    reweighted_pooled_score,
)
from scoring.scorer import point_errors, score_from_errors

# Same names as phase4_composition_audit.BUCKET_SCHEMES (kept in sync by
# test_composition_audit.py::test_sensitivity_sweep_covers_every_audit_scheme).
SCHEMES = {
    "stable":    ((1, 2), (3, 6), (7, 14), (15, 10 ** 9)),
    "occupancy": ((1, 1), (2, 3), (4, 6), (8, 14), (16, 10 ** 9)),
    "fine":      ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 10 ** 9)),
    "coarse":    ((1, 3), (4, 8), (9, 10 ** 9)),
    "none":      ((1, 10 ** 9),),
}


def main(dump_path: str, population: str) -> None:
    d = np.load(dump_path, allow_pickle=False)
    corner = d["corner"]
    errs = point_errors(d["y_true"], d["y_pred"])
    sub = assign_subgroups(d["y_true"], d["nearest_anchor"], d["table_type"])
    group_ids = np.char.add(np.char.add(corner.astype("<U16"), "|"), sub)
    heldout = sorted(set(d["cell"].tolist()))
    corners = sorted(set(corner.tolist()))
    topo = STAGE_TOPOLOGIES[population]

    print(f"dump: {dump_path}")
    print(f"  config={d['meta_config'][0]} stage={d['meta_stage'][0]} "
          f"seeds={d['meta_n_seeds'][0]}  population={population!r}")
    print(f"  measured (held-out training-cell composition): {score_from_errors(errs):.4f}")

    print("\nparsing 15 training corners + 5 population corners...", flush=True)
    train_libs = {parse_corner_filename(str(p)).name: parse_file(str(p))
                  for p in training_set_files()}
    train_comp = {c: per_cell_fall_power_composition(lib) for c, lib in train_libs.items()}
    all_train_cells = sorted(next(iter(train_libs.values())).cells)
    pop_libs = {a: parse_file(str(stage_full_lib(population, a))) for a in topo.anchor_names}
    pop_comp = {a: per_cell_fall_power_composition(lib) for a, lib in pop_libs.items()}
    pop_cells = sorted(next(iter(pop_libs.values())).cells)

    print(f"\n{'scheme':11s} {'k-agg':7s} {'base':10s} {'k_nz':>6s} {'sd_nz':>6s} "
          f"{'k_mix':>6s} {'sd_mix':>7s} {'central':>9s} {'proxy band':>18s}")
    rows = []
    for sname, edges in SCHEMES.items():
        def bkt(c):
            dr = parse_cell_name(c).drive_strength
            for lo, hi in edges:
                if lo <= dr <= hi:
                    return f"{lo}-{hi}"
            raise ValueError(c)

        tr_by_b: dict = {}
        for c in all_train_cells:
            tr_by_b.setdefault(bkt(c), []).append(c)
        pop_share: dict = {}
        for c in pop_cells:
            pop_share[bkt(c)] = pop_share.get(bkt(c), 0.0) + 1.0 / len(pop_cells)
        if any(b not in tr_by_b for b in pop_share):
            print(f"{sname:11s} SKIP (a population bucket is empty in train400)")
            continue

        def expected(cn):
            nz = mx = 0.0
            for b, sh in pop_share.items():
                r_nz, r_mx = aggregate_composition(train_comp[cn], tr_by_b[b])
                nz += sh * r_nz
                mx += sh * r_mx
            return nz, mx

        truths = {a: aggregate_composition(pop_comp[a], pop_cells) for a in topo.anchor_names}
        exps = {a: expected(a) for a in topo.anchor_names}
        ratios_nz = [truths[a][0] / exps[a][0] for a in topo.anchor_names]
        ratios_mx = [truths[a][1] / exps[a][1] for a in topo.anchor_names]
        sd_nz, sd_mx = float(np.std(ratios_nz)), float(np.std(ratios_mx))

        for kagg in ("mean", "pooled"):
            if kagg == "mean":
                k_nz, k_mx = float(np.mean(ratios_nz)), float(np.mean(ratios_mx))
            else:
                k_nz = sum(truths[a][0] for a in topo.anchor_names) / sum(exps[a][0] for a in topo.anchor_names)
                k_mx = sum(truths[a][1] for a in topo.anchor_names) / sum(exps[a][1] for a in topo.anchor_names)
            for base_name, base_cells in (("heldout80", heldout), ("train400", all_train_cells)):
                m_nz, m_mx = {}, {}
                for c in corners:
                    b_nz, b_mx = aggregate_composition(train_comp[c], base_cells)
                    e_nz, e_mx = expected(c)
                    m_nz[c] = k_nz * e_nz / b_nz
                    m_mx[c] = k_mx * e_mx / b_mx

                def mult(fp, nzp):
                    src = {"nz": m_nz, "mixed": m_mx}
                    return {f"{c}|{g}": src[p][c]
                            for c in corners
                            for g, p in ((SUBGROUP_FLIP, fp), (SUBGROUP_NEAR_ZERO, nzp))}

                cen = reweighted_pooled_score(errs, group_ids, mult("mixed", "nz"))
                a1 = reweighted_pooled_score(errs, group_ids, mult("nz", "nz"))
                a2 = reweighted_pooled_score(errs, group_ids, mult("mixed", "mixed"))
                rows.append(cen)
                print(f"{sname:11s} {kagg:7s} {base_name:10s} {k_nz:6.3f} {sd_nz:6.3f} "
                      f"{k_mx:6.3f} {sd_mx:7.3f} {cen:9.4f} "
                      f"[{min(cen, a1, a2):7.4f},{max(cen, a1, a2):7.4f}]")

    print(f"\n=== {len(rows)} reconstructions: central spans "
          f"{min(rows):.4f} - {max(rows):.4f}  (spread {max(rows) - min(rows):.4f}) ===")
    print("  Every row is a defensible reading of docs/round_20260810.md section 2. "
          "Quote the scheme alongside any number taken from this audit.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", help=".npz produced by phase4_final_validate.py --dump-errors")
    ap.add_argument("--population", default="alpha", choices=sorted(STAGE_TOPOLOGIES))
    a = ap.parse_args()
    main(a.dump, a.population)
