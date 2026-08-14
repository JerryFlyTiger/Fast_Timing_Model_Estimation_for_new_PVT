"""Alpha-vs-training pathology composition check (2026-08-09).

Motivation: the 96.45 CV verdict is measured on the 400 TRAINING cells,
but the official deliverable is scored on the 100 ALPHA cells. The
pathological error mass (fall_power near-zero / sign-flip points) is a
property of which cells/arcs have small net fall_power -- observable at
the 5 anchor corners, where BOTH populations have real values. If the
alpha population is systematically cleaner, the CV number UNDERESTIMATES
the official-test score.

Result (2026-08-09 run): alpha cells are ~2x cleaner on every proxy --
near-zero point share 2.2/1.7/2.0% vs training 5.2/3.5/4.4% (ss/ff/tt
anchors), mixed-sign table share 0.31% vs 1.68% (ss m40c). Reweighting
the measured error budget to alpha composition puts the CURRENT model's
expected official-test score at ~97.1-97.9 (vs 96.45 CV), before the
train-on-all-400 and seed-ensemble gains that CV also excludes.
"""
import sys

import numpy as np

sys.path.insert(0, "src")
from liberty.parser import parse_file  # noqa: E402

PAIRS = [
    ("ss0p81vm40c", "testcase/training_set/base_nom_0p9v/lib1_ss0p81vm40c_base_400.tlib",
                    "testcase/alpha_test/full/lib1_ss0p81vm40c_alpha_100.lib"),
    ("ff0p99vm40c", "testcase/training_set/base_nom_0p9v/lib1_ff0p99vm40c_base_400.tlib",
                    "testcase/alpha_test/full/lib1_ff0p99vm40c_alpha_100.lib"),
    ("tt0p9v25c",   "testcase/training_set/base_nom_0p9v/lib1_tt0p9v25c_base_400.tlib",
                    "testcase/alpha_test/full/lib1_tt0p9v25c_alpha_100.lib"),
]

for corner, tr_path, al_path in PAIRS:
    for tag, path in (("train400", tr_path), ("alpha100", al_path)):
        lib = parse_file(path)
        n = nz = mixed_tables = ntab = allzero = 0
        for key, t in lib.tables_by_key.items():
            if key[-1] != "fall_power" or t.values is None:
                continue
            v = t.values.ravel()
            ntab += 1
            if np.all(v == 0):
                allzero += 1
                continue
            n += v.size
            nz += int(np.sum((np.abs(v) < 1e-4) & (v != 0)))
            if v.min() < 0 < v.max():
                mixed_tables += 1
        print(f"{corner} {tag:8s}: fall_power tables {ntab:4d} (all-zero {allzero}), "
              f"near-zero pts {100*nz/max(1,n):5.2f}%, "
              f"mixed-sign tables {100*mixed_tables/max(1,ntab):5.2f}%")
    print()
