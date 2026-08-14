"""Re-derive every score printed in docs/algorithm_report.html.

The report claims its numbers are computed from the error dumps rather
than copied out of prose. That claim is only worth anything if the
computation still exists, and docs/current_status.md section 8 lesson 1
is about exactly this failure mode: the original composition audit lived
only in a session transcript, and rebuilding it revealed the method had
±0.23 of freedom the write-up never pinned down.

scripts/phase4_topology_compare.py already covers section 8 of
docs/round_20260810.md -- the three pooled scores, the flip subgroup's
share of squared error, the paired direction statistics, and the
huber/mse comparison. It does NOT cover three things the report puts on
the page, and this file exists for those:

  per-corner   the 30 scores behind the report's 7.1 and 7.2 bar charts
               (10 target corners x 3 topologies)
  subgroups    the FULL four-way table of 9.1 -- share of points, score,
               and e^2 mass for flip / near-zero / bulk / other. The
               compare script prints only the flip row's e^2 share.
  heatmap      the 60 cells of 7.3: alpha's score per (corner, table
               type), plus the six per-table-type totals

Everything is read from output/_phase4_cache/, which is gitignored --
each merged dump is about 40 MB. Without them this prints which files
are missing and exits non-zero rather than guessing.

Usage:
    python3 scripts/phase4_report_numbers.py
    python3 scripts/phase4_report_numbers.py --json     # machine-readable
    python3 scripts/phase4_report_numbers.py --cache-dir output/_phase4_cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scoring.audits import assign_subgroups, subgroup_stats  # noqa: E402
from scoring.scorer import point_errors, score_from_errors  # noqa: E402

# The merged huber dumps, one per topology. Same files section 8 of
# round_20260810.md uses; see docs/current_status.md section 7 for how
# they were produced and which per-corner dumps they were merged from.
MERGED = {
    "alpha": "alpha_validate_huber_s1_errors.npz",
    "beta": "beta_validate_huber_s1_errors.npz",
    "final": "final_validate_huber_s1_errors.npz",
}

# The report's heatmap keeps this column order rather than alphabetical:
# the four delay tables first, then the two power ones, so the fall_power
# column that carries the whole gap sits at the edge where it reads.
TABLE_ORDER = ("cell_rise", "cell_fall", "rise_transition",
               "fall_transition", "rise_power", "fall_power")

# Reported for alpha only -- that is the topology the report's section 7.3
# breaks down, and one 60-cell table is the point of the figure.
HEATMAP_STAGE = "alpha"


def collect(cache_dir: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    missing = []
    for stage, fname in MERGED.items():
        path = os.path.join(cache_dir, fname)
        if not os.path.exists(path):
            missing.append(path)
            continue
        d = np.load(path, allow_pickle=True)
        yt, yp = d["y_true"], d["y_pred"]
        corner, anchor, table = d["corner"], d["nearest_anchor"], d["table_type"]
        e = point_errors(yt, yp)

        rec: dict = {"pooled": round(score_from_errors(e), 4), "n_points": int(e.size)}
        rec["per_corner"] = {
            c: round(score_from_errors(e[corner == c]), 4)
            for c in sorted(set(corner.tolist()))
        }
        stats = subgroup_stats(e, assign_subgroups(yt, anchor, table))
        total_mass = sum(s.e2_mass for s in stats)
        rec["subgroups"] = [
            {"name": s.name,
             "n_points": s.n_points,
             "share_pct": round(100 * s.share, 3),
             "score": round(s.score, 2),
             "e2_pct": round(100 * s.e2_mass / total_mass, 1)}
            for s in stats
        ]
        # The per-table split is one cheap groupby and it reads for every
        # topology, so all three get it. Only the 60-cell *heatmap* stays
        # alpha-only -- there, one such table is the point of the figure.
        rec["by_table"] = {
            t: round(score_from_errors(e[table == t]), 2) for t in TABLE_ORDER
        }
        if stage == HEATMAP_STAGE:
            rec["heatmap"] = {
                c: {t: round(score_from_errors(e[(corner == c) & (table == t)]), 2)
                    for t in TABLE_ORDER}
                for c in sorted(set(corner.tolist()))
            }
        out[stage] = rec

    if missing:
        print("缺少誤差 dump（gitignored，約 40 MB／個）：", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        print("重跑方式見 docs/current_status.md §7。", file=sys.stderr)
    return out


def report(data: Dict[str, dict]) -> None:
    print("=== 7.1／7.2 逐 corner 分數（huber, 1-seed）===")
    for stage in ("alpha", "beta", "final"):
        if stage not in data:
            continue
        rec = data[stage]
        print(f"\n  {stage}  pooled={rec['pooled']:.4f}  (n={rec['n_points']})")
        for c, s in sorted(rec["per_corner"].items(), key=lambda kv: kv[1]):
            print(f"    {c:14s} {s:8.4f}")

    print("\n=== 9.1 子群：佔點數／分數／e² 質量 ===")
    print(f"  {'子群':22s} {'佔點數':>9s} {'分數':>8s} {'e² 質量':>9s}")
    for stage in ("alpha", "beta", "final"):
        if stage not in data:
            continue
        print(f"  -- {stage}")
        for g in data[stage]["subgroups"]:
            print(f"  {g['name']:22s} {g['share_pct']:8.3f}% {g['score']:8.2f} {g['e2_pct']:8.1f}%")

    if HEATMAP_STAGE in data and "heatmap" in data[HEATMAP_STAGE]:
        rec = data[HEATMAP_STAGE]
        print(f"\n=== 7.3 {HEATMAP_STAGE} × table_type ===")
        print("  " + " " * 14 + "".join(f"{t:>17s}" for t in TABLE_ORDER))
        for c, row in rec["heatmap"].items():
            print(f"  {c:14s}" + "".join(f"{row[t]:17.2f}" for t in TABLE_ORDER))
        print(f"  {'（全部 corner）':14s}"
              + "".join(f"{rec['by_table'][t]:17.2f}" for t in TABLE_ORDER))


def main(cache_dir: str, as_json: bool) -> int:
    data = collect(cache_dir)
    if not data:
        return 1
    if as_json:
        print(json.dumps(data, ensure_ascii=False, indent=1))
    else:
        report(data)
    # A partial run still prints what it has, but must not look clean to
    # anything downstream -- the report quotes all three topologies.
    return 0 if len(data) == len(MERGED) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="output/_phase4_cache",
                    help="誤差 dump 所在目錄（預設 output/_phase4_cache）")
    ap.add_argument("--json", action="store_true",
                    help="輸出 JSON 而非表格")
    _a = ap.parse_args()
    raise SystemExit(main(_a.cache_dir, _a.json))
