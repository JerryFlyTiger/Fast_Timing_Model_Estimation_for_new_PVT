"""Phase 2.5: fit the physical scaling model (src/models/phase2_scaling.py)
on all 5 full corners and use it to predict the 10 partial-corner
tables, writing filled .lib files to output/ (overwriting the Phase 2
baseline). Also runs the Phase 2.5 physical audits
(src/scoring/audits.py, docs/phase2_review.md item 3) against the
freshly-written output and prints a full report -- see
docs/phase25_results.md for the recorded numbers this script produces.

Every one of the 10 partial corners has a same-process, same-temperature
full corner available (see docs/phase2_results.md), so
`models.phase2_scaling.select_anchors` always picks exactly one anchor
here and the temperature term of the model is never exercised (Delta_T
== 0 for every prediction) -- only the alpha-power voltage-scaling term
does real work for this script. `use_process_offset` is deliberately
never passed to `predict_corner` here (defaults to False): the explicit
per-process offset (docs/phase2_review.md item 1) is a LOCO-only term,
see models.phase2_scaling's module docstring.

Usage: python3 scripts/phase2_predict.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from features.corners import parse_corner_filename
from liberty.parser import parse_file, parse_text
from liberty.writer import fill_template
from models.phase2_scaling import boundary_hit_report, fit_phase2_model, new_stats, predict_corner, select_anchors
from scoring.audits import (
    check_delay_scaling_bands,
    check_power_k_band,
    run_cross_corner_inequality_audit,
    scaling_factor_quantiles,
)
from paths import ALPHA_FULL_DIR, ALPHA_PARTIAL_DIR, OUTPUT_DIR

FULL_DIR = ALPHA_FULL_DIR
PARTIAL_DIR = ALPHA_PARTIAL_DIR


def _merge_stats(total: dict, part: dict) -> None:
    for table_type, s in part.items():
        dst = total.setdefault(table_type, new_stats())
        for k, v in s.items():
            dst[k] += v


def main():
    full_paths = sorted(FULL_DIR.glob("*.lib"))
    full_libs = {parse_corner_filename(p): parse_file(str(p)) for p in full_paths}
    print(f"fitting Phase 2.5 model on {len(full_libs)} full corners: "
          f"{', '.join(m.name for m in full_libs)}")
    model = fit_phase2_model(full_libs)

    boundary = boundary_hit_report(model)
    for table_type, p in model.params.items():
        if p.kind == "delay":
            print(
                f"  {table_type:16s} Vth={ {k: round(v, 4) for k, v in p.Vth_by_process.items()} } "
                f"(bound_hit={boundary[table_type]['Vth_at_bound']}) "
                f"alpha={p.alpha:.4f} (bound_hit={boundary[table_type]['alpha_at_bound']}) "
                f"c0_by_process={ {k: round(v, 6) for k, v in p.c0_by_process.items()} } "
                f"offset_by_process={ {k: round(v, 4) for k, v in p.offset_by_process.items()} } (LOCO-only, unused here)"
            )
        else:
            print(
                f"  {table_type:16s} k={p.k:.4f} (bound_hit={boundary[table_type]['k_at_bound']}) "
                f"c0_by_process={ {k: round(v, 6) for k, v in p.c0_by_process.items()} } "
                f"offset_by_process={ {k: round(v, 4) for k, v in p.offset_by_process.items()} } (LOCO-only, unused here)"
            )

    partial_paths = sorted(PARTIAL_DIR.glob("*.lib"))
    OUTPUT_DIR.mkdir(exist_ok=True)
    total_stats: dict = {}
    anchor_of = {}
    print()
    for path in partial_paths:
        target_meta = parse_corner_filename(path)
        target_lib = parse_file(str(path))
        anchors = select_anchors(target_meta, full_libs)
        stats: dict = {}
        predictions = predict_corner(model, target_lib, target_meta, anchors, full_libs, stats=stats)
        filled_text = fill_template(target_lib, predictions)
        out_path = OUTPUT_DIR / path.name
        out_path.write_text(filled_text, encoding="ascii")
        anchor_names = ", ".join(a.name for a in anchors)
        print(f"{path.name:40s} <- {anchor_names:16s} ({len(predictions)} tables filled)")
        _merge_stats(total_stats, stats)
        anchor_of[target_meta.name] = anchors[0].name

    print()
    print("=== aggregate robustness stats across all 10 predicted corners ===")
    for table_type, s in total_stats.items():
        print(
            f"  {table_type:16s} points={s['n_points']:8d} "
            f"gain_clipped={s['n_gain_clipped']:7d} "
            f"delta_clipped={s['n_delta_clipped']:7d} "
            f"monotonic_fixes={s['n_monotonic_violations']:6d} "
            f"shrunk_calls={s['n_shrunk_calls']}/{s['n_calls']}"
        )

    # ---- Phase 2.5 physical audits (docs/phase2_review.md item 3), run
    # against the just-written output/*.lib files, re-parsed exactly the
    # way any downstream consumer would read them. ----
    print()
    print("=== physical audit: cross-corner process-ordering inequalities ===")
    predicted = {parse_corner_filename(p).name: parse_text((OUTPUT_DIR / p.name).read_text())
                 for p in partial_paths}
    truth = {m.name: lib for m, lib in full_libs.items()}
    anchor_libs_by_name = {m.name: lib for m, lib in full_libs.items()}
    inequality_results = run_cross_corner_inequality_audit(predicted, truth)
    for r in inequality_results:
        print("  " + r.summary_line())
        for tt, (v, n) in sorted(r.by_table_type.items()):
            print(f"      {tt:16s} {v}/{n}")

    print()
    print("=== physical audit: scaling-factor distribution (predicted/anchor) ===")
    for name in sorted(predicted):
        rows = scaling_factor_quantiles(name, predicted[name], anchor_libs_by_name[anchor_of[name]])
        for row in rows:
            print(f"  {row.corner:14s} {row.table_type:16s} n={row.n_points:7d} "
                  f"p1={row.p1:8.4f} p50={row.p50:8.4f} p99={row.p99:8.4f}")

    print()
    print("=== physical audit: band checks ===")
    for c in check_delay_scaling_bands(predicted, anchor_libs_by_name, anchor_of):
        print("  " + c.summary_line())
    for c in check_power_k_band(model.params):
        print("  " + c.summary_line())

    all_passed = all(r.passed for r in inequality_results) and all(
        c.passed for c in check_delay_scaling_bands(predicted, anchor_libs_by_name, anchor_of)
    ) and all(c.passed for c in check_power_k_band(model.params))
    print()
    print(f"=== overall physical audit: {'PASS' if all_passed else 'FAIL'} ===")


if __name__ == "__main__":
    main()
