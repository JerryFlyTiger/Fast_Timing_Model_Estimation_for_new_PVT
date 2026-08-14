"""Full 5-corner leave-one-corner-out (LOCO) validation for the Phase 2
scaling model (docs/plan.md Phase 2 acceptance criterion / task item 4).

This supersedes scripts/loco_reference.py's Phase 1 "4-pair" simplification
(which only checked 4 hand-picked source/target pairs and scored a naive
direct copy). Here every one of the 5 full corners is held out in turn:
the Phase2Model is *refit from scratch* on the remaining 4
(docs/plan.md: "每輪拿 4 個 corner 擬合"), used to predict the held-out
corner's tables, and scored against its ground truth with
scoring.scorer.compare_libs.

Source/anchor selection during prediction follows
`models.phase2_scaling.select_anchors`: same-process full corners are
preferred exclusively; the tt0p9v25c fold has none (tt has only one
temperature point among the full corners) and falls back to a blend of
all 4 remaining corners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping

import numpy as np

from features.corners import CornerMeta, parse_corner_filename
from liberty.parser import LibertyFile, ValueTable, parse_file
from models.phase2_scaling import (
    SHRINK_LAMBDA,
    Phase2Model,
    fit_phase2_model,
    new_stats,
    predict_corner,
    select_anchors,
)
from scoring.scorer import ScoreReport, compare_libs


@dataclass
class LocoFoldResult:
    held_out: CornerMeta
    anchors: List[CornerMeta]
    report: ScoreReport
    stats: Dict[str, dict] = field(default_factory=dict)


def load_full_corners(paths) -> Dict[CornerMeta, LibertyFile]:
    return {parse_corner_filename(p): parse_file(str(p)) for p in paths}


def _wrap_predictions_as_lib(reference_lib: LibertyFile, predictions: Mapping) -> LibertyFile:
    """Package a {TableKey: 7x7 array} prediction dict as a LibertyFile so
    it can be passed to scoring.scorer.compare_libs unmodified (that
    function only ever reads `.tables_by_key[key].values`)."""
    tables_by_key = {
        key: ValueTable(
            table_type=key[-1], index_1=(), index_2=(), is_blank=False,
            values=np.asarray(values, dtype=float), row_spans=[], key=key,
        )
        for key, values in predictions.items()
    }
    return LibertyFile(
        path=None, text="", library_name=reference_lib.library_name,
        cells={}, tables=list(tables_by_key.values()), tables_by_key=tables_by_key,
    )


def run_loco(
    libs: Dict[CornerMeta, LibertyFile], *, shrink_lambda: float = SHRINK_LAMBDA
) -> List[LocoFoldResult]:
    """Run all 5 leave-one-corner-out folds and return one LocoFoldResult
    per fold, in the same order as `libs`.

    Predictions are made with `use_process_offset=True` (docs/phase2_review.md
    item 1): LOCO's cross-process folds are exactly the case the explicit
    per-process offset term exists to help, unlike the real deliverable
    (scripts/phase2_predict.py), which never opts into it -- see
    models.phase2_scaling's module docstring.
    """
    results = []
    for held_out in list(libs):
        train = {m: lib for m, lib in libs.items() if m is not held_out}
        model = fit_phase2_model(train, shrink_lambda=shrink_lambda)
        anchors = select_anchors(held_out, train)

        held_out_lib = libs[held_out]
        stats: Dict[str, dict] = {}
        keys = list(held_out_lib.tables_by_key.keys())
        predictions = predict_corner(
            model, held_out_lib, held_out, anchors, train, keys=keys, stats=stats,
            use_process_offset=True,
        )

        predicted_lib = _wrap_predictions_as_lib(held_out_lib, predictions)
        report = compare_libs(held_out_lib, predicted_lib, keys=keys)
        results.append(LocoFoldResult(held_out=held_out, anchors=anchors, report=report, stats=stats))
    return results


def format_loco_results(results: List[LocoFoldResult]) -> str:
    lines = []
    for res in results:
        anchor_names = ", ".join(a.name for a in res.anchors)
        lines.append(f"=== held out: {res.held_out.name}  (anchor(s): {anchor_names}) ===")
        lines.append(str(res.report))
        for table_type, s in res.stats.items():
            lines.append(
                f"  [{table_type}] gain_clipped={s['n_gain_clipped']} "
                f"delta_clipped={s['n_delta_clipped']} "
                f"monotonic_fixes={s['n_monotonic_violations']} "
                f"shrunk_calls={s['n_shrunk_calls']}/{s['n_calls']}"
            )
        lines.append("")
    overall_scores = [r.report.overall for r in results]
    lines.append(f"mean overall across 5 folds: {sum(overall_scores) / len(overall_scores):.4f}")
    return "\n".join(lines)
