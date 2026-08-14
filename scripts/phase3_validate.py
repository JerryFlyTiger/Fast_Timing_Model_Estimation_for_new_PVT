"""Phase 3 user-acceptance validation protocol (docs/plan.md Phase 3,
"使用者驗收門檻", 2026-07-26 addition).

Runs the full 80/20 cell-split validation protocol:

1. Split the 100 cells shared by every full corner into 80 train / 20
   validation cells with a fixed seed (models.phase3_features).
2. Fit the Phase 2.5 physical model (models.phase2_scaling) on the 80
   train cells only.
3. Build one flat sample table per split from every ordered
   (source, target) pair among the 5 full corners (20 pairs total) --
   train samples from the 80 train cells, validation samples from the 20
   validation cells, both feeding the same Phase 2.5 model for the
   "physical prediction" feature (never touching validation-cell data).
4. Fit both ML versions (models.phase3_gbdt, models.phase3_mlp) on the
   train samples, with early stopping on a dev subset carved from the 80
   train cells (never the 20 validation cells).
5. Score all three (Phase 2.5, GBDT, MLP) on the validation samples with
   scoring.scorer's contest formula, reporting the overall score, a
   breakdown by corner-pair, and a breakdown by table_type.
6. Report whether the winning model clears the SCORE_THRESHOLD (98) gate
   that is a prerequisite for scripts/phase3_predict.py to fill in the
   real 10 partial-corner outputs.

Usage: python3 scripts/phase3_validate.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from features.corners import parse_corner_filename
from liberty.parser import parse_file
from models.phase2_scaling import fit_phase2_model
from models.phase3_features import (
    SCORE_THRESHOLD,
    build_arc_attr_index,
    build_family_vocab,
    build_feature_matrix,
    filter_lib_cells,
    make_label,
    ordered_full_corner_pairs,
    reconstruct_predictions,
    score_breakdown,
    split_cells,
    split_dev,
)
from models.phase3_gbdt import fit_gbdt, predict_gbdt
from models.phase3_mlp import fit_mlp, predict_mlp
from paths import ALPHA_FULL_DIR

FULL_DIR = ALPHA_FULL_DIR
LABEL_MODE = "residual"  # chosen after comparing "raw" vs "residual" on a
# dev-cell split (never the 20% validation cells) -- see docs/model_comparison.md


def _print_breakdown(title: str, overall: float, n: int, breakdowns: dict) -> None:
    print(f"--- {title} ---")
    print(f"  overall: {overall:.4f}  (n={n})")
    print("  by corner-pair:")
    for pair, (score, npts) in sorted(breakdowns["pair"].items()):
        print(f"    {pair:28s} {score:8.4f}  (n={npts})")
    print("  by table_type:")
    for tt, (score, npts) in sorted(breakdowns["table_type"].items()):
        print(f"    {tt:18s} {score:8.4f}  (n={npts})")


def main() -> None:
    t_start = time.time()
    full_paths = sorted(FULL_DIR.glob("*.lib"))
    libs = {parse_corner_filename(p): parse_file(str(p)) for p in full_paths}
    print(f"loaded {len(libs)} full corners: {', '.join(m.name for m in libs)}")

    all_cells = sorted(next(iter(libs.values())).cells)
    assert len(all_cells) == 100, f"expected 100 cells, found {len(all_cells)}"

    train_cells, val_cells = split_cells(all_cells)
    dev_train_cells, dev_val_cells = split_dev(train_cells)
    print(f"\ncell split (seed recorded in models.phase3_features.CELL_SPLIT_SEED):")
    print(f"  {len(train_cells)} train cells, {len(val_cells)} validation cells")
    print(f"  train cells: {train_cells}")
    print(f"  validation cells: {val_cells}")
    print(f"  dev split of the 80 train cells (early stopping only): "
          f"{len(dev_train_cells)} dev-train / {len(dev_val_cells)} dev-val")

    pairs = ordered_full_corner_pairs(libs.keys())
    print(f"\n{len(pairs)} ordered cross-corner pairs: {[f'{s.name}->{t.name}' for s, t in pairs]}")

    arc_attr_index = build_arc_attr_index(next(iter(libs.values())))
    family_vocab = build_family_vocab(all_cells)

    print("\nfitting Phase 2.5 physical model on the 80 TRAIN cells only "
          "(zero validation-cell leakage into the physics feature/baseline)...")
    t0 = time.time()
    train_libs_filtered = {m: filter_lib_cells(lib, train_cells) for m, lib in libs.items()}
    phase25_model = fit_phase2_model(train_libs_filtered)
    print(f"  phase2.5 fit time: {time.time() - t0:.1f}s")

    print("\nbuilding TRAIN sample matrix (80 cells x 20 pairs)...")
    t0 = time.time()
    ds_train = build_feature_matrix(libs, train_cells, pairs, phase25_model, arc_attr_index, family_vocab)
    print(f"  n={ds_train.n}  build_time={time.time() - t0:.1f}s")

    print("building VALIDATION sample matrix (20 cells x 20 pairs)...")
    t0 = time.time()
    ds_val = build_feature_matrix(libs, val_cells, pairs, phase25_model, arc_attr_index, family_vocab)
    print(f"  n={ds_val.n}  build_time={time.time() - t0:.1f}s")

    # Leakage guard: no validation cell may appear in the training matrix.
    assert not (set(ds_train.cell.tolist()) & set(val_cells)), "LEAKAGE: a validation cell appears in ds_train"

    groups_val = {"table_type": ds_val.table_type, "pair": ds_val.pair}

    print(f"\n=== Phase 2.5 baseline (this cell-split protocol, label_mode irrelevant) ===")
    p25_overall, p25_n, p25_breakdown = score_breakdown(ds_val.target, ds_val.phase25_pred, groups_val)
    _print_breakdown("Phase 2.5 physical model", p25_overall, p25_n, p25_breakdown)

    print(f"\n=== fitting GBDT (label_mode={LABEL_MODE!r}) ===")
    gbdt_res = fit_gbdt(ds_train, dev_train_cells, dev_val_cells, label_mode=LABEL_MODE)
    print(f"  train_time={gbdt_res.train_seconds:.1f}s  best_n_iter={gbdt_res.best_n_iter}")
    print(f"  dev_val MSE curve: {[round(x, 5) for x in gbdt_res.dev_val_losses]}")
    gbdt_y = predict_gbdt(gbdt_res, ds_val.X)
    gbdt_pred = reconstruct_predictions(ds_val, gbdt_y, LABEL_MODE)
    gbdt_overall, gbdt_n, gbdt_breakdown = score_breakdown(ds_val.target, gbdt_pred, groups_val)
    _print_breakdown("GBDT (HistGradientBoostingRegressor)", gbdt_overall, gbdt_n, gbdt_breakdown)

    print(f"\n=== fitting MLP (label_mode={LABEL_MODE!r}) ===")
    mlp_res = fit_mlp(ds_train, dev_train_cells, dev_val_cells, label_mode=LABEL_MODE)
    print(f"  train_time={mlp_res.train_seconds:.1f}s  best_epoch={mlp_res.best_epoch}")
    print(f"  dev_val MSE curve: {[round(x, 5) for x in mlp_res.dev_val_losses]}")
    mlp_y = predict_mlp(mlp_res, ds_val.X)
    mlp_pred = reconstruct_predictions(ds_val, mlp_y, LABEL_MODE)
    mlp_overall, mlp_n, mlp_breakdown = score_breakdown(ds_val.target, mlp_pred, groups_val)
    _print_breakdown("MLP (PyTorch)", mlp_overall, mlp_n, mlp_breakdown)

    print("\n=== summary ===")
    print(f"  Phase 2.5 baseline : {p25_overall:8.4f}")
    print(f"  GBDT               : {gbdt_overall:8.4f}  (train {gbdt_res.train_seconds:.1f}s)")
    print(f"  MLP                : {mlp_overall:8.4f}  (train {mlp_res.train_seconds:.1f}s)")

    winner_name, winner_score = max(
        [("GBDT", gbdt_overall), ("MLP", mlp_overall), ("Phase 2.5", p25_overall)], key=lambda kv: kv[1]
    )
    print(f"\n  best of the three: {winner_name} = {winner_score:.4f}")
    print(f"  SCORE_THRESHOLD = {SCORE_THRESHOLD}")
    if winner_score >= SCORE_THRESHOLD:
        print(f"  RESULT: PASS -- {winner_name} clears the {SCORE_THRESHOLD} gate; "
              f"scripts/phase3_predict.py may proceed.")
    else:
        print(f"  RESULT: FAIL -- best score {winner_score:.4f} < {SCORE_THRESHOLD}; "
              f"no partial-corner output should be produced from this model.")

    print(f"\ntotal wall time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
