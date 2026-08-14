"""Phase 4 user-acceptance validation protocol (docs/plan.md Phase 4,
2026-07-26 official-training-set rewrite).

Runs the full 320/80 cell-split validation protocol on the real 400-cell
training set:

1. Load the 15 training-set corners (5 anchors + 10 delivery) and the 100
   alpha cells' 5 anchor corners (for the family-code vocabulary and the
   final alpha inference feature build).
2. Split the 400 training cells into 320 train / 80 validation cells with
   a fixed seed (models.phase4_features.PHASE4_CELL_SPLIT_SEED). The 80
   validation cells are **never** used for anything except final scoring.
3. Build one flat, corner-agnostic feature matrix per split (train, val)
   from the 5 anchor corners (models.phase4_features.build_base_dataset).
4. For each of the 10 delivery corners: fit a GBDT and an MLP on the 320
   train cells (early stopping on a dev subset carved from the 320,
   models.phase4_features.PHASE4_DEV_SPLIT_SEED -- never the 80), then
   score both on the 80 validation cells with scoring.scorer's contest
   formula.
5. Report the pooled score across all 10 corners (GBDT and MLP each),
   broken down by corner and by table_type, and report whether the
   winning model clears SCORE_THRESHOLD (98) -- the prerequisite for
   scripts/phase4_predict.py to fill in the real 10 alpha partial-corner
   outputs.

Usage: python3 scripts/phase4_validate.py
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
from models.phase4_features import (
    ANCHOR_CORNER_NAMES,
    DELIVERY_CORNER_NAMES,
    PHASE4_CELL_SPLIT_SEED,
    PHASE4_DEV_SPLIT_SEED,
    SCORE_THRESHOLD,
    build_arc_attr_index,
    build_base_dataset,
    build_family_vocab_for_phase4,
    extract_raw_values,
    make_label,
    reconstruct_predictions,
    score_breakdown,
    split_cells,
    split_dev,
    trainable_mask,
)
from models.phase4_gbdt import fit_gbdt, predict_gbdt
from models.phase4_mlp import fit_mlp, predict_mlp
from paths import ALPHA_FULL_DIR, training_set_files

CACHE_DIR = REPO_ROOT / "output" / "_phase4_cache"


def _cell_mask(cell_arr: np.ndarray, cells) -> np.ndarray:
    return np.isin(cell_arr, np.asarray(list(cells)))


def _load_training_libs():
    libs_by_name = {}
    for path in training_set_files():
        meta = parse_corner_filename(str(path))
        libs_by_name[meta.name] = parse_file(str(path))
    return libs_by_name


def _print_breakdown(title: str, overall: float, n: int, breakdown: dict) -> None:
    print(f"  {title}: overall={overall:.4f}  (n={n})")
    for tt, (score, npts) in sorted(breakdown["table_type"].items()):
        print(f"      {tt:18s} {score:8.4f}  (n={npts})")


def main() -> None:
    t_start = time.time()

    print("loading 15 training-set corner files (400 cells each)...")
    t0 = time.time()
    libs_by_name = _load_training_libs()
    print(f"  loaded {len(libs_by_name)} corners in {time.time() - t0:.1f}s: {sorted(libs_by_name)}")
    assert set(ANCHOR_CORNER_NAMES) <= set(libs_by_name)
    assert set(DELIVERY_CORNER_NAMES) <= set(libs_by_name)

    anchor_libs = {name: libs_by_name[name] for name in ANCHOR_CORNER_NAMES}
    target_libs = {name: libs_by_name[name] for name in DELIVERY_CORNER_NAMES}

    all_cells = sorted(anchor_libs[ANCHOR_CORNER_NAMES[0]].cells)
    assert len(all_cells) == 400, f"expected 400 training cells, found {len(all_cells)}"

    print("\nloading 1 alpha full-corner file (for the family-code vocabulary only)...")
    alpha_probe_path = sorted(ALPHA_FULL_DIR.glob("*.lib"))[0]
    alpha_probe_lib = parse_file(str(alpha_probe_path))
    alpha_cells = sorted(alpha_probe_lib.cells)
    assert len(alpha_cells) == 100
    assert not (set(all_cells) & set(alpha_cells)), "training/alpha cell overlap -- data contamination"

    train_cells, val_cells = split_cells(all_cells, seed=PHASE4_CELL_SPLIT_SEED)
    dev_train_cells, dev_val_cells = split_dev(train_cells, seed=PHASE4_DEV_SPLIT_SEED)
    print(f"\ncell split (seed={PHASE4_CELL_SPLIT_SEED}, recorded in models.phase4_features):")
    print(f"  {len(train_cells)} train cells, {len(val_cells)} validation cells (NEVER touched below except scoring)")
    print(f"  train cells: {train_cells}")
    print(f"  validation cells: {val_cells}")
    print(f"  dev split (seed={PHASE4_DEV_SPLIT_SEED}) of the 320 train cells: "
          f"{len(dev_train_cells)} dev-train / {len(dev_val_cells)} dev-val")

    arc_attr_index = build_arc_attr_index(anchor_libs[ANCHOR_CORNER_NAMES[0]])
    family_vocab = build_family_vocab_for_phase4(all_cells, alpha_cells)

    print("\nbuilding TRAIN feature matrix (320 cells)...")
    t0 = time.time()
    ds_train = build_base_dataset(anchor_libs, train_cells, arc_attr_index, family_vocab)
    print(f"  n={ds_train.n}  build_time={time.time() - t0:.1f}s")

    print("building VALIDATION feature matrix (80 held-out cells)...")
    t0 = time.time()
    ds_val = build_base_dataset(anchor_libs, val_cells, arc_attr_index, family_vocab)
    print(f"  n={ds_val.n}  build_time={time.time() - t0:.1f}s")

    # Leakage guard: no validation cell may appear in the training matrix.
    assert not (set(ds_train.cell.tolist()) & set(val_cells)), "LEAKAGE: a validation cell appears in ds_train"
    assert not (set(ds_val.cell.tolist()) & set(train_cells)), "LEAKAGE: a train cell appears in ds_val"

    is_dev_train_cellmask = _cell_mask(ds_train.cell, dev_train_cells)
    is_dev_val_cellmask = _cell_mask(ds_train.cell, dev_val_cells)

    per_corner = {}
    gbdt_all_errs, mlp_all_errs, ens_all_errs = [], [], []

    for target_name in DELIVERY_CORNER_NAMES:
        print(f"\n=== delivery corner: {target_name} ===")
        target_lib = target_libs[target_name]

        y_train_true = extract_raw_values(target_lib, ds_train.keys)
        nearest_train = ds_train.nearest_anchor(target_name)
        label_train = make_label(nearest_train, y_train_true)
        train_ok = trainable_mask(nearest_train)

        is_dev_train = train_ok & is_dev_train_cellmask
        is_dev_val = train_ok & is_dev_val_cellmask
        print(f"  dev-train rows={int(is_dev_train.sum())}  dev-val rows={int(is_dev_val.sum())}")

        gbdt_res = fit_gbdt(
            ds_train.X[is_dev_train], label_train[is_dev_train],
            ds_train.X[is_dev_val], label_train[is_dev_val],
        )
        print(f"  GBDT: train_time={gbdt_res.train_seconds:.1f}s best_n_iter={gbdt_res.best_n_iter}")

        mlp_res = fit_mlp(
            ds_train.X[is_dev_train], label_train[is_dev_train],
            ds_train.X[is_dev_val], label_train[is_dev_val],
        )
        print(f"  MLP:  train_time={mlp_res.train_seconds:.1f}s best_epoch={mlp_res.best_epoch}")

        y_val_true = extract_raw_values(target_lib, ds_val.keys)
        nearest_val = ds_val.nearest_anchor(target_name)
        groups_val = {"table_type": ds_val.table_type}

        gbdt_y = predict_gbdt(gbdt_res, ds_val.X)
        gbdt_pred = reconstruct_predictions(nearest_val, gbdt_y)
        gbdt_overall, gbdt_n, gbdt_breakdown = score_breakdown(y_val_true, gbdt_pred, groups_val)
        _print_breakdown("GBDT", gbdt_overall, gbdt_n, gbdt_breakdown)

        mlp_y = predict_mlp(mlp_res, ds_val.X)
        mlp_pred = reconstruct_predictions(nearest_val, mlp_y)
        mlp_overall, mlp_n, mlp_breakdown = score_breakdown(y_val_true, mlp_pred, groups_val)
        _print_breakdown("MLP", mlp_overall, mlp_n, mlp_breakdown)

        # Lever 6: simple ensemble -- average the two models' predicted
        # log-ratios (geometric mean on the linear scale) before
        # reconstructing once. Reported purely as a diagnostic/backstop;
        # does not affect which single model would be retrained for
        # delivery (scripts/phase4_predict.py picks one model type).
        ens_y = 0.5 * (gbdt_y + mlp_y)
        ens_pred = reconstruct_predictions(nearest_val, ens_y)
        ens_overall, ens_n, ens_breakdown = score_breakdown(y_val_true, ens_pred, groups_val)
        _print_breakdown("ENSEMBLE", ens_overall, ens_n, ens_breakdown)

        from scoring.scorer import point_errors
        gbdt_all_errs.append(point_errors(y_val_true, gbdt_pred))
        mlp_all_errs.append(point_errors(y_val_true, mlp_pred))
        ens_all_errs.append(point_errors(y_val_true, ens_pred))

        per_corner[target_name] = {
            "gbdt": (gbdt_overall, gbdt_n, gbdt_breakdown, gbdt_res.best_n_iter, gbdt_res.train_seconds),
            "mlp": (mlp_overall, mlp_n, mlp_breakdown, mlp_res.best_epoch, mlp_res.train_seconds),
            "ensemble": (ens_overall, ens_n, ens_breakdown, None, None),
        }

    from scoring.scorer import score_from_errors

    gbdt_pooled = score_from_errors(np.concatenate(gbdt_all_errs))
    ens_pooled = score_from_errors(np.concatenate(ens_all_errs))
    mlp_pooled = score_from_errors(np.concatenate(mlp_all_errs))

    print("\n=== summary: per-corner overall scores ===")
    print(f"  {'corner':16s} {'GBDT':>10s} {'MLP':>10s} {'ENSEMBLE':>10s}")
    for name in DELIVERY_CORNER_NAMES:
        g = per_corner[name]["gbdt"][0]
        m = per_corner[name]["mlp"][0]
        e = per_corner[name]["ensemble"][0]
        print(f"  {name:16s} {g:10.4f} {m:10.4f} {e:10.4f}")

    print("\n=== summary: per-corner x per-table_type scores (GBDT) ===")
    for name in DELIVERY_CORNER_NAMES:
        bd = per_corner[name]["gbdt"][2]["table_type"]
        print(f"  {name}: " + ", ".join(f"{tt}={s:.2f}" for tt, (s, _n) in sorted(bd.items())))

    print("\n=== summary: per-corner x per-table_type scores (MLP) ===")
    for name in DELIVERY_CORNER_NAMES:
        bd = per_corner[name]["mlp"][2]["table_type"]
        print(f"  {name}: " + ", ".join(f"{tt}={s:.2f}" for tt, (s, _n) in sorted(bd.items())))

    print("\n=== summary: per-corner x per-table_type scores (ENSEMBLE) ===")
    for name in DELIVERY_CORNER_NAMES:
        bd = per_corner[name]["ensemble"][2]["table_type"]
        print(f"  {name}: " + ", ".join(f"{tt}={s:.2f}" for tt, (s, _n) in sorted(bd.items())))

    print(f"\n=== pooled overall across all 10 delivery corners (80 held-out cells) ===")
    print(f"  GBDT     pooled: {gbdt_pooled:.4f}")
    print(f"  MLP      pooled: {mlp_pooled:.4f}")
    print(f"  ENSEMBLE pooled: {ens_pooled:.4f}")

    winner_name, winner_score = max(
        [("GBDT", gbdt_pooled), ("MLP", mlp_pooled), ("ENSEMBLE", ens_pooled)], key=lambda kv: kv[1]
    )
    print(f"\n  winner: {winner_name} = {winner_score:.4f}")
    print(f"  SCORE_THRESHOLD = {SCORE_THRESHOLD}")
    if winner_score >= SCORE_THRESHOLD:
        print(f"  RESULT: PASS -- {winner_name} clears the {SCORE_THRESHOLD} gate; "
              f"scripts/phase4_predict.py may proceed.")
    else:
        print(f"  RESULT: FAIL -- best score {winner_score:.4f} < {SCORE_THRESHOLD}; "
              f"no delivery output should be produced from this model.")

    print(f"\ntotal wall time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
