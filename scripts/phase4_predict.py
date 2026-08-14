"""Phase 4 delivery (docs/plan.md Phase 4 item 5): retrain the model type
that won scripts/phase4_validate.py's 320/80 acceptance protocol -- and
only if it cleared the SCORE_THRESHOLD (98) gate -- on **all 400**
official training cells (same hyperparameters as the validation run), and
use it to predict the 100 alpha cells' values at all 10 delivery corners,
filling `testcase/alpha_test/partial/*.lib` templates into `output/`.

The final fit's own early stopping uses a fresh internal dev split of the
full 400 training cells (`models.phase4_features.split_dev`, same seed
convention) -- this is *not* an acceptance check (that already happened
in scripts/phase4_validate.py on the strict 320/80 split); it only
decides how many boosting rounds / epochs the production model gets,
using data the acceptance decision never depended on either way.

Usage: python3 scripts/phase4_predict.py {gbdt,mlp}
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
from liberty.writer import fill_template_file
from models.phase4_features import (
    ANCHOR_CORNER_NAMES,
    DELIVERY_CORNER_NAMES,
    PHASE4_DEV_SPLIT_SEED,
    build_arc_attr_index,
    build_base_dataset,
    build_family_vocab_for_phase4,
    extract_raw_values,
    make_label,
    reconstruct_predictions,
    split_dev,
    trainable_mask,
    unravel_predictions,
)
from models.phase4_gbdt import fit_gbdt, predict_gbdt
from models.phase4_mlp import fit_mlp, predict_mlp
from paths import ALPHA_FULL_DIR, ALPHA_PARTIAL_DIR, OUTPUT_DIR, training_set_files


def _load_named_libs(paths):
    out = {}
    for path in paths:
        meta = parse_corner_filename(str(path))
        out[meta.name] = parse_file(str(path))
    return out


def main(model_choice: str) -> None:
    assert model_choice in ("gbdt", "mlp"), f"unknown model_choice {model_choice!r}"
    t_start = time.time()

    print("loading 15 training-set corner files (400 cells each)...")
    libs_by_name = _load_named_libs(training_set_files())
    anchor_libs = {name: libs_by_name[name] for name in ANCHOR_CORNER_NAMES}
    target_libs = {name: libs_by_name[name] for name in DELIVERY_CORNER_NAMES}
    all_cells = sorted(anchor_libs[ANCHOR_CORNER_NAMES[0]].cells)
    assert len(all_cells) == 400

    print("loading 5 alpha full-corner files (100 cells)...")
    alpha_full_libs = _load_named_libs(sorted(ALPHA_FULL_DIR.glob("*.lib")))
    assert set(ANCHOR_CORNER_NAMES) <= set(alpha_full_libs)
    alpha_cells = sorted(alpha_full_libs[ANCHOR_CORNER_NAMES[0]].cells)
    assert len(alpha_cells) == 100
    assert not (set(all_cells) & set(alpha_cells))

    arc_attr_index_train = build_arc_attr_index(anchor_libs[ANCHOR_CORNER_NAMES[0]])
    arc_attr_index_alpha = build_arc_attr_index(alpha_full_libs[ANCHOR_CORNER_NAMES[0]])
    family_vocab = build_family_vocab_for_phase4(all_cells, alpha_cells)

    print("building FULL training feature matrix (all 400 cells)...")
    t0 = time.time()
    ds_full = build_base_dataset(anchor_libs, all_cells, arc_attr_index_train, family_vocab)
    print(f"  n={ds_full.n}  build_time={time.time() - t0:.1f}s")

    print("building ALPHA inference feature matrix (100 alpha cells)...")
    t0 = time.time()
    ds_alpha = build_base_dataset(alpha_full_libs, alpha_cells, arc_attr_index_alpha, family_vocab)
    print(f"  n={ds_alpha.n}  build_time={time.time() - t0:.1f}s")

    # Fresh internal dev split for the final fit's own early stopping --
    # not an acceptance check, see module docstring.
    dev_train_cells, dev_val_cells = split_dev(all_cells, seed=PHASE4_DEV_SPLIT_SEED, dev_train_frac=0.8)
    is_dev_train_cellmask = np.isin(ds_full.cell, np.asarray(dev_train_cells))
    is_dev_val_cellmask = np.isin(ds_full.cell, np.asarray(dev_val_cells))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for target_name in DELIVERY_CORNER_NAMES:
        print(f"\n=== delivery corner: {target_name} (model={model_choice}) ===")
        target_lib = target_libs[target_name]

        y_full_true = extract_raw_values(target_lib, ds_full.keys)
        nearest_full = ds_full.nearest_anchor(target_name)
        label_full = make_label(nearest_full, y_full_true)
        ok = trainable_mask(nearest_full)

        is_dev_train = ok & is_dev_train_cellmask
        is_dev_val = ok & is_dev_val_cellmask

        if model_choice == "gbdt":
            res = fit_gbdt(
                ds_full.X[is_dev_train], label_full[is_dev_train],
                ds_full.X[is_dev_val], label_full[is_dev_val],
            )
            print(f"  GBDT: train_time={res.train_seconds:.1f}s best_n_iter={res.best_n_iter}")
            y_alpha_ratio = predict_gbdt(res, ds_alpha.X)
        else:
            res = fit_mlp(
                ds_full.X[is_dev_train], label_full[is_dev_train],
                ds_full.X[is_dev_val], label_full[is_dev_val],
            )
            print(f"  MLP: train_time={res.train_seconds:.1f}s best_epoch={res.best_epoch}")
            y_alpha_ratio = predict_mlp(res, ds_alpha.X)

        nearest_alpha = ds_alpha.nearest_anchor(target_name)
        pred_alpha = reconstruct_predictions(nearest_alpha, y_alpha_ratio)
        assert np.isfinite(pred_alpha).all(), f"non-finite predictions for {target_name}"

        predictions = unravel_predictions(ds_alpha.keys, pred_alpha)

        partial_path = ALPHA_PARTIAL_DIR / f"lib1_{target_name}_alpha_100.lib"
        output_path = OUTPUT_DIR / partial_path.name
        fill_template_file(str(partial_path), predictions, str(output_path))
        print(f"  wrote {output_path}")

    print(f"\ntotal wall time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("gbdt", "mlp"):
        print("usage: python3 scripts/phase4_predict.py {gbdt,mlp}", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
