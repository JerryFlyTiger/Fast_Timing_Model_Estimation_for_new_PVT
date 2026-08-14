"""Phase 4 FINAL delivery predict (2026-07-27, post improvement-round
screening; supersedes `scripts/phase4_predict.py` with the single,
already-chosen production config -- see `docs/phase4b_screen_log.md` /
`docs/phase4_results.md`).

**Only run this after `scripts/phase4_final_validate.py` has printed
`RESULT: PASS` for the same `--config`.** This script does not
re-verify that gate itself (matching `scripts/phase4_predict.py`'s
original convention) -- it trusts the caller ran the acceptance script
first. Running it on a FAIL'd config produces output that should not be
submitted.

**Zero-leakage declaration**:

- Trains on all 400 official training cells (there is no held-out split
  at this stage -- the 80/20 acceptance check already happened in
  `scripts/phase4_final_validate.py`; retraining on the full 400 for
  delivery is standard practice and does not reintroduce any leakage
  into the *acceptance number*, which was already finalized).
- The only split here is a *fresh* internal dev-train/dev-val carve-out
  of the full 400 (`models.phase4_features.split_dev`, same seed
  convention), used solely to decide how many boosting rounds / epochs
  the production model gets -- this is a training-time hyperparameter
  decision, not a scoring decision, and the acceptance verdict never
  depended on it either way.
- Prediction runs on the 100 *alpha* cells, which have zero overlap with
  the 400 training cells (asserted at runtime below) -- this is the same
  unseen-cell generalization the acceptance protocol measured.
- Output is produced by `liberty.writer.fill_template_file`, which only
  ever overwrites blank `values(...)` slots in the official partial
  templates byte-for-byte; no other byte of the template is touched.

Safety checks enforced (docs/plan.md hard rules):

- NaN/Inf guard: `reconstruct_predictions`'s clip bounds the exponent so
  no finite input can overflow to inf, and every corner's final
  prediction array is asserted finite before being handed to the writer
  -- a non-finite value fails loudly instead of silently corrupting the
  output file.
- All-zero power table rule (docs/plan.md rule 3): a table whose nearest
  anchor is exactly zero (the known-invalid rise_power/fall_power arc
  marker) is *always* reconstructed to an exact 0, regardless of the
  model's own output (`reconstruct_predictions`'s `nearest_anchor == 0`
  override) -- verified structurally consistent across every training
  corner in `tests/test_phase4_features.py`.

**Seed ensemble** (2026-08-09 addition, `--seeds N`, default 1): fit N
independently-seeded models per delivery corner on the SAME feature
matrix and average the predicted log-ratios before reconstructing --
identical convention (and SEED_BASE) to
`scripts/phase4_final_validate.py --seeds N`, so the delivered model
matches what the acceptance/audit run measured.

Usage:
    python3 scripts/phase4_final_predict.py                      # default config (mlp_w256_b4_full)
    python3 scripts/phase4_final_predict.py --config gbdt_full
    python3 scripts/phase4_final_predict.py --seeds 3            # 3-seed ensemble delivery
    python3 scripts/phase4_final_predict.py --corners ss0p72vm40c --dry-run   # smoke test, no files written
    python3 scripts/phase4_final_predict.py --help
"""

from __future__ import annotations

import argparse
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
from models.phase4_final_config import CONFIGS, DEFAULT_CONFIG_TAG, fit_config
from paths import ALPHA_FULL_DIR, ALPHA_PARTIAL_DIR, OUTPUT_DIR, training_set_files

SEED_BASE = 20260729  # MUST match scripts/phase4_final_validate.py's SEED_BASE


def _load_named_libs(paths):
    out = {}
    for path in paths:
        meta = parse_corner_filename(str(path))
        out[meta.name] = parse_file(str(path))
    return out


def main(config_tag: str, corners, *, dry_run: bool = False, n_seeds: int = 1) -> None:
    assert config_tag in CONFIGS, f"unknown config tag {config_tag!r}; available: {sorted(CONFIGS)}"
    assert set(corners) <= set(DELIVERY_CORNER_NAMES)
    assert n_seeds >= 1
    t_start = time.time()

    print(f"config: {config_tag!r} -> {CONFIGS[config_tag]}  seeds={n_seeds}")
    if dry_run:
        print("--dry-run: predictions computed and finiteness-checked, but no output files written")

    print("loading 15 training-set corner files (400 cells each)...")
    libs_by_name = _load_named_libs(training_set_files())
    anchor_libs = {name: libs_by_name[name] for name in ANCHOR_CORNER_NAMES}
    target_libs = {name: libs_by_name[name] for name in corners}
    all_cells = sorted(anchor_libs[ANCHOR_CORNER_NAMES[0]].cells)
    assert len(all_cells) == 400

    print("loading 5 alpha full-corner files (100 cells)...")
    alpha_full_libs = _load_named_libs(sorted(ALPHA_FULL_DIR.glob("*.lib")))
    assert set(ANCHOR_CORNER_NAMES) <= set(alpha_full_libs)
    alpha_cells = sorted(alpha_full_libs[ANCHOR_CORNER_NAMES[0]].cells)
    assert len(alpha_cells) == 100
    assert not (set(all_cells) & set(alpha_cells)), "training/alpha cell overlap -- data contamination"

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
    assert not (set(ds_alpha.cell.tolist()) & set(all_cells)), "LEAKAGE: a training cell appears in ds_alpha"

    # Fresh internal dev split for the final fit's own early stopping --
    # a training-time decision, not an acceptance check (see module
    # docstring). Uses the FULL 400 cells (no held-out split at this stage).
    dev_train_cells, dev_val_cells = split_dev(all_cells, seed=PHASE4_DEV_SPLIT_SEED, dev_train_frac=0.8)
    is_dev_train_cellmask = np.isin(ds_full.cell, np.asarray(dev_train_cells))
    is_dev_val_cellmask = np.isin(ds_full.cell, np.asarray(dev_val_cells))

    if not dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for target_name in corners:
        print(f"\n=== delivery corner: {target_name} (config={config_tag}) ===")
        target_lib = target_libs[target_name]

        y_full_true = extract_raw_values(target_lib, ds_full.keys)
        nearest_full = ds_full.nearest_anchor(target_name)
        label_full = make_label(nearest_full, y_full_true)
        ok = trainable_mask(nearest_full)

        is_dev_train = ok & is_dev_train_cellmask
        is_dev_val = ok & is_dev_val_cellmask

        # Seed ensemble (--seeds N > 1): same convention as
        # scripts/phase4_final_validate.py -- N independently-seeded fits
        # on the same feature matrix, log-ratio predictions averaged in
        # log space before a single reconstruction.
        y_ratio_sum = None
        total_train_seconds = 0.0
        fit_infos = []
        for member in range(n_seeds):
            member_seed = SEED_BASE + member if n_seeds > 1 else None
            handle = fit_config(
                config_tag,
                ds_full.X[is_dev_train], label_full[is_dev_train],
                ds_full.X[is_dev_val], label_full[is_dev_val],
                seed=member_seed,
            )
            total_train_seconds += handle.train_seconds
            fit_infos.append(handle.info)
            member_ratio = handle.predict(ds_alpha.X)
            y_ratio_sum = member_ratio if y_ratio_sum is None else y_ratio_sum + member_ratio
        y_alpha_ratio = y_ratio_sum / n_seeds
        seed_note = f" [{n_seeds}-seed ensemble]" if n_seeds > 1 else ""
        print(f"  fit: train_time={total_train_seconds:.1f}s{seed_note} " + "; ".join(fit_infos))

        nearest_alpha = ds_alpha.nearest_anchor(target_name)
        pred_alpha = reconstruct_predictions(nearest_alpha, y_alpha_ratio)

        # NaN/Inf guard (docs/plan.md hard rule): fail loudly rather than
        # hand a corrupted value to the writer.
        assert np.isfinite(pred_alpha).all(), f"non-finite predictions for {target_name}"
        # All-zero power table rule (docs/plan.md rule 3), spot-checked:
        # every row whose nearest anchor is exactly 0 must reconstruct to
        # exactly 0, regardless of the model's raw output.
        zero_anchor_rows = nearest_alpha == 0.0
        assert np.all(pred_alpha[zero_anchor_rows] == 0.0), (
            f"rule-3 violation for {target_name}: a zero-anchor row did not reconstruct to exactly 0"
        )

        predictions = unravel_predictions(ds_alpha.keys, pred_alpha)

        if dry_run:
            print(f"  [dry-run] {len(predictions)} table keys predicted, all finite, rule-3 verified -- no file written")
            continue

        partial_path = ALPHA_PARTIAL_DIR / f"lib1_{target_name}_alpha_100.lib"
        output_path = OUTPUT_DIR / partial_path.name
        fill_template_file(str(partial_path), predictions, str(output_path))
        print(f"  wrote {output_path}")

    print(f"\ntotal wall time: {time.time() - t_start:.1f}s")


def _parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default=DEFAULT_CONFIG_TAG, choices=sorted(CONFIGS),
                     help=f"production config tag (models.phase4_final_config.CONFIGS); default {DEFAULT_CONFIG_TAG!r}. "
                          "Must match whichever config scripts/phase4_final_validate.py reported PASS for.")
    ap.add_argument("--corners", default="all",
                     help="comma-separated delivery corner names, or 'all' for all 10 (default). "
                          "A subset is for smoke-testing only -- do not submit a partial output/ directory.")
    ap.add_argument("--dry-run", action="store_true",
                     help="compute + finiteness/rule-3-check predictions but do not write any output/ files")
    ap.add_argument("--seeds", type=int, default=1,
                     help="number of independently-seeded models to fit per corner and average "
                          "(seed ensemble, same SEED_BASE convention as phase4_final_validate.py); "
                          "default 1.")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    corners = list(DELIVERY_CORNER_NAMES) if args.corners == "all" else [c.strip() for c in args.corners.split(",")]
    main(args.config, corners, dry_run=args.dry_run, n_seeds=args.seeds)
