"""Phase 4 FINAL user-acceptance validation (2026-07-27, post
improvement-round screening; supersedes `scripts/phase4_validate.py`'s
GBDT-vs-MLP comparison run with a single, already-chosen production
config -- see `docs/phase4b_screen_log.md` / `docs/phase4_results.md`).

**Zero-leakage declaration** (docs/plan.md Phase 4 acceptance rule,
unchanged from `scripts/phase4_validate.py`):

- The 400 official training cells are split 320 train / 80 validation by
  a fixed seed (`models.phase4_features.PHASE4_CELL_SPLIT_SEED`). The 80
  validation cells are used *here* for scoring ONLY -- this script is the
  one and only place they are allowed to influence anything, and even
  here only as the final scoring target, never as a training input.
- Early stopping during training uses a *second*, disjoint split carved
  out of the 320 train cells only (`PHASE4_DEV_SPLIT_SEED`) -- the 80
  validation cells never appear in the dev-train/dev-val subsets (see the
  runtime assertions below, which fail loudly if this is ever violated).
- The feature matrix (`models.phase4_features.build_base_dataset`) reads
  only the 5 standard-voltage anchor corners for every cell -- it never
  reads a delivery-corner value; the delivery-corner truth is pulled
  in *only* as the label (`extract_raw_values`) for the corner currently
  being trained/scored, exactly mirroring the real alpha-inference input
  shape.

Protocol (per `--config`, defaulting to
`models.phase4_final_config.DEFAULT_CONFIG_TAG` -- the current winning
config, `mlp_w256_b4_huber` since 2026-08-11; do not hardcode a tag name
here again, it has already gone stale once):

1. Build one flat feature matrix each for the 320 train cells and the 80
   validation cells (both anchor-only, corner-agnostic -- reused across
   all 10 delivery-corner models).
2. For each of the 10 delivery corners: fit the configured model on the
   320 train cells (with internal dev-only early stopping), then score
   the fitted model on the 80 held-out cells with the contest scorer.
3. Report per-corner scores, a per-corner x per-table_type breakdown, the
   pooled score across all 10 corners x 80 cells, RESULT: PASS/FAIL
   against `SCORE_THRESHOLD` (98), and a comparison line against the
   already-measured alpha-stage official result (`ALPHA_OFFICIAL_POOLED_SCORE`)
   -- PASS is the prerequisite for `scripts/phase4_final_predict.py` to
   produce real delivery output (alpha stage only -- beta/final are
   robustness simulations, not real deliverables).

**Stage parametrization** (2026-07-27 addition, docs/phase4_results.md
"beta 階段模擬"): which PVT triplet is "known" (the anchor set) vs. "to
predict" (the target set) may change across contest stages. `--stage`
selects a pre-defined `models.phase4_features` topology:

  - `alpha` (default, the current real contest input): anchors = the 5
    standard-voltage corners; targets = the 5 boost + 5 buck corners.
  - `beta`: anchors = the 5 boost corners; targets = the 5
    standard-voltage + 5 buck corners. The buck targets are two voltage
    steps from their nearest anchor (boost -> nominal -> buck), but the
    nearest-anchor rule (same process + same temperature) still resolves
    to the boost anchor, since no nominal-voltage anchor exists in this
    stage.
  - `final`: anchors = the 5 buck corners; targets = the 5 boost + 5
    standard-voltage corners.

Or pass `--anchors`/`--targets` explicitly (both required together) for
an ad-hoc topology outside the three predefined stages; the
nearest-anchor mapping is always inferred automatically
(`models.phase4_features.infer_nearest_anchor_by_target`) by matching
process+temperature, and fails loudly if that pairing isn't 1:1.

**The response-signature features (lever 1) and the nearest-anchor
lookup always resolve their roles from whichever topology is active for
this run** (`models.phase4_features._resolve_anchor_roles` /
`Phase4Dataset.nearest_anchor`), never from a hardcoded alpha-stage
corner name -- this is asserted by
`tests/test_phase4_features.py::test_beta_mode_nearest_anchor_and_sensitivity_features_use_beta_roles_not_alpha`.

**Cross-table features** (2026-07-29 addition, `--feature-mode
full_xtable`): for rise_power/fall_power rows, append the "same event"
delay-family arc's (cell_rise/rise_transition or cell_fall/fall_transition)
own anchor values + response-signature as extra columns
(`models.phase4_features.build_xtable_features`). Built ONCE from
`anchor_libs` (never a target-corner lib -- same zero-leakage guarantee
as the base feature matrix) and reused across all target corners in this
run, exactly like the base feature matrix. Delay-table (non-power) rows
are unaffected -- their extra columns are all zero.

**Seed ensemble** (2026-07-29 addition, `--seeds N`, default 1): fit N
independently-seeded models per target corner on the SAME already-built
feature matrix (feature construction happens once regardless of N -- only
the fit+predict cost multiplies), and average the N predicted log-ratios
before reconstructing to the linear scale.

Usage:
    python3 scripts/phase4_final_validate.py                              # alpha stage, default config, all 10 corners
    python3 scripts/phase4_final_validate.py --stage beta
    python3 scripts/phase4_final_validate.py --stage final --config gbdt_full
    python3 scripts/phase4_final_validate.py --anchors ss0p9v125c,ss0p9vm40c,ff1p1v125c,ff1p1vm40c,tt1p0v25c \\
        --targets ss0p81v125c,ss0p81vm40c,ff0p99v125c,ff0p99vm40c,tt0p9v25c,ss0p72v125c,ss0p72vm40c,ff0p88v125c,ff0p88vm40c,tt0p8v25c
    python3 scripts/phase4_final_validate.py --fold 2                     # 5-fold CV, fold 2's 80-cell split
    python3 scripts/phase4_final_validate.py --feature-mode full_xtable   # cross-table features on
    python3 scripts/phase4_final_validate.py --seeds 3                    # 3-seed MLP ensemble
    python3 scripts/phase4_final_validate.py --stage beta --corners ss0p72vm40c,tt1p0v25c  # smoke-test subset
    python3 scripts/phase4_final_validate.py --help
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from features.corners import parse_corner_filename
from liberty.parser import parse_file
from models.phase4_features import (
    PHASE4_CELL_SPLIT_SEED,
    PHASE4_DEV_SPLIT_SEED,
    SCORE_THRESHOLD,
    STAGE_TOPOLOGIES,
    CornerTopology,
    build_arc_attr_index,
    build_base_dataset,
    build_family_vocab_for_phase4,
    build_power_to_timing_arc_map,
    build_xtable_features,
    extract_raw_values,
    infer_nearest_anchor_by_target,
    make_label,
    reconstruct_predictions,
    score_breakdown,
    split_cells,
    split_dev,
    trainable_mask,
)
from models.phase4_final_config import CONFIGS, DEFAULT_CONFIG_TAG, fit_config
from paths import ALPHA_FULL_DIR, training_set_files

FEATURE_MODES = ("full", "full_xtable")
SEED_BASE = 20260729  # seed ensemble's per-member seeds: SEED_BASE + member_index

# Reference point printed alongside every run's result: the actual pooled
# score an official `--stage alpha` run of this script measured for
# `mlp_w256_b4_full` (docs/phase4_results.md). Fixed constant, not
# recomputed -- purely a comparison anchor so a beta/final simulation's
# number is easy to read relative to the real (alpha) contest result.
ALPHA_OFFICIAL_POOLED_SCORE = 96.2743


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


def main(
    config_tag: str,
    topology: CornerTopology,
    corners=None,
    *,
    fold: Optional[int] = None,
    feature_mode: str = "full",
    n_seeds: int = 1,
    dump_errors: Optional[str] = None,
) -> bool:
    """Returns True iff the pooled score clears SCORE_THRESHOLD (only
    meaningful when `corners` covers all of `topology.target_names`;
    otherwise this is a smoke-test subset run and the caller should not
    treat the return value as an official verdict).

    `fold`: None (default) uses the official 320/80 split; 0-4 selects a
    5-fold CV split instead (fold 4 == the default split).
    `feature_mode`: "full" (default) or "full_xtable" (adds the cross-
    table power<->delay features, see module docstring).
    `n_seeds`: 1 (default, current behavior) or more for a seed ensemble
    -- fits `n_seeds` independently-seeded models per corner on the same
    feature matrix and averages their predicted log-ratios.
    `dump_errors`: optional .npz path; saves the per-point validation
    arrays (corner, cell, table_type, y_true, y_pred, nearest_anchor)
    for every corner in this run, so downstream audits (e.g.
    scripts/phase4_alpha_audit.py's alpha-composition reweighting) can
    decompose the score without retraining. Scoring-neutral: the arrays
    are exactly what this run already computed for its own scoring.
    """
    t_start = time.time()
    assert feature_mode in FEATURE_MODES, f"unknown feature_mode {feature_mode!r}, expected one of {FEATURE_MODES}"
    assert n_seeds >= 1
    if corners is None:
        corners = list(topology.target_names)

    print(f"stage: {topology.name!r}")
    print(f"  anchors: {topology.anchor_names}")
    print(f"  targets: {topology.target_names}")
    print(f"  nearest_anchor_by_target: {topology.nearest_anchor_by_target}")
    print(f"config: {config_tag!r} -> {CONFIGS[config_tag]}")
    print("loading 15 training-set corner files (400 cells each)...")
    t0 = time.time()
    libs_by_name = _load_training_libs()
    print(f"  loaded {len(libs_by_name)} corners in {time.time() - t0:.1f}s")
    assert set(topology.anchor_names) <= set(libs_by_name)
    assert set(corners) <= set(topology.target_names)

    anchor_libs = {name: libs_by_name[name] for name in topology.anchor_names}
    target_libs = {name: libs_by_name[name] for name in corners}

    all_cells = sorted(anchor_libs[topology.anchor_names[0]].cells)
    assert len(all_cells) == 400, f"expected 400 training cells, found {len(all_cells)}"

    print("loading 1 alpha full-corner file (for the family-code vocabulary only)...")
    alpha_probe_path = sorted(ALPHA_FULL_DIR.glob("*.lib"))[0]
    alpha_probe_lib = parse_file(str(alpha_probe_path))
    alpha_cells = sorted(alpha_probe_lib.cells)
    assert len(alpha_cells) == 100
    assert not (set(all_cells) & set(alpha_cells)), "training/alpha cell overlap -- data contamination"

    if fold is None:
        train_cells, val_cells = split_cells(all_cells, seed=PHASE4_CELL_SPLIT_SEED)
    else:
        # 5-fold CV over the same permutation split_cells uses: fold f's
        # validation set is perm[80f:80(f+1)]. Fold 4 reproduces the
        # default split exactly (perm[320:400] == split_cells val set).
        names = sorted(all_cells)
        rng = np.random.default_rng(PHASE4_CELL_SPLIT_SEED)
        perm = rng.permutation(len(names))
        lo, hi = 80 * fold, 80 * (fold + 1)
        val_cells = sorted(names[i] for i in perm[lo:hi])
        train_cells = sorted(names[i] for i in np.concatenate([perm[:lo], perm[hi:]]))
    dev_train_cells, dev_val_cells = split_dev(train_cells, seed=PHASE4_DEV_SPLIT_SEED)
    fold_note = "default (== fold 4)" if fold is None else f"fold {fold}"
    print(f"cell split (seed={PHASE4_CELL_SPLIT_SEED}, {fold_note}): {len(train_cells)} train / "
          f"{len(val_cells)} validation (validation cells used ONLY for the final scoring below)")
    print(f"dev split (seed={PHASE4_DEV_SPLIT_SEED}) of the 320 train cells: "
          f"{len(dev_train_cells)} dev-train / {len(dev_val_cells)} dev-val (early stopping only)")
    # Zero-leakage runtime assertions (fail loudly rather than silently
    # scoring a contaminated split).
    assert not (set(dev_train_cells) & set(val_cells))
    assert not (set(dev_val_cells) & set(val_cells))
    assert set(dev_train_cells) | set(dev_val_cells) == set(train_cells)

    arc_attr_index = build_arc_attr_index(anchor_libs[topology.anchor_names[0]])
    family_vocab = build_family_vocab_for_phase4(all_cells, alpha_cells)

    print("\nbuilding TRAIN feature matrix (320 cells)...")
    t0 = time.time()
    ds_train = build_base_dataset(anchor_libs, train_cells, arc_attr_index, family_vocab,
                                   anchor_names=topology.anchor_names)
    print(f"  n={ds_train.n}  build_time={time.time() - t0:.1f}s  n_features={ds_train.X.shape[1]}")

    print("building VALIDATION feature matrix (80 held-out cells)...")
    t0 = time.time()
    ds_val = build_base_dataset(anchor_libs, val_cells, arc_attr_index, family_vocab,
                                 anchor_names=topology.anchor_names)
    print(f"  n={ds_val.n}  build_time={time.time() - t0:.1f}s")

    # Leakage guard: no validation cell may appear in the training matrix (and
    # vice versa) -- docs/plan.md's non-negotiable zero-leakage rule.
    assert not (set(ds_train.cell.tolist()) & set(val_cells)), "LEAKAGE: a validation cell appears in ds_train"
    assert not (set(ds_val.cell.tolist()) & set(train_cells)), "LEAKAGE: a train cell appears in ds_val"

    if feature_mode == "full_xtable":
        # Built ONCE from anchor_libs (structurally cannot contain a
        # target-corner lib -- see build_base_dataset's docstring) and
        # reused for every target corner below, exactly like ds_train.X
        # / ds_val.X themselves.
        print("\nbuilding cross-table (power<->delay) feature block...")
        t0 = time.time()
        power_to_timing_map, n_arc_matched, n_arc_unmatched = build_power_to_timing_arc_map(
            anchor_libs[topology.anchor_names[0]]
        )
        print(f"  arc map: {n_arc_matched} matched, {n_arc_unmatched} unmatched (fallback) "
              f"({time.time() - t0:.1f}s)")

        t0 = time.time()
        xtrain, xtable_names, n_matched_train, n_fallback_train = build_xtable_features(
            ds_train, anchor_libs, topology.anchor_names, power_to_timing_map
        )
        xval, _xtable_names2, n_matched_val, n_fallback_val = build_xtable_features(
            ds_val, anchor_libs, topology.anchor_names, power_to_timing_map
        )
        assert np.isfinite(xtrain).all() and np.isfinite(xval).all()
        ds_train.X = np.hstack([ds_train.X, xtrain])
        ds_val.X = np.hstack([ds_val.X, xval])
        print(f"  +{xtrain.shape[1]} xtable columns  train: matched={n_matched_train} "
              f"fallback={n_fallback_train}  val: matched={n_matched_val} fallback={n_fallback_val}  "
              f"({time.time() - t0:.1f}s)  n_features={ds_train.X.shape[1]}")

    is_dev_train_cellmask = np.isin(ds_train.cell, np.asarray(dev_train_cells))
    is_dev_val_cellmask = np.isin(ds_train.cell, np.asarray(dev_val_cells))

    per_corner = {}
    all_errs = []
    dump_chunks = []  # (corner, cell, table_type, y_true, y_pred, nearest_anchor) per corner

    from scoring.scorer import point_errors, score_from_errors

    for target_name in corners:
        print(f"\n=== delivery corner: {target_name} (nearest anchor: "
              f"{topology.nearest_anchor_by_target[target_name]}) ===")
        target_lib = target_libs[target_name]

        y_train_true = extract_raw_values(target_lib, ds_train.keys)
        nearest_train = ds_train.nearest_anchor(target_name, topology.nearest_anchor_by_target)
        label_train = make_label(nearest_train, y_train_true)
        train_ok = trainable_mask(nearest_train)

        is_dev_train = train_ok & is_dev_train_cellmask
        is_dev_val = train_ok & is_dev_val_cellmask
        print(f"  dev-train rows={int(is_dev_train.sum())}  dev-val rows={int(is_dev_val.sum())}")

        # Seed ensemble (--seeds N > 1): fit N independently-seeded models
        # on the SAME feature matrix slices built above (feature
        # construction happens once regardless of N), average their
        # predicted log-ratios in log space before reconstructing once.
        y_pred_ratio_sum = None
        total_train_seconds = 0.0
        fit_infos = []
        for member in range(n_seeds):
            member_seed = SEED_BASE + member if n_seeds > 1 else None
            handle = fit_config(
                config_tag,
                ds_train.X[is_dev_train], label_train[is_dev_train],
                ds_train.X[is_dev_val], label_train[is_dev_val],
                seed=member_seed,
            )
            total_train_seconds += handle.train_seconds
            fit_infos.append(handle.info)
            member_pred_ratio = handle.predict(ds_val.X)
            y_pred_ratio_sum = member_pred_ratio if y_pred_ratio_sum is None else y_pred_ratio_sum + member_pred_ratio
        y_pred_ratio = y_pred_ratio_sum / n_seeds
        seed_note = f" [{n_seeds}-seed ensemble]" if n_seeds > 1 else ""
        print(f"  fit: train_time={total_train_seconds:.1f}s{seed_note} " + "; ".join(fit_infos))

        y_val_true = extract_raw_values(target_lib, ds_val.keys)
        nearest_val = ds_val.nearest_anchor(target_name, topology.nearest_anchor_by_target)
        groups_val = {"table_type": ds_val.table_type}

        pred = reconstruct_predictions(nearest_val, y_pred_ratio)
        overall, n, breakdown = score_breakdown(y_val_true, pred, groups_val)
        _print_breakdown(config_tag, overall, n, breakdown)

        all_errs.append(point_errors(y_val_true, pred))
        per_corner[target_name] = (overall, n, breakdown, "; ".join(fit_infos), total_train_seconds)

        if dump_errors is not None:
            dump_chunks.append((
                np.full(len(y_val_true), target_name),
                ds_val.cell.copy(),
                ds_val.table_type.copy(),
                y_val_true.astype(np.float64),
                pred.astype(np.float64),
                nearest_val.astype(np.float64),
            ))

    pooled = score_from_errors(np.concatenate(all_errs))

    if dump_errors is not None:
        dump_path = Path(dump_errors)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dump_path,
            corner=np.concatenate([c[0] for c in dump_chunks]),
            cell=np.concatenate([c[1] for c in dump_chunks]),
            table_type=np.concatenate([c[2] for c in dump_chunks]),
            y_true=np.concatenate([c[3] for c in dump_chunks]),
            y_pred=np.concatenate([c[4] for c in dump_chunks]),
            nearest_anchor=np.concatenate([c[5] for c in dump_chunks]),
            meta_config=np.array([config_tag]),
            meta_stage=np.array([topology.name]),
            meta_fold=np.array(["default" if fold is None else str(fold)]),
            meta_n_seeds=np.array([str(n_seeds)]),
        )
        print(f"\n  dumped per-point validation arrays -> {dump_path}")

    print("\n=== summary: per-corner overall scores ===")
    print(f"  {'corner':16s} {'score':>10s}")
    for name in corners:
        print(f"  {name:16s} {per_corner[name][0]:10.4f}")

    print(f"\n=== summary: per-corner x per-table_type scores ({config_tag}) ===")
    for name in corners:
        bd = per_corner[name][2]["table_type"]
        print(f"  {name}: " + ", ".join(f"{tt}={s:.2f}" for tt, (s, _n) in sorted(bd.items())))

    print(f"\n=== pooled overall across {len(corners)} delivery corner(s), stage={topology.name!r} "
          f"(80 held-out cells) ===")
    print(f"  pooled: {pooled:.4f}")
    print(f"  reference (alpha-stage official mlp_w256_b4_full result): {ALPHA_OFFICIAL_POOLED_SCORE:.4f}  "
          f"(delta: {pooled - ALPHA_OFFICIAL_POOLED_SCORE:+.4f})")
    print(f"  SCORE_THRESHOLD = {SCORE_THRESHOLD}")
    is_full_run = len(corners) == len(topology.target_names)
    passed = pooled >= SCORE_THRESHOLD and is_full_run
    if not is_full_run:
        print(f"  RESULT: N/A -- only {len(corners)}/{len(topology.target_names)} target corners were run "
              f"(smoke-test subset via --corners); re-run with the full target set for an official PASS/FAIL verdict.")
    elif passed:
        print(f"  RESULT: PASS -- {config_tag} clears the {SCORE_THRESHOLD} gate for stage {topology.name!r}"
              + ("; scripts/phase4_final_predict.py may proceed." if topology.name == "alpha"
                 else " (note: beta/final are robustness simulations -- delivery still runs on alpha only)."))
    else:
        print(f"  RESULT: FAIL -- pooled score {pooled:.4f} < {SCORE_THRESHOLD} for stage {topology.name!r}; "
              f"no delivery output should be produced from this model/stage.")

    print(f"\ntotal wall time: {time.time() - t_start:.1f}s")
    return passed


def _parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default=DEFAULT_CONFIG_TAG, choices=sorted(CONFIGS),
                     help=f"production config tag (models.phase4_final_config.CONFIGS); default {DEFAULT_CONFIG_TAG!r}")
    ap.add_argument("--stage", default="alpha", choices=sorted(STAGE_TOPOLOGIES),
                     help="which pre-defined corner topology to use (default 'alpha', the real contest input). "
                          "Ignored if --anchors/--targets are given.")
    ap.add_argument("--anchors", default=None,
                     help="comma-separated list of 5 anchor corner names, overriding --stage. "
                          "Must be given together with --targets.")
    ap.add_argument("--targets", default=None,
                     help="comma-separated list of 10 target corner names, overriding --stage. "
                          "Must be given together with --anchors.")
    ap.add_argument("--fold", type=int, default=None, choices=[0, 1, 2, 3, 4],
                    help="5-fold CV fold index over the fixed permutation; "
                         "fold 4 == the default split. Omit for default behavior.")
    ap.add_argument("--feature-mode", default="full", choices=FEATURE_MODES,
                     help="'full' (default) or 'full_xtable' (adds cross-table power<->delay features).")
    ap.add_argument("--seeds", type=int, default=1,
                     help="number of independently-seeded models to fit per corner and average "
                          "(seed ensemble); default 1 (current behavior, no ensembling).")
    ap.add_argument("--corners", default="all",
                     help="comma-separated SUBSET of the active topology's target corners to run, or 'all' "
                          "(default) for the full target set. A subset is for smoke-testing only -- "
                          "PASS/FAIL is only reported for a full run.")
    ap.add_argument("--dump-errors", default=None, metavar="PATH",
                     help="optional .npz path to save per-point validation arrays "
                          "(corner/cell/table_type/y_true/y_pred/nearest_anchor) for downstream "
                          "audits, e.g. scripts/phase4_alpha_audit.py.")
    return ap.parse_args()


def _resolve_topology(args) -> CornerTopology:
    if args.anchors or args.targets:
        assert args.anchors and args.targets, "--anchors and --targets must be given together"
        anchors = tuple(a.strip() for a in args.anchors.split(","))
        targets = tuple(t.strip() for t in args.targets.split(","))
        return CornerTopology(
            "custom", anchors, targets, infer_nearest_anchor_by_target(anchors, targets)
        )
    return STAGE_TOPOLOGIES[args.stage]


if __name__ == "__main__":
    args = _parse_args()
    topology = _resolve_topology(args)
    corners = list(topology.target_names) if args.corners == "all" else [c.strip() for c in args.corners.split(",")]
    ok = main(args.config, topology, corners, fold=args.fold, feature_mode=args.feature_mode,
              n_seeds=args.seeds, dump_errors=args.dump_errors)
    sys.exit(0 if ok or args.corners != "all" else 1)
