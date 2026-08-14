"""Phase 4b improvement-round screening harness (docs/phase4_results.md
"改進實驗" round, 2026-07-27).

**Dev-only tuning tool** -- this script NEVER builds a feature matrix for
the 80 held-out validation cells and never imports/calls anything that
would touch them. It only uses the internal 256-dev-train / 64-dev-val
split carved out of the 320 training cells
(`models.phase4_features.split_dev`, same seed as the official
`scripts/phase4_validate.py`), per docs/plan.md's rule: "80 驗證 cell
只能用於最終評分，絕不能參與任何調參決策（調參用 320 內部 dev）". Once a
config wins here, re-measure it with the *official* protocol
(`scripts/phase4_validate.py`, which does score the 80 held-out cells)
to get the number that actually counts.

What this covers (docs/plan.md Phase 4 improvement-round levers, in
priority order):

1. Response-signature features on/off (`--feature-mode full|no_lever12`)
   -- ablates `SENSITIVITY_FEATURE_NAMES` + `GRADIENT_FEATURE_NAMES`
   (models.phase4_features) by dropping those columns, so the *same*
   underlying dataset build is reused for both arms (cheap: no need to
   rebuild the feature matrix twice).
2. MLP capacity variants (width x n_blocks) -- models.phase4_mlp's
   residual architecture, several sizes registered in CONFIGS below.
3. Label definition (`--label-mode ratio|raw`) -- `ratio` is the
   existing log(|target|/|nearest_anchor|) label; `raw` is
   log(|target|) directly (docs/plan.md improvement-round lever 5).
   Both GBDT and MLP use whichever mode a given config specifies, so a
   fair same-definition comparison is just two configs with matching
   `label_mode`.
4. Ensemble (`--ensemble tagA,tagB`) -- average two *already-run*
   ratio-mode configs' predicted log-ratios (geometric mean on the
   linear scale) and score the blend. Only meaningful for two
   ratio-mode configs against the same target corner in the same run
   (predictions are cached in-memory per (corner, tag) for this).

Usage (chunk your own runs -- each target-corner x config pair can take
anywhere from ~10s (GBDT) to ~200s+ (a wide/deep MLP on full dev-train),
so keep a single invocation to a handful of (corner, config) pairs if
you want to stay under ~10 minutes):

    python3 scripts/phase4b_screen.py --corners ss0p72vm40c --configs gbdt_full,mlp_w192_b3_full
    python3 scripts/phase4b_screen.py --corners all --configs all          # full sweep, slow
    python3 scripts/phase4b_screen.py --list                               # print available corners/configs and exit

Every run appends its result rows to `docs/phase4b_screen_log.md`
(created with a header if missing) in addition to printing them, so
partial/chunked runs accumulate into one record.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from features.corners import parse_corner_filename
from liberty.parser import parse_file
from models.phase4_features import (
    ANCHOR_CORNER_NAMES,
    BETA_TOPOLOGY,
    FINAL_TOPOLOGY,
    DELIVERY_CORNER_NAMES,
    EPS,
    FEATURE_NAMES,
    GRADIENT_FEATURE_NAMES,
    PHASE4_CELL_SPLIT_SEED,
    PHASE4_DEV_SPLIT_SEED,
    SENSITIVITY_FEATURE_NAMES,
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
from models.phase4_mlp import HUBER_DELTA, fit_mlp, predict_mlp
from paths import ALPHA_FULL_DIR, training_set_files

CACHE_DIR = REPO_ROOT / "output" / "_phase4_cache"
LIBS_CACHE = CACHE_DIR / "phase4b_libs_cache.pkl"
LOG_MD = REPO_ROOT / "docs" / "phase4b_screen_log.md"

# ---------------------------------------------------------------------------
# Feature-ablation column index (lever 1+2 on/off)
# ---------------------------------------------------------------------------
_ABLATE_NAMES = set(SENSITIVITY_FEATURE_NAMES) | set(GRADIENT_FEATURE_NAMES)
LEVER12_COLS = np.array([i for i, name in enumerate(FEATURE_NAMES) if name in _ABLATE_NAMES])
assert LEVER12_COLS.size == len(_ABLATE_NAMES)


def drop_lever12_columns(X: np.ndarray) -> np.ndarray:
    return np.delete(X, LEVER12_COLS, axis=1)


# ---------------------------------------------------------------------------
# Label-mode helpers (lever 5). "ratio" mode is exactly
# models.phase4_features.make_label/reconstruct_predictions. "raw" mode
# predicts log(|target|) directly (no anchor-relative ratio); the
# zero-anchor-forces-zero rule (docs/plan.md rule 3) still applies at
# reconstruction since a zero *anchor* row is always a known-invalid
# power arc regardless of which label definition trained the model.
# ---------------------------------------------------------------------------


def make_label_raw(target: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(np.abs(target) + EPS)


def reconstruct_raw(nearest_anchor: np.ndarray, y_pred: np.ndarray, *, clip: float = 20.0) -> np.ndarray:
    y_pred = np.clip(np.asarray(y_pred, dtype=float), -clip, clip)
    pred = np.exp(y_pred)
    return np.where(nearest_anchor == 0.0, 0.0, pred)


# ---------------------------------------------------------------------------
# "signed_ratio" label mode -- fall_power diagnosis fix (2026-07-27
# coordinator request). Root cause: fall_power (unlike rise_power, which
# is non-negative in every corner checked) genuinely flips sign between
# the nearest anchor and several target corners for a small (~0.2-0.9%
# of both-nonzero points) but consequential subset of grid points --
# concentrated exactly where |value| is near zero. The existing
# log(|target|/|anchor|) label only ever regresses *magnitude*, and
# `reconstruct_predictions` always re-applies `sign(nearest_anchor)` --
# so a sign-flip point is a guaranteed ~100%-relative-error point no
# matter how good the model is. At that frequency, RMS-of-squared
# capped-errors is enough on its own to explain most of the observed
# fall_power-vs-rise_power score gap (verified: anchor==0-but-target!=0
# never occurs -- rule 3 is intact and is NOT the cause; see
# docs/phase4b_screen_log.md diagnosis notes).
#
# Fix: let the label represent the *signed* ratio `r = target/anchor`
# (r < 0 exactly when the sign flips) via `y = arcsinh(r - 1)`:
#   - centered at 0 for the common "no change" case (r=1 -> y=0),
#     unlike a raw signed ratio which would center at 1;
#   - smooth and finite for every real r, including r <= 0 (sign flips
#     and near-zero targets) where log(r) would be undefined -- this is
#     exactly the case the plain ratio-log label cannot represent;
#   - asymptotically log-like for large |r-1|, so it degrades gracefully
#     back to ~the same behavior as the existing label for the
#     well-behaved (sign-stable, large-ratio) majority of points.
# Inverse: r = sinh(y) + 1, pred = anchor * r (zero-anchor still forces
# an exact 0 prediction, same rule-3 convention as every other mode).
# ---------------------------------------------------------------------------


def make_label_signed_ratio(nearest_anchor: np.ndarray, target: np.ndarray) -> np.ndarray:
    anchor_safe = np.where(nearest_anchor == 0.0, 1.0, nearest_anchor)  # dummy; excluded via trainable_mask
    r = target / anchor_safe
    return np.arcsinh(r - 1.0)


def reconstruct_signed_ratio(nearest_anchor: np.ndarray, y_pred: np.ndarray, *, clip: float = 20.0) -> np.ndarray:
    y_pred = np.clip(np.asarray(y_pred, dtype=float), -clip, clip)
    r = np.sinh(y_pred) + 1.0
    pred = nearest_anchor * r
    return np.where(nearest_anchor == 0.0, 0.0, pred)


# ---------------------------------------------------------------------------
# Sign-flip propensity features (coordinator follow-up, 2026-07-27) --
# ANCHOR_CORNER_NAMES's fixed sorted order is (ff0p99v125c, ff0p99vm40c,
# ss0p81v125c, ss0p81vm40c, tt0p9v25c) == (ff_hot, ff_cold, ss_hot,
# ss_cold, tt_mid), same indices models.phase4_features uses internally
# for SENSITIVITY_FEATURE_NAMES. All derived purely from the 5 anchor
# tables (corner-agnostic, same dataset reused for every target model,
# same rationale as models.phase4_features.build_base_dataset).
# ---------------------------------------------------------------------------
_FF_HOT, _FF_COLD, _SS_HOT, _SS_COLD, _TT_MID = range(5)

SIGNFLIP_FEATURE_NAMES = [
    "sign_ff_hot", "sign_ff_cold", "sign_ss_hot", "sign_ss_cold", "sign_tt_mid",
    "log_min_abs_anchor",       # how close does ANY anchor get to zero at this grid point
    "sign_agree_ff_hot_cold",   # ff temperature-sign stability
    "sign_agree_ss_hot_cold",   # ss temperature-sign stability
    "sign_agree_ss_ff_hot",     # process-sign agreement at 125C
    "sign_agree_ss_ff_cold",    # process-sign agreement at -40C
    "n_anchors_negative",       # 0-5, overall sign spread across the 5 anchors
    "neighbor_sign_disagree_ff_hot", "neighbor_sign_disagree_ff_cold",
    "neighbor_sign_disagree_ss_hot", "neighbor_sign_disagree_ss_cold",
    "neighbor_sign_disagree_tt_mid",
]


def build_signflip_features(ds) -> np.ndarray:
    """One row per ds row (49-row blocks per key, same layout as ds.X).
    Returns an (n, len(SIGNFLIP_FEATURE_NAMES)) float32 array to hstack
    onto ds.X for the 'full_plus_signflip' feature_mode."""
    anchor_vals = ds.anchor_values  # (n, 5) raw linear values
    sign = np.sign(anchor_vals)  # {-1, 0, 1}
    n = ds.n

    log_min_abs = np.log(np.min(np.abs(anchor_vals), axis=1) + EPS).astype(np.float32).reshape(-1, 1)
    sign_agree_ff = (sign[:, _FF_HOT] == sign[:, _FF_COLD]).astype(np.float32).reshape(-1, 1)
    sign_agree_ss = (sign[:, _SS_HOT] == sign[:, _SS_COLD]).astype(np.float32).reshape(-1, 1)
    sign_agree_ss_ff_hot = (sign[:, _SS_HOT] == sign[:, _FF_HOT]).astype(np.float32).reshape(-1, 1)
    sign_agree_ss_ff_cold = (sign[:, _SS_COLD] == sign[:, _FF_COLD]).astype(np.float32).reshape(-1, 1)
    n_negative = np.sum(sign < 0, axis=1, keepdims=True).astype(np.float32)

    # Neighbor sign-disagreement fraction per anchor, within each key's
    # 7x7 grid block (up/down/left/right, edge-clamped -- only existing
    # neighbors counted in the denominator).
    neighbor_frac = np.zeros((n, 5), dtype=np.float32)
    i = 0
    while i < n:
        j = i + 49
        block_sign = sign[i:j].reshape(7, 7, 5)
        for a in range(5):
            g = block_sign[:, :, a]
            disagree = np.zeros((7, 7), dtype=np.float32)
            count = np.zeros((7, 7), dtype=np.float32)
            disagree[1:, :] += (g[1:, :] != g[:-1, :]).astype(np.float32)
            count[1:, :] += 1
            disagree[:-1, :] += (g[:-1, :] != g[1:, :]).astype(np.float32)
            count[:-1, :] += 1
            disagree[:, 1:] += (g[:, 1:] != g[:, :-1]).astype(np.float32)
            count[:, 1:] += 1
            disagree[:, :-1] += (g[:, :-1] != g[:, 1:]).astype(np.float32)
            count[:, :-1] += 1
            neighbor_frac[i:j, a] = (disagree / np.maximum(count, 1)).ravel()
        i = j

    return np.hstack([
        sign.astype(np.float32),
        log_min_abs,
        sign_agree_ff, sign_agree_ss, sign_agree_ss_ff_hot, sign_agree_ss_ff_cold,
        n_negative,
        neighbor_frac,
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# Surface features (2026-08-10). A sign flip is a *contour* phenomenon --
# the zero-crossing curve of a 2D surface moving as the corner changes --
# and a per-point regressor structurally cannot represent a contour. The
# base feature set only carries this point's own anchor values plus
# hand-made local gradients.
#
# Deliberately NOT another hand-crafted "where I think the contour is"
# feature: `build_signflip_features` above already tried that and it COST
# 1.06 points (gbdt_signflip_feats_only 94.05 vs gbdt_signed_ratio 95.11,
# both signed_ratio label). Instead this hands over the raw profiles --
# the full 7-value row and column of the nearest-anchor surface passing
# through this point -- and lets the model extract whatever it wants.
#
# Unlike the sign-flip block this is target-corner DEPENDENT (it reads the
# nearest anchor), so it must be rebuilt per corner rather than once.
# ---------------------------------------------------------------------------

SURFACE_FEATURE_NAMES = (
    [f"surf_row_logabs_{k}" for k in range(7)]
    + [f"surf_row_sign_{k}" for k in range(7)]
    + [f"surf_col_logabs_{k}" for k in range(7)]
    + [f"surf_col_sign_{k}" for k in range(7)]
)


def build_surface_features(nearest: np.ndarray) -> np.ndarray:
    """(n, 28) from the per-row nearest-anchor values, which arrive in the
    same 49-row (7x7, row-major over index_1 x index_2) blocks as ds.X."""
    n = nearest.shape[0]
    assert n % 49 == 0, f"expected whole 7x7 blocks, got n={n}"
    blocks = nearest.reshape(-1, 7, 7)
    logabs = np.log(np.abs(blocks) + EPS).astype(np.float32)
    sign = np.sign(blocks).astype(np.float32)
    n_blocks = blocks.shape[0]

    # For point (i, j): its whole row i and whole column j.
    out = np.empty((n_blocks, 7, 7, 28), dtype=np.float32)
    for i in range(7):
        for j in range(7):
            out[:, i, j, 0:7] = logabs[:, i, :]
            out[:, i, j, 7:14] = sign[:, i, :]
            out[:, i, j, 14:21] = logabs[:, :, j]
            out[:, i, j, 21:28] = sign[:, :, j]
    return out.reshape(n, 28)


# ---------------------------------------------------------------------------
# Sample weighting (coordinator follow-up, 2026-07-27) -- fall_power's
# near-zero grid points carry most of the fall_power score gap
# (docs/phase4b_screen_log.md diagnosis: +4.6 pts theoretical from fixing
# them vs +2.5 for sign-flips alone). Weight is looked up by *anchor*
# magnitude (never the label/target) so it is a legitimate input-side
# choice, not a leak.
# ---------------------------------------------------------------------------


def compute_sample_weights(
    table_type: np.ndarray,
    nearest_anchor: np.ndarray,
    *,
    weight: float = 5.0,
    threshold: float = 3e-4,
    target_table_type: str = "fall_power",
) -> Optional[np.ndarray]:
    if weight == 1.0:
        return None
    w = np.ones(table_type.shape[0], dtype=np.float64)
    mask = (table_type == target_table_type) & (np.abs(nearest_anchor) < threshold)
    w[mask] = weight
    return w


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------
# Each config: tag -> dict(model="gbdt"|"mlp", feature_mode="full"|"no_lever12",
# label_mode="ratio"|"raw", **model_kwargs passed straight to fit_gbdt/fit_mlp).
CONFIGS: Dict[str, dict] = {
    "gbdt_full": dict(model="gbdt", feature_mode="full", label_mode="ratio"),
    "gbdt_no_lever12": dict(model="gbdt", feature_mode="no_lever12", label_mode="ratio"),
    "gbdt_full_raw_label": dict(model="gbdt", feature_mode="full", label_mode="raw"),
    "mlp_w128_b2_full": dict(model="mlp", feature_mode="full", label_mode="ratio",
                              width=128, n_blocks=2, max_epochs=100, patience=12),
    "mlp_w192_b3_full": dict(model="mlp", feature_mode="full", label_mode="ratio",
                              width=192, n_blocks=3, max_epochs=100, patience=12),
    "mlp_w256_b4_full": dict(model="mlp", feature_mode="full", label_mode="ratio",
                              width=256, n_blocks=4, max_epochs=150, patience=15),
    "mlp_w384_b5_full": dict(model="mlp", feature_mode="full", label_mode="ratio",
                              width=384, n_blocks=5, max_epochs=150, patience=15),
    "mlp_w192_b3_no_lever12": dict(model="mlp", feature_mode="no_lever12", label_mode="ratio",
                                    width=192, n_blocks=3, max_epochs=100, patience=12),
    "mlp_w192_b3_raw_label": dict(model="mlp", feature_mode="full", label_mode="raw",
                                   width=192, n_blocks=3, max_epochs=100, patience=12),
    # fall_power sign-flip fix (see make_label_signed_ratio's docstring above).
    "gbdt_signed_ratio": dict(model="gbdt", feature_mode="full", label_mode="signed_ratio"),
    "mlp_w128_b2_signed_ratio": dict(model="mlp", feature_mode="full", label_mode="signed_ratio",
                                      width=128, n_blocks=2, max_epochs=60, patience=8),
    # fall_power near-zero sample weighting + sign-flip features (coordinator
    # follow-up round). `sample_weight`/`sample_weight_threshold` are popped
    # off before reaching fit_gbdt/fit_mlp's model kwargs -- see main loop.
    "gbdt_fp_weighted": dict(model="gbdt", feature_mode="full_plus_signflip", label_mode="signed_ratio",
                              sample_weight=5.0, sample_weight_threshold=3e-4),
    # isolation arm: sign-flip features + signed_ratio label WITHOUT sample
    # weighting, to separate "do the features help" from "does weighting hurt".
    "gbdt_signflip_feats_only": dict(model="gbdt", feature_mode="full_plus_signflip", label_mode="signed_ratio",
                                      sample_weight=1.0),
    "gbdt_fp_weighted_w10": dict(model="gbdt", feature_mode="full_plus_signflip", label_mode="signed_ratio",
                                  sample_weight=10.0, sample_weight_threshold=3e-4),
    "mlp_w256_b4_fp_weighted_signed": dict(model="mlp", feature_mode="full_plus_signflip", label_mode="signed_ratio",
                                            width=256, n_blocks=4, max_epochs=150, patience=15,
                                            sample_weight=5.0, sample_weight_threshold=3e-4),
    # 2026-08-10 robust / score-aligned training loss (models.phase4_mlp
    # `_elementwise_loss`). Mirror image of the harmful (-3.57) 5x near-zero
    # up-weighting arm above: the ~0.085% sign-flip points score 0 whatever
    # we predict, yet their log-ratio labels (std 1.90) dominate an unbounded
    # MSE gradient. Identical architecture/features/label to the production
    # `mlp_w256_b4_full` arm -- the loss is the only difference, so the
    # comparison isolates it.
    "mlp_w256_b4_huber": dict(model="mlp", feature_mode="full", label_mode="ratio",
                               width=256, n_blocks=4, max_epochs=150, patience=15,
                               loss="huber"),
    "mlp_w256_b4_huber_d035": dict(model="mlp", feature_mode="full", label_mode="ratio",
                                    width=256, n_blocks=4, max_epochs=150, patience=15,
                                    loss="huber", huber_delta=0.35),
    "mlp_w256_b4_scoreloss": dict(model="mlp", feature_mode="full", label_mode="ratio",
                                   width=256, n_blocks=4, max_epochs=150, patience=15,
                                   loss="score"),
    # 2026-08-10 recommendation 2: one model per table type (see the
    # per_table_type branch in main()). NOTE both arms below hit the
    # max_epochs=150 cap on the four delay tables (each model sees only
    # ~1/6 of the rows, so an epoch is ~1/6 the training) -- their logged
    # scores are under-trained and do NOT refute per-table modelling.
    # Raise max_epochs well above 150 before reading anything into them.
    "mlp_w256_b4_per_table": dict(model="mlp", feature_mode="full", label_mode="ratio",
                                   width=256, n_blocks=4, max_epochs=150, patience=15,
                                   per_table_type=True),
    "mlp_w256_b4_per_table_huber": dict(model="mlp", feature_mode="full", label_mode="ratio",
                                         width=256, n_blocks=4, max_epochs=150, patience=15,
                                         per_table_type=True, loss="huber"),
    # 2026-08-10 follow-up: table-type-dependent clip strength inside ONE
    # model (delay tables aggressive, power tables mild), which is what the
    # per-table screen's monotone split actually calls for.
    "mlp_w256_b4_huber_ptd": dict(model="mlp", feature_mode="full", label_mode="ratio",
                                   width=256, n_blocks=4, max_epochs=150, patience=15,
                                   loss="huber", per_table_delta=(0.35, HUBER_DELTA)),
    "mlp_w256_b4_huber_ptd20": dict(model="mlp", feature_mode="full", label_mode="ratio",
                                     width=256, n_blocks=4, max_epochs=150, patience=15,
                                     loss="huber", per_table_delta=(0.20, HUBER_DELTA)),
    # 2026-08-10 surface bet: same loss as the winning `mlp_w256_b4_huber`
    # arm, so the ONLY difference is the 28 raw row/column profile features
    # (build_surface_features). Compare against mlp_w256_b4_huber, not
    # against mlp_w256_b4_full.
    "mlp_w256_b4_surface": dict(model="mlp", feature_mode="full_plus_surface", label_mode="ratio",
                                 width=256, n_blocks=4, max_epochs=150, patience=15,
                                 loss="huber"),
    # 2026-08-11 multi-topology augmentation: the surface arm failed because
    # 320 cells cannot support extra feature DIMENSIONS, so add training ROWS
    # instead. For each alpha target, the same 400 training cells also supply
    # a "predict this target from a different anchor set" view -- boost
    # anchors for a buck target, buck anchors for a boost target (whichever
    # set does not contain the target itself). Doubles dev-train rows.
    # dev-val stays alpha-topology-only: that is the deployment configuration.
    "mlp_w256_b4_huber_aug": dict(model="mlp", feature_mode="full", label_mode="ratio",
                                   width=256, n_blocks=4, max_epochs=150, patience=15,
                                   loss="huber", augment=True),
}


# Gap-matched augmentation. The first attempt (2026-08-11) reused the SAME
# target under another topology and lost 2.16 points: for alpha target
# ss0p72vm40c the deployment mapping is ss0p81vm40c -> ss0p72vm40c (ONE
# voltage step down), but the beta-topology row for that same target is
# anchored at ss0p9vm40c (TWO steps down). Different label distribution,
# no feature telling the model which, so it averaged two incompatible
# mappings. Fix: keep the *gap* fixed and move the target instead --
# the standard-voltage corner of the same process+temperature is always
# exactly one step from the other topology's anchor, in the same direction
# as the deployment mapping.
_STANDARD_BY_PROCESS_TEMP = {
    ("ss", "v125c"): "ss0p81v125c", ("ss", "vm40c"): "ss0p81vm40c",
    ("ff", "v125c"): "ff0p99v125c", ("ff", "vm40c"): "ff0p99vm40c",
    ("tt", "v25c"): "tt0p9v25c",
}


def augment_topology_for(target_name: str):
    """(topology, augmenting target) whose anchor->target voltage gap
    matches `target_name`'s alpha-stage gap in size and direction, or
    (None, None).

    A buck alpha target (standard -> buck, one step down) is matched by
    the beta topology's boost -> standard mapping; a boost alpha target
    (standard -> boost, one step up) by the final topology's buck ->
    standard mapping. Either way the augmenting target is the standard
    corner at the same process and temperature."""
    proc = target_name[:2]
    temp = "v125c" if target_name.endswith("v125c") else ("vm40c" if target_name.endswith("vm40c") else "v25c")
    aug_target = _STANDARD_BY_PROCESS_TEMP.get((proc, temp))
    if aug_target is None:
        return None, None
    for topo in (BETA_TOPOLOGY, FINAL_TOPOLOGY):
        if aug_target in topo.target_names and target_name in topo.anchor_names:
            # `target_name` is an anchor of this topology == the topology
            # steps from the opposite extreme through the standard corner,
            # i.e. the same direction as the deployment mapping.
            continue
        if aug_target in topo.target_names and target_name not in topo.anchor_names:
            return topo, aug_target
    return None, None
ENSEMBLE_PAIRS: List[Tuple[str, str]] = [
    ("gbdt_full", "mlp_w192_b3_full"),
]


def _load_training_libs():
    libs_by_name = {}
    for path in training_set_files():
        meta = parse_corner_filename(str(path))
        libs_by_name[meta.name] = parse_file(str(path))
    return libs_by_name


def _load_or_build_libs_cache(refresh: bool):
    if not refresh and LIBS_CACHE.exists():
        print(f"loading cached parsed libs from {LIBS_CACHE} ...")
        t0 = time.time()
        with open(LIBS_CACHE, "rb") as f:
            data = pickle.load(f)
        print(f"  loaded in {time.time()-t0:.1f}s")
        return data["libs_by_name"], data["alpha_probe_lib"]

    print("parsing 15 training-set corner files (400 cells each; ~35s)...")
    t0 = time.time()
    libs_by_name = _load_training_libs()
    alpha_probe_path = sorted(ALPHA_FULL_DIR.glob("*.lib"))[0]
    alpha_probe_lib = parse_file(str(alpha_probe_path))
    print(f"  parsed in {time.time()-t0:.1f}s")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LIBS_CACHE, "wb") as f:
        pickle.dump({"libs_by_name": libs_by_name, "alpha_probe_lib": alpha_probe_lib}, f, protocol=4)
    print(f"  cached to {LIBS_CACHE} for future invocations")
    return libs_by_name, alpha_probe_lib


def _print_and_log(lines: List[str]) -> None:
    text = "\n".join(lines)
    print(text)
    LOG_MD.parent.mkdir(parents=True, exist_ok=True)
    is_new = not LOG_MD.exists()
    with open(LOG_MD, "a") as f:
        if is_new:
            f.write(
                "# Phase 4b screening log (dev-only; 256/64 internal split of the 320 train "
                "cells -- NEVER the 80 held-out validation cells)\n\n"
                "Auto-appended by `scripts/phase4b_screen.py`. Each row is one "
                "(target_corner, config) run scored on the internal dev-val subset.\n\n"
                "| corner | config | overall | cell_rise | cell_fall | rise_transition | "
                "fall_transition | rise_power | fall_power | train_s | notes |\n"
                "|---|---|---|---|---|---|---|---|---|---|---|\n"
            )
        f.write(text + "\n")


def _row_md(corner: str, tag: str, overall: float, bd: dict, train_s, notes: str) -> str:
    def g(tt):
        return f"{bd.get(tt, (float('nan'), 0))[0]:.2f}" if tt in bd else "-"

    ts = f"{train_s:.1f}" if train_s is not None else "-"
    return (f"| {corner} | {tag} | {overall:.4f} | {g('cell_rise')} | {g('cell_fall')} | "
            f"{g('rise_transition')} | {g('fall_transition')} | {g('rise_power')} | "
            f"{g('fall_power')} | {ts} | {notes} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corners", default="ss0p72vm40c",
                     help="comma-separated delivery corner names, or 'all' for all 10")
    ap.add_argument("--configs", default="gbdt_full,mlp_w192_b3_full",
                     help="comma-separated config tags (see CONFIGS), or 'all'")
    ap.add_argument("--ensemble", default="",
                     help="comma-separated tagA,tagB to also ensemble (both must be in --configs and ratio-mode)")
    ap.add_argument("--refresh-cache", action="store_true", help="re-parse libs even if a cache pickle exists")
    ap.add_argument("--list", action="store_true", help="print available corners/configs and exit")
    args = ap.parse_args()

    if args.list:
        print("delivery corners:", ", ".join(DELIVERY_CORNER_NAMES))
        print("configs:", ", ".join(CONFIGS))
        return

    corners = list(DELIVERY_CORNER_NAMES) if args.corners == "all" else [c.strip() for c in args.corners.split(",")]
    tags = list(CONFIGS) if args.configs == "all" else [c.strip() for c in args.configs.split(",")]
    for c in corners:
        assert c in DELIVERY_CORNER_NAMES, f"unknown corner {c!r}"
    for t in tags:
        assert t in CONFIGS, f"unknown config tag {t!r} (see --list)"

    ensemble_pairs: List[Tuple[str, str]] = []
    if args.ensemble:
        a, b = args.ensemble.split(",")
        ensemble_pairs.append((a.strip(), b.strip()))

    libs_by_name, alpha_probe_lib = _load_or_build_libs_cache(args.refresh_cache)
    anchor_libs = {name: libs_by_name[name] for name in ANCHOR_CORNER_NAMES}
    all_cells = sorted(anchor_libs[ANCHOR_CORNER_NAMES[0]].cells)
    alpha_cells = sorted(alpha_probe_lib.cells)

    train_cells, val_cells = split_cells(all_cells, seed=PHASE4_CELL_SPLIT_SEED)
    dev_train_cells, dev_val_cells = split_dev(train_cells, seed=PHASE4_DEV_SPLIT_SEED)
    del val_cells  # NEVER used below -- see module docstring

    arc_attr_index = build_arc_attr_index(anchor_libs[ANCHOR_CORNER_NAMES[0]])
    family_vocab = build_family_vocab_for_phase4(all_cells, alpha_cells)

    print("building TRAIN feature matrix (320 cells, dev-only)...")
    t0 = time.time()
    ds_train = build_base_dataset(anchor_libs, train_cells, arc_attr_index, family_vocab)
    print(f"  n={ds_train.n}  build_time={time.time()-t0:.1f}s  n_features={ds_train.X.shape[1]}")

    needs_signflip = any(CONFIGS[t]["feature_mode"] == "full_plus_signflip" for t in tags)
    needs_surface = any(CONFIGS[t]["feature_mode"] == "full_plus_surface" for t in tags)
    signflip_X = None
    if needs_signflip:
        t0 = time.time()
        signflip_X = build_signflip_features(ds_train)
        print(f"  built {signflip_X.shape[1]} sign-flip features in {time.time()-t0:.1f}s")

    is_dev_train_cellmask = np.isin(ds_train.cell, np.asarray(dev_train_cells))
    is_dev_val_cellmask = np.isin(ds_train.cell, np.asarray(dev_val_cells))

    aug_cache: Dict[str, object] = {}  # topology name -> its Phase4Dataset (built at most once)

    for target_name in corners:
        print(f"\n########## target corner: {target_name} ##########")
        target_lib = libs_by_name[target_name]
        y_train_true = extract_raw_values(target_lib, ds_train.keys)
        nearest_train = ds_train.nearest_anchor(target_name)
        train_ok = trainable_mask(nearest_train)
        # Surface features read the nearest anchor, so unlike the sign-flip
        # block they are target-corner dependent -- rebuild once per corner.
        surface_X = None
        if needs_surface:
            t_surf = time.time()
            surface_X = build_surface_features(nearest_train)
            print(f"  built {surface_X.shape[1]} surface features in {time.time()-t_surf:.1f}s")

        is_dev_train = train_ok & is_dev_train_cellmask
        is_dev_val = train_ok & is_dev_val_cellmask

        nearest_dev = nearest_train[is_dev_val]
        y_dev_true = y_train_true[is_dev_val]
        groups_dev = {"table_type": ds_train.table_type[is_dev_val]}

        cached_ratio_preds: Dict[str, np.ndarray] = {}  # tag -> predicted log-ratio on dev-val

        for tag in tags:
            cfg = dict(CONFIGS[tag])
            model_kind = cfg.pop("model")
            feature_mode = cfg.pop("feature_mode")
            label_mode = cfg.pop("label_mode")
            sw_weight = cfg.pop("sample_weight", 1.0)
            sw_threshold = cfg.pop("sample_weight_threshold", 3e-4)
            sw_table_type = cfg.pop("sample_weight_table_type", "fall_power")
            per_table_type = cfg.pop("per_table_type", False)
            per_table_delta = cfg.pop("per_table_delta", None)
            augment = cfg.pop("augment", False)

            if feature_mode == "full":
                X_all = ds_train.X
            elif feature_mode == "no_lever12":
                X_all = drop_lever12_columns(ds_train.X)
            elif feature_mode == "full_plus_signflip":
                X_all = np.hstack([ds_train.X, signflip_X])
            elif feature_mode == "full_plus_surface":
                X_all = np.hstack([ds_train.X, surface_X])
            else:
                raise ValueError(f"unknown feature_mode {feature_mode!r}")
            X_fit = X_all[is_dev_train]
            X_dev = X_all[is_dev_val]

            if label_mode == "ratio":
                label_all = make_label(nearest_train, y_train_true)
            elif label_mode == "signed_ratio":
                label_all = make_label_signed_ratio(nearest_train, y_train_true)
            else:
                label_all = make_label_raw(y_train_true)
            y_fit = label_all[is_dev_train]
            y_dev = label_all[is_dev_val]

            if augment:
                if feature_mode != "full" or label_mode != "ratio":
                    raise ValueError("augment requires feature_mode='full' and label_mode='ratio'")
                if per_table_delta is not None or sw_weight != 1.0:
                    raise ValueError("augment cannot be combined with per-row delta / sample weighting "
                                      "(the extra rows would not have matching per-row arrays)")

            sample_weight = compute_sample_weights(
                ds_train.table_type[is_dev_train], nearest_train[is_dev_train],
                weight=sw_weight, threshold=sw_threshold, target_table_type=sw_table_type,
            )

            if per_table_delta is not None:
                # Per-ROW Huber clip strength keyed on table type: the
                # 2026-08-10 screen found delay tables want an aggressive
                # clip and power tables a mild one. Done inside one model,
                # so unlike `per_table_type` it costs no data dilution.
                if per_table_type:
                    raise ValueError("per_table_delta and per_table_type are mutually exclusive")
                if model_kind != "mlp":
                    raise ValueError(f"per_table_delta only applies to the MLP, not {model_kind!r}")
                if cfg.get("loss") != "huber":
                    # huber_delta is ignored by every other loss, so a config
                    # that sets one without the other is silently a no-op.
                    raise ValueError("per_table_delta requires loss='huber'")
                delay_d, power_d = per_table_delta
                power_tt = ("rise_power", "fall_power")
                cfg["huber_delta"] = np.where(
                    np.isin(ds_train.table_type[is_dev_train], power_tt), power_d, delay_d)
                cfg["huber_delta_dev"] = np.where(
                    np.isin(ds_train.table_type[is_dev_val], power_tt), power_d, delay_d)

            def _fit_predict(Xf, yf, Xd, yd, sw):
                """One (fit, predict) on an arbitrary row subset. Returns
                (dev predictions, train_seconds, short description)."""
                if model_kind == "gbdt":
                    r = fit_gbdt(Xf, yf, Xd, yd, sample_weight=sw)
                    return predict_gbdt(r, Xd), r.train_seconds, f"best_n_iter={r.best_n_iter}"
                r = fit_mlp(Xf, yf, Xd, yd, sample_weight=sw, **cfg)
                return predict_mlp(r, Xd), r.train_seconds, f"best_epoch={r.best_epoch} device={r.device}"

            if augment:
                topo, aug_target = augment_topology_for(target_name)
                if topo is None:
                    raise RuntimeError(f"no augmenting topology available for {target_name!r}")
                ds_aug = aug_cache.get(topo.name)
                if ds_aug is None:
                    t_aug = time.time()
                    ds_aug = build_base_dataset(
                        {n: libs_by_name[n] for n in topo.anchor_names},
                        train_cells, arc_attr_index, family_vocab,
                        anchor_names=topo.anchor_names,
                    )
                    aug_cache[topo.name] = ds_aug
                    print(f"  built {topo.name}-topology dataset n={ds_aug.n} "
                          f"in {time.time()-t_aug:.1f}s")
                nearest_aug = ds_aug.nearest_anchor(aug_target, topo.nearest_anchor_by_target)
                y_aug_true = extract_raw_values(libs_by_name[aug_target], ds_aug.keys)
                keep_aug = trainable_mask(nearest_aug) & np.isin(ds_aug.cell, np.asarray(dev_train_cells))
                X_fit = np.vstack([X_fit, ds_aug.X[keep_aug]])
                y_fit = np.concatenate([y_fit, make_label(nearest_aug, y_aug_true)[keep_aug]])
                print(f"  [{tag}] augmented with {topo.name} topology targeting {aug_target} "
                      f"(anchor {topo.nearest_anchor_by_target[aug_target]}): "
                      f"+{int(keep_aug.sum())} rows -> {X_fit.shape[0]} total")

            t0 = time.time()
            if per_table_type:
                # One model per table type instead of a single model with a
                # table_type one-hot: fall_power (signed, crosses zero) shares
                # no representation with the four delay tables, which already
                # score 98.5-99.4 (2026-08-10 round, recommendation 2).
                tt_fit = ds_train.table_type[is_dev_train]
                tt_dev = ds_train.table_type[is_dev_val]
                y_pred = np.zeros(X_dev.shape[0], dtype=np.float64)
                train_s = 0.0
                parts: List[str] = []
                for tt in sorted(set(tt_dev.tolist())):
                    m_fit = tt_fit == tt
                    m_dev = tt_dev == tt
                    if not m_fit.any():
                        raise RuntimeError(f"table_type {tt!r} present in dev-val but absent from dev-train")
                    sw_sub = sample_weight[m_fit] if sample_weight is not None else None
                    p, ts, desc = _fit_predict(X_fit[m_fit], y_fit[m_fit], X_dev[m_dev], y_dev[m_dev], sw_sub)
                    y_pred[m_dev] = p
                    train_s += ts
                    parts.append(f"{tt}={desc.split()[0].split('=')[1]}")
                extra = "per_table_type(" + ",".join(parts) + ")"
            else:
                y_pred, train_s, extra = _fit_predict(X_fit, y_fit, X_dev, y_dev, sample_weight)
            elapsed = time.time() - t0

            if label_mode == "ratio":
                cached_ratio_preds[tag] = y_pred
                pred = reconstruct_predictions(nearest_dev, y_pred)
            elif label_mode == "signed_ratio":
                pred = reconstruct_signed_ratio(nearest_dev, y_pred)
            else:
                pred = reconstruct_raw(nearest_dev, y_pred)

            overall, n, bd = score_breakdown(y_dev_true, pred, groups_dev)
            lines = [f"  [{tag}] feature_mode={feature_mode} label_mode={label_mode} "
                     f"wall={elapsed:.1f}s {extra} overall={overall:.4f} (n={n})"]
            for tt, (s, npts) in sorted(bd["table_type"].items()):
                lines.append(f"      {tt:18s} {s:8.4f}  (n={npts})")
            _print_and_log(lines + [_row_md(target_name, tag, overall, bd["table_type"], train_s,
                                             f"feature_mode={feature_mode};label_mode={label_mode};{extra}")])

        for tag_a, tag_b in ensemble_pairs:
            if tag_a not in cached_ratio_preds or tag_b not in cached_ratio_preds:
                print(f"  [ensemble {tag_a}+{tag_b}] skipped -- both tags must be in --configs and ratio-mode")
                continue
            ens_y = 0.5 * (cached_ratio_preds[tag_a] + cached_ratio_preds[tag_b])
            ens_pred = reconstruct_predictions(nearest_dev, ens_y)
            overall, n, bd = score_breakdown(y_dev_true, ens_pred, groups_dev)
            tag = f"ensemble({tag_a}+{tag_b})"
            lines = [f"  [{tag}] overall={overall:.4f} (n={n})"]
            for tt, (s, npts) in sorted(bd["table_type"].items()):
                lines.append(f"      {tt:18s} {s:8.4f}  (n={npts})")
            _print_and_log(lines + [_row_md(target_name, tag, overall, bd["table_type"], None, "")])

    print(f"\nlog appended to {LOG_MD}")


if __name__ == "__main__":
    main()
