"""Phase 4 finalized production config registry (2026-07-27, post
improvement-round screening -- see `docs/phase4b_screen_log.md` and
`docs/phase4_results.md`).

Shared by `scripts/phase4_final_validate.py` (80/20-cell acceptance
protocol) and `scripts/phase4_final_predict.py` (full-400-cell delivery
retrain) so both scripts fit *exactly* the same model type/hyperparameters
for a given `--config` tag -- the acceptance run and the delivery retrain
must never silently diverge in what they measure vs. what they ship.

Winning config as of 2026-08-11: **`mlp_w256_b4_huber`** -- identical to
`mlp_w256_b4_full` below except for the training loss (see that entry and
docs/round_20260810.md section 4.1). `mlp_w256_b4_full` was the winner of
the 2026-07-27 improvement round and remains the reference against which
every later arm is measured; everything the next paragraph says about
architecture, features and label applies unchanged to both.

`mlp_w256_b4_full`
-- `models.phase4_mlp`'s residual MLP at its module defaults (width=256,
n_blocks=4), on the full feature set (`models.phase4_features.
build_base_dataset` already unconditionally includes the lever-1/2
response-signature + spatial-gradient features -- there is no separate
"ablated" feature build in production), with the plain ratio label
(`models.phase4_features.make_label` / `reconstruct_predictions`, i.e.
log(|target|/|nearest_anchor|)). The `signed_ratio` label, sign-flip
features, and sample-weighting variants explored during screening are
**not** part of this registry: they either underperformed or were not
chosen (see docs/phase4b_screen_log.md's fall_power diagnosis notes).
`gbdt_full` is kept as a fallback/comparison option (also the module
defaults for `models.phase4_gbdt.fit_gbdt`), not the selected winner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from models.phase4_gbdt import fit_gbdt, predict_gbdt
from models.phase4_mlp import fit_mlp, predict_mlp

# 2026-08-11: repointed from "mlp_w256_b4_full" after the official protocol
# (80 held-out cells, all 10 corners, 1 seed) measured the huber-loss arm at
# +0.0616 pooled (96.2742 -> 96.3359), per-corner mean +0.0613 +- 0.0376,
# 9/10 corners positive, ~95% CI [+0.038, +0.085]. NOTE the documented
# 3-seed numbers (96.32 training caliber / 97.34 alpha caliber) belong to the
# OLD default -- a 3-seed huber run was never completed, so the huber
# alpha-caliber number is unmeasured. See docs/round_20260810.md section 4.
DEFAULT_CONFIG_TAG = "mlp_w256_b4_huber"

# tag -> dict(model="gbdt"|"mlp", **kwargs passed straight to fit_gbdt/fit_mlp).
# Every entry here uses the plain ratio label + the full (always-on lever
# 1/2) feature set -- no per-config feature/label switching in production.
CONFIGS: Dict[str, Dict[str, Any]] = {
    "mlp_w256_b4_full": dict(model="mlp", width=256, n_blocks=4, max_epochs=150, patience=15),
    # 2026-08-10 robust-loss round: identical to `mlp_w256_b4_full` except
    # the training loss clips large label-space residuals at log 2 (see
    # models.phase4_mlp `_elementwise_loss`). Dev-only screen won on 3/3
    # corners (mean +0.063); the `score` (exact capped metric) and
    # per-table-delta variants were screened and did not beat it. The
    # official protocol then measured +0.0616 pooled on the 80 held-out
    # cells, which is why `DEFAULT_CONFIG_TAG` above now points here --
    # see docs/round_20260810.md section 4.1 for the per-corner table and
    # for what that measurement does and does not establish.
    "mlp_w256_b4_huber": dict(model="mlp", width=256, n_blocks=4, max_epochs=150, patience=15,
                               loss="huber"),
    "mlp_w192_b3_full": dict(model="mlp", width=192, n_blocks=3, max_epochs=100, patience=12),
    "gbdt_full": dict(model="gbdt"),
    # Pipeline-plumbing smoke tests only (tiny capacity/epoch caps) --
    # NOT production candidates, never use for a real acceptance/delivery
    # run. Exist so scripts/phase4_final_{validate,predict}.py can be
    # smoke-tested end-to-end (imports, leakage assertions, writer
    # integration) in well under a minute instead of the several minutes
    # a real mlp_w256_b4_full fit takes per corner.
    "mlp_tiny_smoke": dict(model="mlp", width=32, n_blocks=1, max_epochs=3, patience=2),
    "gbdt_tiny_smoke": dict(model="gbdt", max_iter_cap=20, check_every=10, patience=1),
}


@dataclass
class FitHandle:
    """Opaque wrapper so callers don't need to branch on model kind
    themselves -- `predict()` dispatches to the right predict_* function."""

    kind: str  # "gbdt" | "mlp"
    result: Any
    train_seconds: float
    info: str  # short human-readable summary (best_n_iter / best_epoch etc.)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "gbdt":
            return predict_gbdt(self.result, X)
        return predict_mlp(self.result, X)


def fit_config(
    tag: str,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    *,
    seed: Optional[int] = None,
) -> FitHandle:
    """`seed` (2026-07-29 seed-ensemble addition, optional): overrides the
    config's own random seed for this one fit -- `fit_gbdt`'s
    `random_state` kwarg for a gbdt config, `fit_mlp`'s `seed` kwarg
    (governs both weight init and minibatch shuffling) for an mlp config.
    `None` (default) leaves the config's own default seed untouched, so
    every pre-2026-07-29 call site (a single seed per corner) is
    unaffected. `scripts/phase4_final_validate.py --seeds N` calls this
    N times with N different seeds on the *same* already-built feature
    matrix and averages the resulting log-ratio predictions."""
    if tag not in CONFIGS:
        raise ValueError(f"unknown config tag {tag!r}; available: {sorted(CONFIGS)}")
    cfg = dict(CONFIGS[tag])
    kind = cfg.pop("model")
    if kind == "gbdt":
        if seed is not None:
            cfg["random_state"] = seed
        res = fit_gbdt(X_fit, y_fit, X_dev, y_dev, **cfg)
        return FitHandle(kind="gbdt", result=res, train_seconds=res.train_seconds,
                          info=f"best_n_iter={res.best_n_iter}")
    elif kind == "mlp":
        if seed is not None:
            cfg["seed"] = seed
        res = fit_mlp(X_fit, y_fit, X_dev, y_dev, **cfg)
        return FitHandle(kind="mlp", result=res, train_seconds=res.train_seconds,
                          info=f"best_epoch={res.best_epoch} device={res.device}")
    else:
        raise ValueError(f"unknown model kind {kind!r} in config {tag!r}")


__all__ = ["DEFAULT_CONFIG_TAG", "CONFIGS", "FitHandle", "fit_config"]
