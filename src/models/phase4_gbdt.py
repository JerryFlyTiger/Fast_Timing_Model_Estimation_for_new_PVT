"""Phase 4 version A: scikit-learn HistGradientBoostingRegressor, one per
delivery corner, trained on models.phase4_features's log-ratio-vs-
nearest-anchor label.

This module is deliberately decoupled from any particular Dataset class
(`fit_gbdt`/`predict_gbdt` take plain numpy arrays) since
models.phase4_features.Phase4Dataset is corner-agnostic -- the caller
(scripts/phase4_validate.py) slices out the per-corner label/trainable
mask/dev-split before calling in here. Early-stopping mechanics (manual
cell-level warm-start loop) are unchanged from models.phase3_gbdt -- see
that module's docstring for why a cell-level split is used instead of
sklearn's own row-level `early_stopping=True`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

MAX_ITER_CAP = 500
CHECK_EVERY = 20
PATIENCE = 3  # consecutive non-improving dev-loss checks before stopping
LEARNING_RATE = 0.1
MAX_LEAF_NODES = 63
MIN_SAMPLES_LEAF = 50
RANDOM_STATE = 0


@dataclass
class GBDTFitResult:
    model: HistGradientBoostingRegressor
    best_n_iter: int
    dev_val_losses: List[float] = field(default_factory=list)
    train_seconds: float = 0.0


def fit_gbdt(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    *,
    max_iter_cap: int = MAX_ITER_CAP,
    check_every: int = CHECK_EVERY,
    patience: int = PATIENCE,
    learning_rate: float = LEARNING_RATE,
    max_leaf_nodes: int = MAX_LEAF_NODES,
    min_samples_leaf: int = MIN_SAMPLES_LEAF,
    random_state: int = RANDOM_STATE,
    sample_weight: np.ndarray | None = None,
) -> GBDTFitResult:
    """`sample_weight` (2026-07-27 fall_power improvement round, optional,
    default None == uniform): per-row training weight, e.g. up-weighting
    near-zero fall_power rows (docs/phase4b_screen_log.md diagnosis --
    those rows carry most of the fall_power score gap). Only applied to
    `.fit()` calls -- the dev-val loss used for early stopping stays
    unweighted so it tracks the actual (unweighted) contest scorer."""
    t0 = time.time()

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        max_iter=check_every,
        warm_start=True,
        early_stopping=False,  # manual cell-level early stopping below
        random_state=random_state,
    )

    best_loss = np.inf
    best_iter = 0
    no_improve = 0
    dev_val_losses: List[float] = []
    n_iter = 0
    while n_iter < max_iter_cap:
        model.max_iter = n_iter + check_every
        model.fit(X_fit, y_fit, sample_weight=sample_weight)
        n_iter = model.n_iter_
        pred = model.predict(X_dev)
        loss = float(np.mean((pred - y_dev) ** 2))
        dev_val_losses.append(loss)
        if loss < best_loss - 1e-9:
            best_loss = loss
            best_iter = n_iter
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    # Refit a clean model truncated to `best_iter` trees rather than
    # relying on the (larger, possibly-overfit) final warm_start state.
    final_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        max_iter=max(best_iter, 1),
        random_state=random_state,
    )
    final_model.fit(X_fit, y_fit, sample_weight=sample_weight)

    return GBDTFitResult(
        model=final_model,
        best_n_iter=best_iter,
        dev_val_losses=dev_val_losses,
        train_seconds=time.time() - t0,
    )


def predict_gbdt(fit_result: GBDTFitResult, X: np.ndarray) -> np.ndarray:
    return fit_result.model.predict(X)
