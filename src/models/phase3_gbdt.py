"""Phase 3 version A: scikit-learn HistGradientBoostingRegressor on top of
the shared Phase 3 feature/label pipeline (models.phase3_features).

docs/plan.md Phase 3: "表格型資料的預期最強基準". Trained on the log-ratio
label produced by `phase3_features.make_label` (either "raw" or
"residual" mode -- both versions A and B must agree on which, per the
task's "兩版必須用同一目標定義").

Early stopping (docs/plan.md Phase 3 "防過欄合": "早停用訓練 cell 內部再切
的 dev 子集，絕不能碰 20% 驗證 cell") is done *by cell*, not by row:
`dev_train_cells`/`dev_val_cells` (models.phase3_features.split_dev) are
both subsets of the 80 training cells, so this never touches the 20%
validation cells. sklearn's own `early_stopping=True` only supports a
random *row*-level split from whatever X/y it is given, which -- since
rows from the same cell appear at many (pair, table, grid-point)
combinations -- would let closely related rows leak across the
train/dev-val boundary within a cell; a cell-level split avoids that.
`warm_start=True` + manually growing `max_iter` lets us evaluate a
cell-level dev loss every `check_every` boosting rounds without repeatedly
refitting from scratch (verified empirically: a second `.fit()` call with
a larger `max_iter` only trains the incremental trees).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from models.phase3_features import Phase3Dataset, make_label

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
    label_mode: str
    best_n_iter: int
    dev_val_losses: List[float] = field(default_factory=list)
    train_seconds: float = 0.0


def _cell_mask(ds: Phase3Dataset, cells: Sequence[str]) -> np.ndarray:
    return np.isin(ds.cell, np.asarray(list(cells)))


def fit_gbdt(
    ds_train: Phase3Dataset,
    dev_train_cells: Sequence[str],
    dev_val_cells: Sequence[str],
    *,
    label_mode: str = "residual",
    max_iter_cap: int = MAX_ITER_CAP,
    check_every: int = CHECK_EVERY,
    patience: int = PATIENCE,
    learning_rate: float = LEARNING_RATE,
    max_leaf_nodes: int = MAX_LEAF_NODES,
    min_samples_leaf: int = MIN_SAMPLES_LEAF,
    random_state: int = RANDOM_STATE,
) -> GBDTFitResult:
    t0 = time.time()
    y_all = make_label(ds_train, label_mode)
    trainable = ds_train.trainable_mask

    is_dev_train = trainable & _cell_mask(ds_train, dev_train_cells)
    is_dev_val = trainable & _cell_mask(ds_train, dev_val_cells)
    X_fit, y_fit = ds_train.X[is_dev_train], y_all[is_dev_train]
    X_dev, y_dev = ds_train.X[is_dev_val], y_all[is_dev_val]

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
        model.fit(X_fit, y_fit)
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
    final_model.fit(X_fit, y_fit)

    return GBDTFitResult(
        model=final_model,
        label_mode=label_mode,
        best_n_iter=best_iter,
        dev_val_losses=dev_val_losses,
        train_seconds=time.time() - t0,
    )


def predict_gbdt(fit_result: GBDTFitResult, X: np.ndarray) -> np.ndarray:
    return fit_result.model.predict(X)
