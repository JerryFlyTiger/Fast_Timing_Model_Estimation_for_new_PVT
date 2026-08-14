"""Phase 3 version B: a small PyTorch MLP on top of the same
feature/label pipeline as models.phase3_gbdt (models.phase3_features).

docs/plan.md Phase 3: "可繳交格式；驗證 NN 在此資料上的表現". CPU-only
training (`torch` installed from the CPU wheel index, see
requirements.txt), 3 hidden layers, plain feed-forward ReLU network.

Early stopping mirrors phase3_gbdt.fit_gbdt exactly: `dev_train_cells` /
`dev_val_cells` are both subsets of the 80 training cells
(models.phase3_features.split_dev), so the 20% validation cells are never
touched during training, per docs/plan.md Phase 3's "防過欄合" clause.
Feature standardization (mean/std) is likewise fit only on the dev-train
rows -- see `fit_scaler` -- not on the validation cells.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
import torch
from torch import nn

from models.phase3_features import Phase3Dataset, make_label

HIDDEN_SIZES = (128, 64, 32)
LEARNING_RATE = 1e-3
BATCH_SIZE = 8192
MAX_EPOCHS = 60
PATIENCE = 5
WEIGHT_DECAY = 1e-5
RANDOM_STATE = 0
PREDICT_CHUNK = 200_000


class Phase3MLP(nn.Module):
    """2-3 hidden layer feed-forward regressor, per docs/plan.md Phase 3
    ("PyTorch 小型 MLP（2-3 隱藏層）"). Default `HIDDEN_SIZES` uses 3."""

    def __init__(self, n_features: int, hidden_sizes: Sequence[int] = HIDDEN_SIZES):
        super().__init__()
        layers: List[nn.Module] = []
        prev = n_features
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


def fit_scaler(X: np.ndarray) -> FeatureScaler:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)  # guard constant columns (e.g. an
    # all-zero one-hot column if a category never appears in dev-train)
    return FeatureScaler(mean=mean.astype(np.float32), std=std.astype(np.float32))


@dataclass
class MLPFitResult:
    model: Phase3MLP
    scaler: FeatureScaler
    label_mode: str
    best_epoch: int
    dev_val_losses: List[float] = field(default_factory=list)
    train_seconds: float = 0.0


def _cell_mask(ds: Phase3Dataset, cells: Sequence[str]) -> np.ndarray:
    return np.isin(ds.cell, np.asarray(list(cells)))


def fit_mlp(
    ds_train: Phase3Dataset,
    dev_train_cells: Sequence[str],
    dev_val_cells: Sequence[str],
    *,
    label_mode: str = "residual",
    hidden_sizes: Sequence[int] = HIDDEN_SIZES,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    weight_decay: float = WEIGHT_DECAY,
    seed: int = RANDOM_STATE,
) -> MLPFitResult:
    t0 = time.time()
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    y_all = make_label(ds_train, label_mode)
    trainable = ds_train.trainable_mask
    is_dev_train = trainable & _cell_mask(ds_train, dev_train_cells)
    is_dev_val = trainable & _cell_mask(ds_train, dev_val_cells)

    X_fit_raw = ds_train.X[is_dev_train]
    y_fit = y_all[is_dev_train].astype(np.float32)
    X_dev_raw = ds_train.X[is_dev_val]
    y_dev = y_all[is_dev_val].astype(np.float32)

    scaler = fit_scaler(X_fit_raw)
    X_fit = scaler.transform(X_fit_raw).astype(np.float32)
    X_dev = scaler.transform(X_dev_raw).astype(np.float32)

    model = Phase3MLP(X_fit.shape[1], hidden_sizes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    X_fit_t = torch.from_numpy(X_fit)
    y_fit_t = torch.from_numpy(y_fit)
    X_dev_t = torch.from_numpy(X_dev)
    y_dev_t = torch.from_numpy(y_dev)

    n = X_fit_t.shape[0]
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0
    dev_val_losses: List[float] = []

    rng = np.random.default_rng(seed)
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = X_fit_t[idx]
            yb = y_fit_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            dev_pred = model(X_dev_t)
            dev_loss = float(loss_fn(dev_pred, y_dev_t).item())
        dev_val_losses.append(dev_loss)

        if dev_loss < best_loss - 1e-6:
            best_loss = dev_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return MLPFitResult(
        model=model,
        scaler=scaler,
        label_mode=label_mode,
        best_epoch=best_epoch,
        dev_val_losses=dev_val_losses,
        train_seconds=time.time() - t0,
    )


def predict_mlp(fit_result: MLPFitResult, X: np.ndarray) -> np.ndarray:
    """Batched inference (docs/plan.md: avoid materializing a huge
    activation tensor for the ~1M-row validation set in one shot)."""
    fit_result.model.eval()
    X_scaled = fit_result.scaler.transform(X.astype(np.float32)).astype(np.float32)
    out = np.empty(X_scaled.shape[0], dtype=np.float64)
    with torch.no_grad():
        for start in range(0, X_scaled.shape[0], PREDICT_CHUNK):
            chunk = torch.from_numpy(X_scaled[start : start + PREDICT_CHUNK])
            out[start : start + PREDICT_CHUNK] = fit_result.model(chunk).numpy()
    return out
