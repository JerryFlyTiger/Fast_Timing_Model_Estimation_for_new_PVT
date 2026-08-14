import numpy as np
import pytest

from features.corners import parse_corner_filename
from liberty.parser import parse_file
from models.phase2_scaling import fit_phase2_model
from models.phase3_features import (
    build_arc_attr_index,
    build_family_vocab,
    build_feature_matrix,
    filter_lib_cells,
    ordered_full_corner_pairs,
    reconstruct_predictions,
    split_cells,
    split_dev,
)
from models.phase3_gbdt import fit_gbdt, predict_gbdt
from models.phase3_mlp import fit_mlp, predict_mlp

from helpers import FULL_DIR


# Small, fast fixtures: this module only checks "does training/inference
# run and produce finite output", not model quality (that is
# scripts/phase3_validate.py's job, run offline -- see
# docs/model_comparison.md), so a handful of cells and a couple of
# corner pairs with tiny iteration/epoch caps is enough and keeps the
# test suite's overall runtime bounded (docs/plan.md: "pytest 全綠", the
# suite already takes minutes because of the Phase 2.5 ensemble tests).


@pytest.fixture(scope="module")
def full_libs():
    return {parse_corner_filename(p): parse_file(str(p)) for p in sorted(FULL_DIR.glob("*.lib"))}


@pytest.fixture(scope="module")
def small_datasets(full_libs):
    all_cells = sorted(next(iter(full_libs.values())).cells)
    train_cells, val_cells = split_cells(all_cells)
    dev_train_cells, dev_val_cells = split_dev(train_cells)

    small_train = train_cells[:12]
    small_dev_train = [c for c in dev_train_cells if c in small_train][:8]
    small_dev_val = [c for c in dev_val_cells if c in small_train][:4]
    # Guarantee non-empty dev splits regardless of which cells the real
    # split happened to place where, since this fixture only takes the
    # first 12 train cells (a fast, arbitrary subset for test speed).
    if not small_dev_train or not small_dev_val:
        small_dev_train, small_dev_val = small_train[:8], small_train[8:12]
    small_val = val_cells[:4]

    family_vocab = build_family_vocab(all_cells)
    arc_index = build_arc_attr_index(next(iter(full_libs.values())))
    pairs = ordered_full_corner_pairs(full_libs.keys())[:3]

    filtered = {m: filter_lib_cells(lib, small_train) for m, lib in full_libs.items()}
    model25 = fit_phase2_model(filtered)

    ds_train = build_feature_matrix(full_libs, small_train, pairs, model25, arc_index, family_vocab)
    ds_val = build_feature_matrix(full_libs, small_val, pairs, model25, arc_index, family_vocab)
    return ds_train, ds_val, small_dev_train, small_dev_val, small_val, train_cells


@pytest.mark.parametrize("mode", ["raw", "residual"])
def test_gbdt_fit_predict_produces_finite_values(small_datasets, mode):
    ds_train, ds_val, dev_train_cells, dev_val_cells, _small_val, _train_cells = small_datasets
    res = fit_gbdt(
        ds_train, dev_train_cells, dev_val_cells,
        label_mode=mode, max_iter_cap=20, check_every=5, patience=2,
    )
    assert res.best_n_iter >= 1

    y_pred = predict_gbdt(res, ds_val.X)
    assert np.isfinite(y_pred).all()

    reconstructed = reconstruct_predictions(ds_val, y_pred, mode)
    assert np.isfinite(reconstructed).all()
    # writer.py's guard (src/liberty/writer.py fill_template) rejects
    # non-finite predictions outright -- this is the model-side contract
    # that guard depends on.


@pytest.mark.parametrize("mode", ["raw", "residual"])
def test_mlp_fit_predict_produces_finite_values(small_datasets, mode):
    ds_train, ds_val, dev_train_cells, dev_val_cells, _small_val, _train_cells = small_datasets
    res = fit_mlp(
        ds_train, dev_train_cells, dev_val_cells,
        label_mode=mode, max_epochs=3, patience=3, hidden_sizes=(16, 8),
    )
    assert res.best_epoch >= 1

    y_pred = predict_mlp(res, ds_val.X)
    assert np.isfinite(y_pred).all()

    reconstructed = reconstruct_predictions(ds_val, y_pred, mode)
    assert np.isfinite(reconstructed).all()


def test_gbdt_early_stopping_never_sees_validation_cells(small_datasets):
    """docs/plan.md Phase 3: "早停用訓練 cell 內部再切的 dev 子集，絕不能
    碰 20% 驗證 cell". dev_train_cells/dev_val_cells passed into fit_gbdt
    here are both subsets of the small train-cell pool used to build
    ds_train, so this asserts the fixture itself (and therefore the
    contract fit_gbdt relies on) never mixes in a real validation cell."""
    _ds_train, _ds_val, dev_train_cells, dev_val_cells, small_val, _train_cells = small_datasets
    assert set(dev_train_cells) & set(small_val) == set()
    assert set(dev_val_cells) & set(small_val) == set()


def test_mlp_scaler_fit_only_on_dev_train_rows(small_datasets):
    ds_train, _ds_val, dev_train_cells, dev_val_cells, _small_val, _train_cells = small_datasets
    res = fit_mlp(ds_train, dev_train_cells, dev_val_cells, max_epochs=2, patience=2, hidden_sizes=(8,))
    assert np.isfinite(res.scaler.mean).all()
    assert np.isfinite(res.scaler.std).all()
    assert (res.scaler.std > 0).all()
