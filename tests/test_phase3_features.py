import numpy as np
import pytest

from features.corners import parse_corner_filename
from liberty.parser import parse_file
from models.phase2_scaling import fit_phase2_model
from models.phase3_features import (
    CELL_SPLIT_SEED,
    DEV_SPLIT_SEED,
    FEATURE_NAMES,
    build_arc_attr_index,
    build_family_vocab,
    build_feature_matrix,
    filter_lib_cells,
    make_label,
    ordered_full_corner_pairs,
    reconstruct_predictions,
    score_breakdown,
    split_cells,
    split_dev,
)

from helpers import FULL_DIR


# ---------------------------------------------------------------------------
# split determinism / cell-leakage guarantees (pure functions, no I/O)
# ---------------------------------------------------------------------------


def test_split_cells_is_deterministic_for_same_seed():
    cells = [f"CELL{i}" for i in range(100)]
    train_a, val_a = split_cells(cells, seed=CELL_SPLIT_SEED)
    train_b, val_b = split_cells(cells, seed=CELL_SPLIT_SEED)
    assert train_a == train_b
    assert val_a == val_b


def test_split_cells_is_order_independent():
    """docs/plan.md Phase 3: the split must be reproducible regardless of
    how the caller happens to enumerate the cell names (dict/set
    iteration order is not guaranteed by Python)."""
    cells = [f"CELL{i}" for i in range(100)]
    shuffled = list(reversed(cells))
    train_a, val_a = split_cells(cells, seed=CELL_SPLIT_SEED)
    train_b, val_b = split_cells(shuffled, seed=CELL_SPLIT_SEED)
    assert train_a == train_b
    assert val_a == val_b


def test_split_cells_covers_all_cells_with_no_overlap():
    cells = [f"CELL{i}" for i in range(100)]
    train, val = split_cells(cells)
    assert len(train) == 80
    assert len(val) == 20
    assert set(train) & set(val) == set()
    assert set(train) | set(val) == set(cells)


def test_split_cells_different_seed_gives_different_split():
    cells = [f"CELL{i}" for i in range(100)]
    train_a, val_a = split_cells(cells, seed=CELL_SPLIT_SEED)
    train_b, val_b = split_cells(cells, seed=CELL_SPLIT_SEED + 1)
    assert val_a != val_b


def test_split_dev_stays_within_train_cells_never_touches_validation():
    """docs/plan.md Phase 3: "早停用訓練 cell 內部再切的 dev 子集，絕不能
    碰 20% 驗證 cell". split_dev only ever receives the 80 train cells as
    input, so this is structurally guaranteed, but assert it explicitly
    against the real 100-cell split as an end-to-end sanity check."""
    all_cells = [f"CELL{i}" for i in range(100)]
    train_cells, val_cells = split_cells(all_cells)
    dev_train, dev_val = split_dev(train_cells, seed=DEV_SPLIT_SEED)

    assert set(dev_train) | set(dev_val) == set(train_cells)
    assert set(dev_train) & set(dev_val) == set()
    assert set(dev_train) & set(val_cells) == set()
    assert set(dev_val) & set(val_cells) == set()
    assert len(dev_train) == 64
    assert len(dev_val) == 16


def test_ordered_full_corner_pairs_covers_every_ordered_combination():
    class FakeMeta:
        def __init__(self, name):
            self.name = name

    metas = [FakeMeta(n) for n in ["a", "b", "c", "d", "e"]]
    pairs = ordered_full_corner_pairs(metas)
    assert len(pairs) == 20  # 5*4, source != target
    names = {(s.name, t.name) for s, t in pairs}
    assert all(s != t for s, t in names)
    assert len(names) == 20


# ---------------------------------------------------------------------------
# label / reconstruction round-trip (pure math, no I/O)
# ---------------------------------------------------------------------------


class _FakeDataset:
    def __init__(self, anchor, target, phase25_pred):
        self.anchor = np.asarray(anchor, dtype=float)
        self.target = np.asarray(target, dtype=float)
        self.phase25_pred = np.asarray(phase25_pred, dtype=float)


@pytest.mark.parametrize("mode", ["raw", "residual"])
def test_make_label_and_reconstruct_predictions_roundtrip(mode):
    anchor = np.array([1.0, 2.0, -3.0, 0.5])
    target = np.array([1.5, 1.0, -6.0, 0.25])
    phase25_pred = np.array([1.2, 1.8, -4.0, 0.4])
    ds = _FakeDataset(anchor, target, phase25_pred)

    label = make_label(ds, mode)
    reconstructed = reconstruct_predictions(ds, label, mode)
    np.testing.assert_allclose(reconstructed, target, rtol=1e-8)


def test_reconstruct_predictions_forces_zero_anchor_to_zero_regardless_of_prediction():
    ds = _FakeDataset(anchor=[0.0, 1.0], target=[0.0, 2.0], phase25_pred=[0.0, 1.9])
    wild_pred = np.array([37.0, 0.1])  # would blow up exp() for the anchor==0 row
    reconstructed = reconstruct_predictions(ds, wild_pred, "raw")
    assert reconstructed[0] == 0.0
    assert np.isfinite(reconstructed).all()


def test_reconstruct_predictions_clips_extreme_outputs_to_finite():
    ds = _FakeDataset(anchor=[1.0], target=[1.0], phase25_pred=[1.0])
    reconstructed = reconstruct_predictions(ds, np.array([1e6]), "raw")
    assert np.isfinite(reconstructed).all()


def test_score_breakdown_matches_scorer_overall_and_sums_group_counts():
    y_true = np.array([1.0, 2.0, 4.0, 0.0])
    y_pred = np.array([1.0, 2.0, 4.0, 0.0])
    groups = {"table_type": np.array(["a", "a", "b", "b"])}
    overall, n, breakdown = score_breakdown(y_true, y_pred, groups)
    assert overall == pytest.approx(100.0)
    assert n == 4
    assert sum(v[1] for v in breakdown["table_type"].values()) == 4


# ---------------------------------------------------------------------------
# real-data feature matrix construction: zero validation-cell leakage
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_libs():
    return {parse_corner_filename(p): parse_file(str(p)) for p in sorted(FULL_DIR.glob("*.lib"))}


@pytest.fixture(scope="module")
def split(full_libs):
    all_cells = sorted(next(iter(full_libs.values())).cells)
    return split_cells(all_cells)


@pytest.fixture(scope="module")
def small_phase25_model(full_libs, split):
    """Fit on a small subset of train cells only, for test speed -- the
    leakage guarantee under test is about *cell membership*, not about
    how well-fit the model is."""
    train_cells, _val_cells = split
    small_train = train_cells[:10]
    filtered = {m: filter_lib_cells(lib, small_train) for m, lib in full_libs.items()}
    return fit_phase2_model(filtered), small_train


def test_build_feature_matrix_never_includes_validation_cells(full_libs, split, small_phase25_model):
    train_cells, val_cells = split
    model, small_train = small_phase25_model
    all_cells = sorted(next(iter(full_libs.values())).cells)
    family_vocab = build_family_vocab(all_cells)
    arc_index = build_arc_attr_index(next(iter(full_libs.values())))
    pairs = ordered_full_corner_pairs(full_libs.keys())[:2]  # keep the test fast

    ds = build_feature_matrix(full_libs, small_train, pairs, model, arc_index, family_vocab)

    assert set(ds.cell.tolist()) <= set(small_train)
    assert set(ds.cell.tolist()) & set(val_cells) == set()
    assert ds.X.shape == (ds.n, len(FEATURE_NAMES))
    assert np.isfinite(ds.X).all()
    assert np.isfinite(ds.anchor).all()
    assert np.isfinite(ds.target).all()
    assert np.isfinite(ds.phase25_pred).all()


def test_build_feature_matrix_restricted_to_validation_cells_only_returns_those_cells(
    full_libs, split, small_phase25_model
):
    train_cells, val_cells = split
    model, _small_train = small_phase25_model
    all_cells = sorted(next(iter(full_libs.values())).cells)
    family_vocab = build_family_vocab(all_cells)
    arc_index = build_arc_attr_index(next(iter(full_libs.values())))
    pairs = ordered_full_corner_pairs(full_libs.keys())[:2]

    small_val = val_cells[:5]
    ds_val = build_feature_matrix(full_libs, small_val, pairs, model, arc_index, family_vocab)
    assert set(ds_val.cell.tolist()) <= set(small_val)
    assert set(ds_val.cell.tolist()) & set(train_cells) == set()
