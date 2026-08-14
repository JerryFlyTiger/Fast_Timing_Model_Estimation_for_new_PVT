"""Phase 3 dataset construction: 80/20 cell split + cross-corner-pair
sample assembly + feature engineering shared by both ML versions
(models/phase3_gbdt.py, models/phase3_mlp.py) and the validation script
(scripts/phase3_validate.py).

docs/plan.md Phase 3 "使用者驗收門檻" (2026-07-26):

- The 80/20 split is **by cell** (80 train cells / 20 validation cells
  out of the 100 cells shared by every corner), not by point -- a
  point-level split would leak table-interior information (adjacent grid
  points of the same table are highly correlated) and inflate the score.
- Samples are built from **every ordered cross-corner pair** (source ->
  target) among the 5 full corners in `testcase/alpha_test/full/` -- 5*4 = 20
  ordered pairs, so all 5 full corners participate on both the source and
  target side. Training rows only ever come from the 80 train cells;
  validation rows are built the same way but restricted to the 20
  validation cells, mirroring the real delivery input shape (anchor table
  + corner/cell/grid features -> predict the other corner's true values;
  the anchor is an input, never the label -- docs/plan.md explicitly
  allows this).
- The Phase 2.5 physical model (`models.phase2_scaling`) is refit **on
  the 80 train cells only** (never on the 20 validation cells) so that
  its own predictions, used here both as (a) a comparison baseline under
  this same protocol and (b) an input feature for the two ML models
  ("讓 ML 學殘差修正的效果"), carry zero information about the validation
  cells. This is a stricter leakage bar than the real Phase 2.5 delivery
  fit (`scripts/phase2_predict.py`), which uses all 100 cells -- this
  module's fit is *only* for the Phase 3 comparison protocol.

All seeds are fixed module constants (recorded here and in
docs/model_comparison.md, per the task's "seed 寫死在 config 並記錄在結果
文件" requirement) so every run reproduces the same split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from features.cellinfo import parse_cell_name
from features.corners import CornerMeta
from liberty.parser import TABLE_KINDS, LibertyFile, TableKey
from models.phase2_scaling import Phase2Model

# ---------------------------------------------------------------------------
# Fixed seeds / split fractions (docs/plan.md Phase 3 acceptance: "seed 寫死
# 在 config 並記錄在結果文件；順帶記錄 cell 清單" -- the cell lists
# themselves are printed/recorded by scripts/phase3_validate.py).
# ---------------------------------------------------------------------------
CELL_SPLIT_SEED = 20260726  # 80/20 train/validation cell split
DEV_SPLIT_SEED = 20260727   # secondary split of the 80 train cells, used
                             # only to carve an early-stopping dev set that
                             # never touches the 20% validation cells
TRAIN_CELL_FRAC = 0.8
DEV_TRAIN_FRAC = 0.8        # of the 80 train cells: 64 dev-train / 16 dev-early-stop

SCORE_THRESHOLD = 98.0  # docs/plan.md Phase 3 user acceptance gate

EPS = 1e-30  # floor for log(|x|); real table values never go below ~9e-8
             # in this dataset (verified against testcase/alpha_test/full), so this
             # only ever engages on the anchor==0 known-invalid rows,
             # which are excluded from training and overridden to 0 at
             # reconstruction time regardless of the model's output.

# ---------------------------------------------------------------------------
# Categorical vocabularies (fixed, not fit from data -- these are just an
# encoding convention, not model parameters, so listing every value
# observed anywhere in the dataset -- including validation cells -- is not
# a leak: it carries no information about any particular cell's timing
# values, only about which categorical labels *exist* in the Liberty
# grammar/library, which is public before any split is made).
# ---------------------------------------------------------------------------
TIMING_SENSE_VALUES = ("positive_unate", "negative_unate", "non_unate", "na")
TIMING_TYPE_VALUES = ("combinational", "rising_edge", "falling_edge", "preset", "clear", "na")
TABLE_TYPE_VALUES = TABLE_KINDS  # cell_rise, cell_fall, rise_transition, fall_transition, rise_power, fall_power
PROCESS_VALUES = ("ff", "tt", "ss")

NUMERIC_FEATURE_NAMES = [
    "log_anchor",
    "log_phase25_pred",
    "V_source",
    "V_target",
    "dV",
    "T_source",
    "T_target",
    "dT",
    "slew_idx_norm",
    "load_idx_norm",
    "log_slew",
    "log_load",
    "log_drive_strength",
    "family_code",
    "same_process",
]

FEATURE_NAMES: List[str] = (
    NUMERIC_FEATURE_NAMES
    + [f"proc_src_{p}" for p in PROCESS_VALUES]
    + [f"proc_tgt_{p}" for p in PROCESS_VALUES]
    + [f"sense_{s}" for s in TIMING_SENSE_VALUES]
    + [f"ttype_{t}" for t in TIMING_TYPE_VALUES]
    + [f"table_{t}" for t in TABLE_TYPE_VALUES]
)


def split_cells(
    all_cell_names: Iterable[str], *, seed: int = CELL_SPLIT_SEED, train_frac: float = TRAIN_CELL_FRAC
) -> Tuple[List[str], List[str]]:
    """Deterministic 80/20 cell split. Sorts names first (dict/set
    iteration order is not guaranteed) so the same `seed` always yields
    the same split regardless of caller-side ordering."""
    names = sorted(all_cell_names)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(names))
    n_train = int(round(len(names) * train_frac))
    train = sorted(names[i] for i in perm[:n_train])
    val = sorted(names[i] for i in perm[n_train:])
    return train, val


def split_dev(
    train_cells: Sequence[str], *, seed: int = DEV_SPLIT_SEED, dev_train_frac: float = DEV_TRAIN_FRAC
) -> Tuple[List[str], List[str]]:
    """Split the 80 training cells again (never touching the 20%
    validation cells) into a dev-train subset and an early-stopping dev
    subset, per docs/plan.md Phase 3: "防過擬合：早停用訓練 cell 內部再切
    的 dev 子集，絕不能碰 20% 驗證 cell"."""
    return split_cells(train_cells, seed=seed, train_frac=dev_train_frac)


def ordered_full_corner_pairs(full_metas: Iterable[CornerMeta]) -> List[Tuple[CornerMeta, CornerMeta]]:
    """All 5*4=20 ordered (source, target) pairs among the full corners,
    source != target, in a deterministic order."""
    metas = sorted(full_metas, key=lambda m: m.name)
    return [(s, t) for s in metas for t in metas if s.name != t.name]


def filter_lib_cells(lib: LibertyFile, keep_cells: Sequence[str]) -> LibertyFile:
    """Shadow LibertyFile exposing only `keep_cells` (same pattern as
    scoring.ensemble._filter_lib_cells, duplicated here rather than
    imported since it's a small module-private helper in that module and
    this is a distinct use site with its own docstring/context)."""
    keep = set(keep_cells)
    cells = {name: c for name, c in lib.cells.items() if name in keep}
    tables = [t for t in lib.tables if t.key[0] in keep]
    tables_by_key = {k: v for k, v in lib.tables_by_key.items() if k[0] in keep}
    return LibertyFile(
        path=lib.path, text=lib.text, library_name=lib.library_name,
        cells=cells, tables=tables, tables_by_key=tables_by_key,
    )


def build_arc_attr_index(lib: LibertyFile) -> Dict[tuple, Tuple[str, str]]:
    """Map (cell, pin, group_type, arc_index) -> (timing_sense, timing_type),
    with `None` normalized to the "na" vocabulary entry. Arc structure
    (which arcs exist, their timing_sense/timing_type) is a technology
    characteristic shared by every PVT corner -- Phase 1/2 confirmed all
    15 released .lib files share the same 5804 table keys -- so this only
    needs to be built once from any single full corner and reused for
    every corner pair."""
    idx: Dict[tuple, Tuple[str, str]] = {}
    for cell in lib.cells.values():
        for pin in cell.pins.values():
            for arc in pin.arcs:
                idx[(cell.name, pin.name, arc.group_type, arc.arc_index)] = (
                    arc.timing_sense if arc.timing_sense in TIMING_SENSE_VALUES else "na",
                    arc.timing_type if arc.timing_type in TIMING_TYPE_VALUES else "na",
                )
    return idx


def build_family_vocab(cell_names: Iterable[str]) -> Dict[str, int]:
    """Fixed ordinal code per cell "function family" (features.cellinfo),
    e.g. AN2AM16 -> family "AN". Built once from the *full* 100-cell set
    (not per-split) and reused for every train/dev/val feature build so
    train and validation rows share one consistent encoding -- this is a
    vocabulary, not a fitted statistic, so building it from all 100 cells
    (including the 20 validation ones) carries no information leak about
    any cell's timing values (see module docstring)."""
    families = sorted({parse_cell_name(name).family for name in cell_names})
    return {f: i for i, f in enumerate(families)}


def _onehot_row(value: str, vocab: Sequence[str]) -> np.ndarray:
    row = np.zeros(len(vocab), dtype=np.float32)
    row[vocab.index(value)] = 1.0
    return row


@dataclass
class Phase3Dataset:
    """Flat, point-level (one row per (source, target, table key, grid
    point) sample) feature matrix plus the raw quantities needed to build
    training labels under either target-definition mode and to
    reconstruct + score predictions afterwards."""

    X: np.ndarray             # (n, len(FEATURE_NAMES)) float32
    anchor: np.ndarray        # (n,) float64 -- source (input) value
    target: np.ndarray        # (n,) float64 -- true target-corner value (label source)
    phase25_pred: np.ndarray  # (n,) float64 -- Phase 2.5 physical model's own prediction
    table_type: np.ndarray    # (n,) <U16
    pair: np.ndarray          # (n,) <U32, "source_name->target_name"
    cell: np.ndarray          # (n,) <U16
    keys: list                # (n,) list[TableKey], parallel to the rows above

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def trainable_mask(self) -> np.ndarray:
        """Rows with a nonzero anchor -- the only rows carrying a defined
        log-ratio label. Zero-anchor rows (docs/plan.md rule 3:
        known-invalid rise_power/fall_power arcs) always reconstruct to
        an exact 0 prediction regardless of any model's output, so they
        are excluded from training but *not* from validation scoring
        (the aggregate validation score must cover every point, matching
        the real scorer's domain)."""
        return self.anchor != 0.0


def build_feature_matrix(
    libs: Mapping[CornerMeta, LibertyFile],
    cells: Sequence[str],
    pairs: Sequence[Tuple[CornerMeta, CornerMeta]],
    phase25_model: Phase2Model,
    arc_attr_index: Mapping[tuple, Tuple[str, str]],
    family_vocab: Mapping[str, int],
    *,
    use_process_offset: bool = True,
) -> Phase3Dataset:
    """Build one flat Phase3Dataset from every (source, target, table,
    grid point) combination where `key[0]` (the cell name) is in `cells`
    and both corners of the pair have a non-blank table at that key.

    `phase25_model` supplies both the "Phase 2.5 physical prediction"
    feature (docs/plan.md Phase 3 feature list) and, when this function
    is called with the full validation cell set, the Phase 2.5 baseline
    score under the same protocol -- see scripts/phase3_validate.py.
    `use_process_offset=True` matches scoring.loco.run_loco's convention
    for cross-corner transfers (most of the 20 ordered pairs here are
    cross-process, exactly the case the offset term exists for); see
    models.phase2_scaling's module docstring.
    """
    cell_set = set(cells)
    proc_onehot = {p: _onehot_row(p, PROCESS_VALUES) for p in PROCESS_VALUES}
    sense_onehot = {s: _onehot_row(s, TIMING_SENSE_VALUES) for s in TIMING_SENSE_VALUES}
    ttype_onehot = {t: _onehot_row(t, TIMING_TYPE_VALUES) for t in TIMING_TYPE_VALUES}
    table_onehot = {t: _onehot_row(t, TABLE_TYPE_VALUES) for t in TABLE_TYPE_VALUES}

    row_idx, col_idx = np.indices((7, 7))
    slew_idx_norm_grid = ((row_idx - 3.0) / 3.0).ravel().astype(np.float32)
    load_idx_norm_grid = ((col_idx - 3.0) / 3.0).ravel().astype(np.float32)

    X_chunks: List[np.ndarray] = []
    anchor_chunks: List[np.ndarray] = []
    target_chunks: List[np.ndarray] = []
    phase25_chunks: List[np.ndarray] = []
    table_type_chunks: List[np.ndarray] = []
    pair_chunks: List[np.ndarray] = []
    cell_chunks: List[np.ndarray] = []
    keys: list = []

    for source_meta, target_meta in pairs:
        lib_s = libs[source_meta]
        lib_t = libs[target_meta]
        pair_label = f"{source_meta.name}->{target_meta.name}"
        proc_src_row = proc_onehot[source_meta.process]
        proc_tgt_row = proc_onehot[target_meta.process]
        same_process = 1.0 if source_meta.process == target_meta.process else 0.0

        for key, ts in lib_s.tables_by_key.items():
            cell_name = key[0]
            if cell_name not in cell_set or ts.values is None:
                continue
            tt = lib_t.tables_by_key.get(key)
            if tt is None or tt.values is None:
                continue

            table_type = key[-1]
            arc_key = key[:-1]
            sense, ttype = arc_attr_index.get(arc_key, ("na", "na"))
            info = parse_cell_name(cell_name)
            family_code = float(family_vocab.get(info.family, -1))
            log_strength = float(np.log(info.drive_strength))

            phase25_vals = phase25_model.predict_table(
                key, ts.values, source_meta, target_meta, ts.index_2, tt.index_2,
                use_process_offset=use_process_offset,
            )

            anchor_flat = ts.values.ravel()
            target_flat = tt.values.ravel()
            phase25_flat = phase25_vals.ravel()
            n = anchor_flat.size  # 49

            log_slew_grid = np.log(np.asarray(tt.index_1, dtype=float))[row_idx].ravel()
            log_load_grid = np.log(np.asarray(tt.index_2, dtype=float))[col_idx].ravel()

            numeric = np.empty((n, len(NUMERIC_FEATURE_NAMES)), dtype=np.float32)
            numeric[:, 0] = np.log(np.abs(anchor_flat) + EPS)
            numeric[:, 1] = np.log(np.abs(phase25_flat) + EPS)
            numeric[:, 2] = source_meta.voltage
            numeric[:, 3] = target_meta.voltage
            numeric[:, 4] = target_meta.voltage - source_meta.voltage
            numeric[:, 5] = source_meta.temperature
            numeric[:, 6] = target_meta.temperature
            numeric[:, 7] = target_meta.temperature - source_meta.temperature
            numeric[:, 8] = slew_idx_norm_grid
            numeric[:, 9] = load_idx_norm_grid
            numeric[:, 10] = log_slew_grid
            numeric[:, 11] = log_load_grid
            numeric[:, 12] = log_strength
            numeric[:, 13] = family_code
            numeric[:, 14] = same_process

            cat = np.tile(
                np.concatenate([proc_src_row, proc_tgt_row, sense_onehot[sense], ttype_onehot[ttype], table_onehot[table_type]]),
                (n, 1),
            ).astype(np.float32)

            X_chunks.append(np.hstack([numeric, cat]))
            anchor_chunks.append(anchor_flat)
            target_chunks.append(target_flat)
            phase25_chunks.append(phase25_flat)
            table_type_chunks.append(np.full(n, table_type))
            pair_chunks.append(np.full(n, pair_label))
            cell_chunks.append(np.full(n, cell_name))
            keys.extend([key] * n)

    if not X_chunks:
        raise ValueError("no samples produced -- empty cells/pairs?")

    return Phase3Dataset(
        X=np.concatenate(X_chunks, axis=0),
        anchor=np.concatenate(anchor_chunks),
        target=np.concatenate(target_chunks),
        phase25_pred=np.concatenate(phase25_chunks),
        table_type=np.concatenate(table_type_chunks),
        pair=np.concatenate(pair_chunks),
        cell=np.concatenate(cell_chunks),
        keys=keys,
    )


def make_label(ds: Phase3Dataset, mode: str) -> np.ndarray:
    """Training label. `mode == "raw"`: log(|target| / |anchor|).
    `mode == "residual"`: log(|target| / |phase25_pred|), i.e. the
    log-ratio *on top of* the Phase 2.5 physical prediction ("讓 ML 學殘差
    修正的效果", docs/plan.md). Both are well-defined wherever
    `ds.trainable_mask` holds (anchor != 0, which also implies
    phase25_pred != 0 -- Phase2Model.predict_table maps a zero source to
    an exact zero prediction, see models.phase2_scaling)."""
    if mode == "raw":
        base = ds.anchor
    elif mode == "residual":
        base = ds.phase25_pred
    else:
        raise ValueError(f"unknown label mode {mode!r}, expected 'raw' or 'residual'")
    ratio = np.abs(ds.target) / np.maximum(np.abs(base), EPS)
    # Rows with target == 0 (only possible where anchor == 0 too, docs/plan.md
    # rule 3) give ratio == 0 -> log(0) == -inf; this is expected and benign,
    # those rows are never used for training or reconstruction (see
    # `trainable_mask` / `reconstruct_predictions`'s anchor==0 override), so
    # the divide-by-zero warning is suppressed rather than left to alarm
    # callers about a case that is already handled.
    with np.errstate(divide="ignore"):
        return np.log(ratio)


def score_breakdown(y_true: np.ndarray, y_pred: np.ndarray, groups: Mapping[str, np.ndarray]):
    """Run scoring.scorer's point_errors/score_from_errors over
    `(y_true, y_pred)` and additionally break the result down by every
    named grouping array in `groups` (e.g. {"table_type": ds.table_type,
    "pair": ds.pair}). Returns `(overall_score, n_points, {group_name:
    {group_value: (score, n_points)}})`."""
    from scoring.scorer import point_errors, score_from_errors

    errs = point_errors(y_true, y_pred)
    overall = score_from_errors(errs)
    breakdowns: Dict[str, Dict[str, Tuple[float, int]]] = {}
    for group_name, group_values in groups.items():
        per_value: Dict[str, Tuple[float, int]] = {}
        for value in sorted(set(group_values.tolist())):
            mask = group_values == value
            per_value[value] = (score_from_errors(errs[mask]), int(mask.sum()))
        breakdowns[group_name] = per_value
    return float(overall), int(errs.size), breakdowns


def reconstruct_predictions(
    ds: Phase3Dataset, y_pred: np.ndarray, mode: str, *, clip: float = 20.0
) -> np.ndarray:
    """Invert `make_label`: turn a model's predicted log-ratio back into
    a value on the original scale, preserving the anchor's sign (same
    multiplicative convention as models.phase2_scaling.Phase2Model --
    `predicted = base * exp(y_pred)` never flips sign). `clip` bounds the
    exponent so a wild model output cannot overflow to inf (writer.py's
    NaN/Inf guard would reject that anyway; clipping here keeps the
    reconstructed value merely very large/small, which the scorer's error
    cap already treats as a saturated failed point). Zero-anchor rows are
    forced to exactly 0 regardless of `y_pred`, per docs/plan.md rule 3."""
    if mode == "raw":
        base = ds.anchor
    elif mode == "residual":
        base = ds.phase25_pred
    else:
        raise ValueError(f"unknown label mode {mode!r}, expected 'raw' or 'residual'")
    y_pred = np.clip(np.asarray(y_pred, dtype=float), -clip, clip)
    pred = base * np.exp(y_pred)
    return np.where(ds.anchor == 0.0, 0.0, pred)
