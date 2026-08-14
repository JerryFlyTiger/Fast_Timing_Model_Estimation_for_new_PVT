"""Phase 4 dataset construction: same-corner cell-generalization pipeline
built on the official 400-cell training set (docs/plan.md section 3 /
Phase 4, 2026-07-26).

Task-structure change from Phase 2/3
-------------------------------------
`testcase/training_set/base_nom_{0p8v,0p9v,1p0v}/` supplies all 400
training cells at *every one* of the 15 released corners -- the same 5
"anchor" (standard-voltage) corners `testcase/alpha_test/full/` has for
the 100 alpha cells, plus the 10 "delivery" corners
`testcase/alpha_test/partial/` needs predictions for. So for every
delivery corner, real ground truth exists for 400 cells that are *not*
the alpha 100 -- this is now a same-corner, unseen-*cell* generalization
problem, not the Phase 2/3 cross-corner-pair transfer problem (see
docs/plan.md section 3's "任務結構自此改變").

Model shape
-----------
One model per delivery corner (10 total, x2 for GBDT/MLP), each trained
on point-level rows built from the 400 training cells' 5 anchor tables
(never the target table -- the target table is *only* ever the label).
The label is the log-ratio against each target corner's "nearest anchor"
-- the one anchor that shares the target's process *and* temperature, so
the two differ *only* in voltage (`NEAREST_ANCHOR_BY_TARGET` below).
Every one of the 10 real delivery corners has exactly one such anchor
(docs/phase2_results.md's finding, unchanged from Phase 2), so this
mirrors the real inference input shape exactly: predict a pure ~+-10%VDD
shift from the alpha cell's own standard-voltage anchor tables.

Features (docs/plan.md Phase 4 item 3): log of all 5 anchor corners'
values at the same (cell, pin, arc, table, grid-point) key -- not just
the nearest one, so the model can see the *shape* across process/voltage
if that helps -- plus grid coordinates, cell type/drive strength, arc
attributes (timing_sense/timing_type), and table_type. This is exactly
the information available at alpha inference time (100 alpha cells only
have the 5 standard-voltage corners populated).

`build_base_dataset` is deliberately agnostic to which corner is being
predicted (it never touches a target-corner lib) -- it produces one flat
feature matrix per split (train/val, or the full alpha 100 cells for
inference) that is reused for all 10 delivery-corner models, since the
row set (cell, pin, arc, table, grid-point) is identical for every
target given all 15 training corners share exactly the same table-key
set per cell (verified, see docs/phase4_results.md "parser 覆核"). Only
the per-corner label (`make_label` against `extract_raw_values(target_lib,
...)`) and the corresponding "nearest anchor" column differ per model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from features.cellinfo import parse_cell_name
from features.corners import parse_corner_filename
from liberty.parser import TABLE_KINDS, LibertyFile, TableKey
from models.phase3_features import (
    build_arc_attr_index,
    build_family_vocab,
    split_cells,
    split_dev,
)

# ---------------------------------------------------------------------------
# Corner topology (docs/plan.md Phase 4 item 3 / section 3 table).
#
# 2026-07-27 beta/final-stage simulation addition: the contest may reveal
# a *different* PVT triplet as the "known" (anchor) set at later judging
# stages -- see docs/plan.md improvement round / docs/phase4_results.md
# "beta 階段模擬". `CornerTopology` below generalizes the alpha-stage
# constants (kept as-is, unchanged, for 100% backward compatibility with
# every existing script/test) into a parametrizable triple
# (anchor_names, target_names, nearest_anchor_by_target) that
# `build_base_dataset`/`Phase4Dataset.nearest_anchor` can accept
# explicitly instead of only ever reading the alpha-stage module globals.
# ---------------------------------------------------------------------------

# The 5 standard-voltage "anchor" corners -- fixed column order for the
# "log_anchor_*" features below. Sorted so the order is deterministic
# regardless of dict/glob iteration order. This is the ALPHA-stage anchor
# set (the current real contest input shape) -- kept as the module
# default everywhere below.
ANCHOR_CORNER_NAMES: Tuple[str, ...] = tuple(
    sorted(("ff0p99v125c", "ff0p99vm40c", "ss0p81v125c", "ss0p81vm40c", "tt0p9v25c"))
)

# Each of the 10 delivery corners -> the one anchor sharing its process
# *and* temperature (a pure voltage-only shift, Delta_T == 0 -- the same
# "every real target has a same-process-same-temperature anchor" fact
# Phase 2/2.5/3 all relied on, docs/phase2_results.md).
NEAREST_ANCHOR_BY_TARGET: Dict[str, str] = {
    "ff0p88v125c": "ff0p99v125c",
    "ff0p88vm40c": "ff0p99vm40c",
    "ff1p1v125c": "ff0p99v125c",
    "ff1p1vm40c": "ff0p99vm40c",
    "ss0p72v125c": "ss0p81v125c",
    "ss0p72vm40c": "ss0p81vm40c",
    "ss0p9v125c": "ss0p81v125c",
    "ss0p9vm40c": "ss0p81vm40c",
    "tt0p8v25c": "tt0p9v25c",
    "tt1p0v25c": "tt0p9v25c",
}
DELIVERY_CORNER_NAMES: Tuple[str, ...] = tuple(sorted(NEAREST_ANCHOR_BY_TARGET))

assert set(NEAREST_ANCHOR_BY_TARGET.values()) <= set(ANCHOR_CORNER_NAMES)


def _meta(corner_name: str):
    """Parse a bare corner short-name (e.g. 'ff1p1v125c') into its
    CornerMeta (process/voltage/temperature) by wrapping it in the
    filename convention `features.corners.parse_corner_filename` expects
    -- same trick `tests/test_phase4_features.py` already uses."""
    return parse_corner_filename(f"lib1_{corner_name}_alpha_100.lib")


def infer_nearest_anchor_by_target(
    anchor_names: Sequence[str], target_names: Sequence[str]
) -> Dict[str, str]:
    """Generalizes `NEAREST_ANCHOR_BY_TARGET`'s hand-written alpha-stage
    mapping to an arbitrary (anchor_names, target_names) pair: for each
    target, find the anchor sharing its process *and* temperature
    (asserted unique -- fails loudly if a stage's corner sets don't have
    the expected one-to-one same-process-same-temperature pairing, rather
    than silently picking an arbitrary candidate)."""
    anchor_metas = {name: _meta(name) for name in anchor_names}
    mapping: Dict[str, str] = {}
    for target in target_names:
        tmeta = _meta(target)
        candidates = [
            name for name, ameta in anchor_metas.items()
            if ameta.process == tmeta.process and ameta.temperature == tmeta.temperature
        ]
        assert len(candidates) == 1, (
            f"expected exactly one same-process-same-temperature anchor for {target!r} "
            f"among {anchor_names}, found {candidates}"
        )
        mapping[target] = candidates[0]
    return mapping


@dataclass(frozen=True)
class CornerTopology:
    """One (anchor set, target set, nearest-anchor mapping) triple --
    what changes between the alpha/beta/final contest stages. Everything
    downstream (`build_base_dataset`, `Phase4Dataset.nearest_anchor`)
    takes these as explicit parameters (defaulting to the alpha-stage
    globals for backward compatibility) rather than reading a hardcoded
    module global, so a different stage's topology never silently falls
    back to alpha-stage corner names."""

    name: str
    anchor_names: Tuple[str, ...]
    target_names: Tuple[str, ...]
    nearest_anchor_by_target: Dict[str, str]

    def __post_init__(self) -> None:
        assert len(self.anchor_names) == 5, f"{self.name}: expected 5 anchors, got {len(self.anchor_names)}"
        assert len(self.target_names) == 10, f"{self.name}: expected 10 targets, got {len(self.target_names)}"
        assert not (set(self.anchor_names) & set(self.target_names)), (
            f"{self.name}: anchor/target corner sets overlap: "
            f"{set(self.anchor_names) & set(self.target_names)}"
        )
        assert set(self.nearest_anchor_by_target) == set(self.target_names)
        assert set(self.nearest_anchor_by_target.values()) <= set(self.anchor_names)


ALPHA_TOPOLOGY = CornerTopology(
    "alpha", ANCHOR_CORNER_NAMES, DELIVERY_CORNER_NAMES, NEAREST_ANCHOR_BY_TARGET
)

# Beta stage (docs/phase4_results.md "beta 階段模擬"): the "boost" (升壓)
# corners are known; targets are the standard-voltage 5 + the "buck"
# (降壓) 5. The buck targets sit *two* voltage steps from their nearest
# anchor (boost -> nominal -> buck), but the nearest-anchor rule is
# unchanged: same process + same temperature, which is still always the
# boost anchor (there is no nominal-voltage anchor available in this
# stage) -- `infer_nearest_anchor_by_target` derives this automatically.
BETA_ANCHOR_NAMES: Tuple[str, ...] = tuple(
    sorted(("ss0p9v125c", "ss0p9vm40c", "ff1p1v125c", "ff1p1vm40c", "tt1p0v25c"))
)
BETA_TARGET_NAMES: Tuple[str, ...] = tuple(
    sorted((
        "ss0p81v125c", "ss0p81vm40c", "ff0p99v125c", "ff0p99vm40c", "tt0p9v25c",
        "ss0p72v125c", "ss0p72vm40c", "ff0p88v125c", "ff0p88vm40c", "tt0p8v25c",
    ))
)
BETA_TOPOLOGY = CornerTopology(
    "beta", BETA_ANCHOR_NAMES, BETA_TARGET_NAMES,
    infer_nearest_anchor_by_target(BETA_ANCHOR_NAMES, BETA_TARGET_NAMES),
)

# Final stage: the "buck" (降壓) corners are known; targets are the boost
# 5 + the standard-voltage 5.
FINAL_ANCHOR_NAMES: Tuple[str, ...] = tuple(
    sorted(("ss0p72v125c", "ss0p72vm40c", "ff0p88v125c", "ff0p88vm40c", "tt0p8v25c"))
)
FINAL_TARGET_NAMES: Tuple[str, ...] = tuple(
    sorted((
        "ss0p9v125c", "ss0p9vm40c", "ff1p1v125c", "ff1p1vm40c", "tt1p0v25c",
        "ss0p81v125c", "ss0p81vm40c", "ff0p99v125c", "ff0p99vm40c", "tt0p9v25c",
    ))
)
FINAL_TOPOLOGY = CornerTopology(
    "final", FINAL_ANCHOR_NAMES, FINAL_TARGET_NAMES,
    infer_nearest_anchor_by_target(FINAL_ANCHOR_NAMES, FINAL_TARGET_NAMES),
)

STAGE_TOPOLOGIES: Dict[str, CornerTopology] = {
    "alpha": ALPHA_TOPOLOGY,
    "beta": BETA_TOPOLOGY,
    "final": FINAL_TOPOLOGY,
}

# Consistency check: the generic inference function must reproduce the
# hand-written alpha-stage mapping exactly (protects against the two
# ever silently diverging).
assert infer_nearest_anchor_by_target(ANCHOR_CORNER_NAMES, DELIVERY_CORNER_NAMES) == NEAREST_ANCHOR_BY_TARGET

# ---------------------------------------------------------------------------
# Fixed seeds (docs/plan.md Phase 4 acceptance: "記錄 seed 與名單").
# Distinct constants from models.phase3_features's (different cell
# population: 400 official training cells, not the alpha 100), but the
# same "encode today's date" convention.
# ---------------------------------------------------------------------------
PHASE4_CELL_SPLIT_SEED = 20260726  # 400 -> 320 train / 80 validation cells
PHASE4_DEV_SPLIT_SEED = 20260727   # 320 train cells -> 256 dev-train / 64 dev-val
                                    # (early stopping only; never touches the 80)
TRAIN_CELL_FRAC = 0.8
DEV_TRAIN_FRAC = 0.8

SCORE_THRESHOLD = 98.0  # docs/plan.md Phase 4 user acceptance gate

EPS = 1e-30  # floor for log(|x|) -- only engages on the known-invalid
             # all-zero rise_power/fall_power rows (docs/plan.md rule 3),
             # which are excluded from training via `trainable_mask` and
             # forced to exactly 0 at reconstruction regardless of any
             # model output.

CLIP_LOG_RATIO = 20.0  # bound on the model's predicted log-ratio before
                        # exponentiating back to linear scale, so a wild
                        # model output cannot overflow to inf (matches
                        # models.phase3_features.reconstruct_predictions's
                        # convention).

# ---------------------------------------------------------------------------
# Categorical vocabularies (fixed, not fit from data -- see
# models.phase3_features's identical rationale: these just enumerate
# labels the Liberty grammar can contain, not a fitted statistic).
# ---------------------------------------------------------------------------
TIMING_SENSE_VALUES = ("positive_unate", "negative_unate", "non_unate", "na")
TIMING_TYPE_VALUES = ("combinational", "rising_edge", "falling_edge", "preset", "clear", "na")
TABLE_TYPE_VALUES = TABLE_KINDS


# ---------------------------------------------------------------------------
# Cell response-signature features (2026-07-27 Phase 4 improvement round,
# docs/phase4_results.md lever 1): per-cell, per-grid-point *measured*
# sensitivities derived from the 5 anchor tables themselves -- log-ratios
# between anchor pairs that isolate a pure temperature shift or a pure
# process shift. These carry the cell's own empirical response shape
# (much more informative than a categorical cell-family code), and are
# exact linear combinations of the log_anchor_* columns -- cheap for an
# MLP's first linear layer to reconstruct, but genuinely new information
# for a GBDT, which can only approximate a difference of two columns via
# multiple tree splits otherwise.
#
# 2026-07-27 beta/final-stage addition: which literal corner plays the
# "ff_hot"/"ff_cold"/"ss_hot"/"ss_cold"/"tt_mid" role changes per stage
# (e.g. beta's "ff_hot" is ff1p1v125c, not alpha's ff0p99v125c) -- so
# these roles are resolved *dynamically* per `anchor_names` sequence by
# (process, temperature), NEVER by matching a literal alpha-stage corner
# name. This is the one place a stage bug (accidentally reading an
# alpha-specific name/value while running beta) would most easily creep
# in, so `_resolve_anchor_roles` is the single choke point for it and is
# covered by `tests/test_phase4_features.py`'s stage-topology tests.
# ---------------------------------------------------------------------------


def _resolve_anchor_roles(anchor_names: Sequence[str]) -> Dict[str, int]:
    """Map role name ('ff_hot'/'ff_cold'/'ss_hot'/'ss_cold'/'tt_mid') to
    its column index within `anchor_names`, resolved purely from each
    anchor's (process, temperature) -- works for any 5-anchor stage
    topology (alpha/beta/final), never falls back to a hardcoded name."""
    roles: Dict[str, int] = {}
    for i, name in enumerate(anchor_names):
        meta = _meta(name)
        if meta.process == "ff" and meta.temperature == 125.0:
            roles["ff_hot"] = i
        elif meta.process == "ff" and meta.temperature == -40.0:
            roles["ff_cold"] = i
        elif meta.process == "ss" and meta.temperature == 125.0:
            roles["ss_hot"] = i
        elif meta.process == "ss" and meta.temperature == -40.0:
            roles["ss_cold"] = i
        elif meta.process == "tt":
            roles["tt_mid"] = i
    expected = {"ff_hot", "ff_cold", "ss_hot", "ss_cold", "tt_mid"}
    missing = expected - set(roles)
    assert not missing, f"anchor set {anchor_names} is missing role(s) {missing} -- not a valid 5-anchor topology"
    return roles


SENSITIVITY_FEATURE_NAMES: List[str] = [
    "log_ratio_ff_hot_cold",  # temperature sensitivity, ff process
    "log_ratio_ss_hot_cold",  # temperature sensitivity, ss process
    "log_ratio_ss_ff_hot",    # process sensitivity at 125C
    "log_ratio_ss_ff_cold",   # process sensitivity at -40C
    "log_ratio_tt_ss_hot",    # tt0p9v25c located relative to ss0p81v125c
    "log_ratio_tt_ff_hot",    # tt0p9v25c located relative to ff0p99v125c
]

# Spatial-context features (lever 2): within-table row/col gradients of
# each anchor's log value (np.gradient, edge-clamped one-sided at the
# table boundary) -- lets the model see the local surface shape around
# each grid point, not just its own value.
#
# NOTE these *labels* are fixed to the alpha-stage anchor names for
# documentation purposes only (FEATURE_NAMES's length/shape never
# changes across stages -- always 5 anchors -> the same column count).
# The actual VALUES `build_base_dataset` computes always follow whichever
# `anchor_names` topology is passed in at call time (see
# `_resolve_anchor_roles`) -- column i is "whatever anchor is in
# position i of that call's anchor_names", not literally tied to the
# alpha corner name printed here.
GRADIENT_FEATURE_NAMES: List[str] = [f"log_grad_row_{name}" for name in ANCHOR_CORNER_NAMES] + [
    f"log_grad_col_{name}" for name in ANCHOR_CORNER_NAMES
]

NUMERIC_FEATURE_NAMES: List[str] = (
    [f"log_anchor_{name}" for name in ANCHOR_CORNER_NAMES]
    + SENSITIVITY_FEATURE_NAMES
    + GRADIENT_FEATURE_NAMES
    + [
        "slew_idx_norm",
        "load_idx_norm",
        "log_slew",
        "log_load",
        "log_drive_strength",
        "family_code",
    ]
)

FEATURE_NAMES: List[str] = (
    NUMERIC_FEATURE_NAMES
    + [f"sense_{s}" for s in TIMING_SENSE_VALUES]
    + [f"ttype_{t}" for t in TIMING_TYPE_VALUES]
    + [f"table_{t}" for t in TABLE_TYPE_VALUES]
)


def _onehot_row(value: str, vocab: Sequence[str]) -> np.ndarray:
    row = np.zeros(len(vocab), dtype=np.float32)
    row[vocab.index(value)] = 1.0
    return row


@dataclass
class Phase4Dataset:
    """Flat, point-level (one row per (cell, pin, arc, table, grid point)
    sample) feature matrix, corner-agnostic: it never reads a target-
    corner value, so it is built once per cell split and reused for all
    10 delivery-corner models (see module docstring)."""

    X: np.ndarray               # (n, len(FEATURE_NAMES)) float32
    anchor_values: np.ndarray   # (n, len(anchor_names)) float64, raw (linear-space) anchor values
    table_type: np.ndarray      # (n,) <U16
    cell: np.ndarray            # (n,) <U16
    keys: list                  # (n,) list[TableKey], parallel to the rows above; each key repeats
                                 # in a contiguous 49-row block (7x7 raveled)
    anchor_names: Tuple[str, ...] = ANCHOR_CORNER_NAMES  # this dataset's actual anchor-column order
                                 # (defaults to the alpha-stage set for backward compatibility with
                                 # every pre-2026-07-27 call site that never passed anchor_names)

    @property
    def n(self) -> int:
        return self.X.shape[0]

    def nearest_anchor(
        self, target_corner: str, nearest_anchor_by_target: Optional[Mapping[str, str]] = None
    ) -> np.ndarray:
        """This dataset's raw anchor values for the one anchor matching
        `target_corner`'s process+temperature (docs module docstring).
        `nearest_anchor_by_target` defaults to the alpha-stage
        `NEAREST_ANCHOR_BY_TARGET` global (backward compatible with every
        existing call site); pass a beta/final-stage
        `CornerTopology.nearest_anchor_by_target` explicitly for those
        stages -- this dataset's own `anchor_names` (not the alpha-stage
        global) is always used to resolve the column index, so this is
        correct for whichever topology `build_base_dataset` was actually
        called with."""
        mapping = NEAREST_ANCHOR_BY_TARGET if nearest_anchor_by_target is None else nearest_anchor_by_target
        col = self.anchor_names.index(mapping[target_corner])
        return self.anchor_values[:, col]


def build_family_vocab_for_phase4(*cell_name_groups: Sequence[str]) -> Dict[str, int]:
    """Build one fixed family-code vocabulary covering every cell name in
    every group passed in (docs/plan.md: vocabularies built from the
    union of all cell names -- training and alpha -- are not a leak, see
    models.phase3_features.build_family_vocab's identical rationale).
    Callers pass both the 400 training cell names and the 100 alpha cell
    names so a single consistent encoding serves training and inference."""
    all_names: List[str] = []
    for group in cell_name_groups:
        all_names.extend(group)
    return build_family_vocab(all_names)


def build_base_dataset(
    anchor_libs: Mapping[str, LibertyFile],
    cells: Sequence[str],
    arc_attr_index: Mapping[tuple, Tuple[str, str]],
    family_vocab: Mapping[str, int],
    anchor_names: Sequence[str] = ANCHOR_CORNER_NAMES,
) -> Phase4Dataset:
    """Build one Phase4Dataset from every (cell, pin, arc, table, grid
    point) key where `key[0]` (the cell name) is in `cells` and all 5
    `anchor_libs` (keyed by corner short name -- must contain every name
    in `anchor_names`) have a non-blank table at that key. Never touches
    any delivery/target-corner lib -- `anchor_libs` only ever needs to
    contain the 5 `anchor_names` keys, nothing else, which is exactly
    what makes this safe to call with a dict that structurally *cannot*
    contain a target corner's values (see module docstring and the
    2026-07-27 beta/final-stage zero-leakage tests).

    `anchor_names` defaults to the alpha-stage `ANCHOR_CORNER_NAMES` for
    backward compatibility with every pre-2026-07-27 call site; pass a
    `CornerTopology.anchor_names` explicitly (e.g. `BETA_ANCHOR_NAMES`)
    to build a beta/final-stage feature matrix instead -- the response-
    signature features (lever 1) resolve their ff/ss/tt hot/cold roles
    dynamically from whichever `anchor_names` is passed in
    (`_resolve_anchor_roles`), never from a hardcoded alpha-stage name.
    """
    anchor_names = tuple(anchor_names)
    cell_set = set(cells)
    sense_onehot = {s: _onehot_row(s, TIMING_SENSE_VALUES) for s in TIMING_SENSE_VALUES}
    ttype_onehot = {t: _onehot_row(t, TIMING_TYPE_VALUES) for t in TIMING_TYPE_VALUES}
    table_onehot = {t: _onehot_row(t, TABLE_TYPE_VALUES) for t in TABLE_TYPE_VALUES}

    row_idx, col_idx = np.indices((7, 7))
    slew_idx_norm_grid = ((row_idx - 3.0) / 3.0).ravel().astype(np.float32)
    load_idx_norm_grid = ((col_idx - 3.0) / 3.0).ravel().astype(np.float32)

    master_lib = anchor_libs[anchor_names[0]]
    n_anchors = len(anchor_names)
    roles = _resolve_anchor_roles(anchor_names)
    ff_hot, ff_cold = roles["ff_hot"], roles["ff_cold"]
    ss_hot, ss_cold = roles["ss_hot"], roles["ss_cold"]
    tt_mid = roles["tt_mid"]

    X_chunks: List[np.ndarray] = []
    anchor_chunks: List[np.ndarray] = []
    table_type_chunks: List[np.ndarray] = []
    cell_chunks: List[np.ndarray] = []
    keys: list = []

    for key, t0 in master_lib.tables_by_key.items():
        cell_name = key[0]
        if cell_name not in cell_set or t0.values is None:
            continue

        anchor_tables = []
        ok = True
        for name in anchor_names:
            t = anchor_libs[name].tables_by_key.get(key)
            if t is None or t.values is None:
                ok = False
                break
            anchor_tables.append(t)
        if not ok:
            continue

        table_type = key[-1]
        arc_key = key[:-1]
        sense, ttype = arc_attr_index.get(arc_key, ("na", "na"))
        info = parse_cell_name(cell_name)
        family_code = float(family_vocab.get(info.family, -1))
        log_strength = float(np.log(info.drive_strength))

        anchor_vals = np.stack([t.values.ravel() for t in anchor_tables], axis=1)  # (49, n_anchors)
        n = anchor_vals.shape[0]  # 49

        log_slew_grid = np.log(np.asarray(t0.index_1, dtype=float))[row_idx].ravel()
        log_load_grid = np.log(np.asarray(t0.index_2, dtype=float))[col_idx].ravel()

        log_anchor = np.log(np.abs(anchor_vals) + EPS).astype(np.float32)  # (49, n_anchors)

        # Lever 1: per-cell measured response-signature features (see
        # SENSITIVITY_FEATURE_NAMES docstring above) -- pure log-ratios
        # between anchor pairs, roles resolved per `anchor_names` (see
        # `_resolve_anchor_roles` -- NEVER a hardcoded alpha-stage index).
        n_sens = len(SENSITIVITY_FEATURE_NAMES)
        sens = np.empty((n, n_sens), dtype=np.float32)
        sens[:, 0] = log_anchor[:, ff_hot] - log_anchor[:, ff_cold]
        sens[:, 1] = log_anchor[:, ss_hot] - log_anchor[:, ss_cold]
        sens[:, 2] = log_anchor[:, ss_hot] - log_anchor[:, ff_hot]
        sens[:, 3] = log_anchor[:, ss_cold] - log_anchor[:, ff_cold]
        sens[:, 4] = log_anchor[:, tt_mid] - log_anchor[:, ss_hot]
        sens[:, 5] = log_anchor[:, tt_mid] - log_anchor[:, ff_hot]

        # Lever 2: within-table row/col gradients per anchor (surface
        # shape context). Reshape relies on t.values.ravel()'s default
        # row-major order matching row_idx/col_idx's -- verified above.
        log_anchor_grid = log_anchor.reshape(7, 7, n_anchors)
        grad_row = np.gradient(log_anchor_grid, axis=0).reshape(n, n_anchors).astype(np.float32)
        grad_col = np.gradient(log_anchor_grid, axis=1).reshape(n, n_anchors).astype(np.float32)

        numeric = np.empty((n, len(NUMERIC_FEATURE_NAMES)), dtype=np.float32)
        off = 0
        numeric[:, off : off + n_anchors] = log_anchor
        off += n_anchors
        numeric[:, off : off + n_sens] = sens
        off += n_sens
        numeric[:, off : off + n_anchors] = grad_row
        off += n_anchors
        numeric[:, off : off + n_anchors] = grad_col
        off += n_anchors
        numeric[:, off + 0] = slew_idx_norm_grid
        numeric[:, off + 1] = load_idx_norm_grid
        numeric[:, off + 2] = log_slew_grid
        numeric[:, off + 3] = log_load_grid
        numeric[:, off + 4] = log_strength
        numeric[:, off + 5] = family_code

        cat = np.tile(
            np.concatenate([sense_onehot[sense], ttype_onehot[ttype], table_onehot[table_type]]),
            (n, 1),
        ).astype(np.float32)

        X_chunks.append(np.hstack([numeric, cat]))
        anchor_chunks.append(anchor_vals.astype(np.float64))
        table_type_chunks.append(np.full(n, table_type))
        cell_chunks.append(np.full(n, cell_name))
        keys.extend([key] * n)

    if not X_chunks:
        raise ValueError("no samples produced -- empty cells, or no key has all 5 anchors present?")

    return Phase4Dataset(
        X=np.concatenate(X_chunks, axis=0),
        anchor_values=np.concatenate(anchor_chunks, axis=0),
        table_type=np.concatenate(table_type_chunks),
        cell=np.concatenate(cell_chunks),
        keys=keys,
        anchor_names=anchor_names,
    )


# ---------------------------------------------------------------------------
# Cross-table features (2026-07-29 improvement round, docs/phase4_results.md
# "跨表格特徵"): the largest remaining error is fall_power/rise_power (see
# the fall_power diagnosis notes). Physically, an internal_power arc and
# its "same event" delay-family arcs (cell_fall/fall_transition for
# fall_power; cell_rise/rise_transition for rise_power) describe the SAME
# output-pin transition -- so the delay arc's own anchor values and
# response-signature carry directly relevant information a power-only
# model never sees. This is an *additive* feature block
# (`build_xtable_features`), never baked into `build_base_dataset`'s
# default output -- delay-table predictions are completely unaffected
# (docs/phase4_results.md "delay 表預測不變"), and callers opt in
# explicitly (e.g. a `feature_mode="full_xtable"` switch) by hstacking
# the extra columns onto `ds.X`.
# ---------------------------------------------------------------------------

XTABLE_COMPANION_TABLE_TYPES: Dict[str, Tuple[str, str]] = {
    "fall_power": ("cell_fall", "fall_transition"),
    "rise_power": ("cell_rise", "rise_transition"),
}


def build_power_to_timing_arc_map(lib: LibertyFile) -> Tuple[Dict[tuple, tuple], int, int]:
    """Map each internal_power arc's key-prefix `(cell, pin,
    "internal_power", arc_index)` to the "same event" timing arc's key-
    prefix `(cell, pin, "timing", arc_index)`, matched within the same
    (cell, pin) by `related_pin` (narrowed by `when` if `related_pin`
    alone is ambiguous). `arc_index` is assigned independently per
    group_type (see `liberty.parser.Arc`'s docstring) so it can *not* be
    assumed to line up between a pin's timing arcs and its internal_power
    arcs -- this match is by physical identity (related_pin/when), never
    by position.

    Arc structure (which arcs exist, their related_pin/when) is a
    technology characteristic shared by every corner (same rationale as
    `build_arc_attr_index`), so this only needs to be built once from any
    single anchor lib and reused for every cell split / delivery-corner
    model.

    Returns `(mapping, n_matched, n_unmatched)` -- `n_unmatched` counts
    internal_power arcs with zero or an ambiguous (>1) related_pin/when
    match; those arcs fall back to the existing (non-cross-table)
    features only (`build_xtable_features` zero-fills their extra columns
    and marks `xtable_has_match=0`), never raise.
    """
    mapping: Dict[tuple, tuple] = {}
    n_matched = 0
    n_unmatched = 0
    for cell in lib.cells.values():
        for pin in cell.pins.values():
            timing_arcs = [a for a in pin.arcs if a.group_type == "timing"]
            power_arcs = [a for a in pin.arcs if a.group_type == "internal_power"]
            if not timing_arcs or not power_arcs:
                continue
            for pa in power_arcs:
                candidates = [ta for ta in timing_arcs if ta.related_pin == pa.related_pin]
                if len(candidates) > 1:
                    when_candidates = [ta for ta in candidates if ta.when == pa.when]
                    if len(when_candidates) == 1:
                        candidates = when_candidates
                if len(candidates) == 1:
                    mapping[(cell.name, pin.name, "internal_power", pa.arc_index)] = (
                        cell.name, pin.name, "timing", candidates[0].arc_index,
                    )
                    n_matched += 1
                else:
                    n_unmatched += 1
    return mapping, n_matched, n_unmatched


def _xtable_feature_names(anchor_names: Sequence[str], companion_label: str) -> List[str]:
    n = len(anchor_names)
    return (
        [f"xtable_{companion_label}_log_anchor_{i}" for i in range(n)]
        + [f"xtable_{companion_label}_{s}" for s in SENSITIVITY_FEATURE_NAMES]
    )


def xtable_feature_names(anchor_names: Sequence[str] = ANCHOR_CORNER_NAMES) -> List[str]:
    """Column names `build_xtable_features` returns, in order -- exposed
    so callers can hstack onto `FEATURE_NAMES` for their own bookkeeping."""
    return (
        _xtable_feature_names(anchor_names, "delay1")
        + _xtable_feature_names(anchor_names, "delay2")
        + ["xtable_has_match"]
    )


def build_xtable_features(
    ds: "Phase4Dataset",
    anchor_libs: Mapping[str, LibertyFile],
    anchor_names: Sequence[str],
    power_to_timing_map: Mapping[tuple, tuple],
) -> Tuple[np.ndarray, List[str], int, int]:
    """Build the cross-table feature block for `ds` (a Phase4Dataset built
    with the same `anchor_names`): for every row belonging to a
    rise_power/fall_power table key, look up the "same event" delay-
    family arc via `power_to_timing_map` and append that arc's two
    companion tables' (`XTABLE_COMPANION_TABLE_TYPES`) own log_anchor_*
    and response-signature features (identical formulas to the base
    features, computed on the *companion* table's anchor values -- never
    the power table's own values). Non-power rows, and power rows whose
    arc has no unique timing-arc match or whose companion tables aren't
    fully populated across `anchor_names`, get every xtable column filled
    with exactly 0.0 and `xtable_has_match=0.0` ("回退到現有特徵並記數" --
    the existing base features in `ds.X` are completely untouched by this
    function; it only ever supplies *additional* columns).

    Never reads any delivery/target-corner lib: `anchor_libs` is the same
    dict `build_base_dataset` was called with, which structurally cannot
    contain a target corner (see that function's docstring) -- this
    function inherits the same zero-leakage guarantee by construction.

    Returns `(extra_X, extra_feature_names, n_matched_rows, n_fallback_rows)`
    -- `n_matched_rows`/`n_fallback_rows` count POINT rows (49 per key),
    for a quick sanity ratio against `ds.n`.
    """
    anchor_names = tuple(anchor_names)
    n_anchors = len(anchor_names)
    roles = _resolve_anchor_roles(anchor_names)
    ff_hot, ff_cold = roles["ff_hot"], roles["ff_cold"]
    ss_hot, ss_cold = roles["ss_hot"], roles["ss_cold"]
    tt_mid = roles["tt_mid"]
    n_sens = len(SENSITIVITY_FEATURE_NAMES)
    block_width = n_anchors + n_sens  # one companion table's contribution
    names = xtable_feature_names(anchor_names)
    n_cols = len(names)

    extra = np.zeros((ds.n, n_cols), dtype=np.float32)
    n_matched_rows = 0
    n_fallback_rows = 0

    n_keys = len(ds.keys) // 49
    for k in range(n_keys):
        i = k * 49
        key = ds.keys[i]
        cell_name, pin_name, group_type, arc_index, table_type = key
        companions = XTABLE_COMPANION_TABLE_TYPES.get(table_type) if group_type == "internal_power" else None
        timing_prefix = power_to_timing_map.get((cell_name, pin_name, group_type, arc_index)) if companions else None

        if companions is None or timing_prefix is None:
            n_fallback_rows += 49
            continue

        row = np.zeros((49, n_cols), dtype=np.float32)
        ok = True
        for c_idx, companion_type in enumerate(companions):
            companion_key = timing_prefix + (companion_type,)
            companion_tables = []
            for name in anchor_names:
                t = anchor_libs[name].tables_by_key.get(companion_key)
                if t is None or t.values is None:
                    ok = False
                    break
                companion_tables.append(t)
            if not ok:
                break
            companion_vals = np.stack([t.values.ravel() for t in companion_tables], axis=1)  # (49, n_anchors)
            log_companion = np.log(np.abs(companion_vals) + EPS).astype(np.float32)
            sens = np.empty((49, n_sens), dtype=np.float32)
            sens[:, 0] = log_companion[:, ff_hot] - log_companion[:, ff_cold]
            sens[:, 1] = log_companion[:, ss_hot] - log_companion[:, ss_cold]
            sens[:, 2] = log_companion[:, ss_hot] - log_companion[:, ff_hot]
            sens[:, 3] = log_companion[:, ss_cold] - log_companion[:, ff_cold]
            sens[:, 4] = log_companion[:, tt_mid] - log_companion[:, ss_hot]
            sens[:, 5] = log_companion[:, tt_mid] - log_companion[:, ff_hot]
            off = c_idx * block_width
            row[:, off : off + n_anchors] = log_companion
            row[:, off + n_anchors : off + block_width] = sens

        if not ok:
            n_fallback_rows += 49
            continue

        row[:, -1] = 1.0  # xtable_has_match
        extra[i : i + 49, :] = row
        n_matched_rows += 49

    return extra, names, n_matched_rows, n_fallback_rows


def extract_raw_values(lib: LibertyFile, keys: Sequence[TableKey]) -> np.ndarray:
    """Pull `lib`'s table values in the same row order as `keys` (a
    Phase4Dataset.keys list -- contiguous 49-row blocks of the same key).
    Used only to fetch the *label* source (a delivery-corner's true
    values, for training/validation) -- inference never calls this."""
    n = len(keys)
    out = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        key = keys[i]
        j = i + 49
        assert all(k == key for k in keys[i:j]), "keys must be in contiguous 49-row blocks"
        table = lib.tables_by_key.get(key)
        if table is None or table.values is None:
            raise KeyError(f"lib {lib.path!r} has no populated table for key {key!r}")
        out[i:j] = table.values.ravel()
        i = j
    return out


def make_label(nearest_anchor: np.ndarray, target: np.ndarray) -> np.ndarray:
    """log(|target| / |nearest_anchor|). Well-defined wherever
    `trainable_mask(nearest_anchor)` holds; rows with target == 0 (only
    possible where nearest_anchor == 0 too, docs/plan.md rule 3, verified
    against the real training set -- see docs/phase4_results.md) give
    log(0) == -inf, which is expected and benign since those rows are
    excluded from training/reconstruction regardless."""
    ratio = np.abs(target) / np.maximum(np.abs(nearest_anchor), EPS)
    with np.errstate(divide="ignore"):
        return np.log(ratio)


def trainable_mask(nearest_anchor: np.ndarray) -> np.ndarray:
    """Rows with a nonzero nearest-anchor value -- the only rows with a
    defined log-ratio label. Zero-anchor rows (docs/plan.md rule 3:
    known-invalid rise_power/fall_power arcs) always reconstruct to an
    exact 0 prediction regardless of any model's output."""
    return nearest_anchor != 0.0


def reconstruct_predictions(
    nearest_anchor: np.ndarray, y_pred: np.ndarray, *, clip: float = CLIP_LOG_RATIO
) -> np.ndarray:
    """Invert `make_label`: turn a model's predicted log-ratio back into a
    value on the original scale, preserving the nearest anchor's sign
    (`predicted = nearest_anchor * exp(y_pred)`, matching
    models.phase3_features.reconstruct_predictions's convention). Rows
    with nearest_anchor == 0 are forced to exactly 0 regardless of
    `y_pred`, per docs/plan.md rule 3."""
    y_pred = np.clip(np.asarray(y_pred, dtype=float), -clip, clip)
    pred = nearest_anchor * np.exp(y_pred)
    return np.where(nearest_anchor == 0.0, 0.0, pred)


def unravel_predictions(keys: Sequence[TableKey], values: np.ndarray) -> Dict[TableKey, np.ndarray]:
    """Group a flat (n,) array of per-row predictions (same row order as
    a Phase4Dataset's `.keys`, contiguous 49-row blocks) back into a dict
    of TableKey -> (7, 7) arrays, ready for liberty.writer.fill_template.
    """
    n = len(keys)
    out: Dict[TableKey, np.ndarray] = {}
    i = 0
    while i < n:
        key = keys[i]
        j = i + 49
        out[key] = values[i:j].reshape(7, 7)
        i = j
    return out


def score_breakdown(y_true: np.ndarray, y_pred: np.ndarray, groups: Mapping[str, np.ndarray]):
    """Run scoring.scorer's point_errors/score_from_errors over
    `(y_true, y_pred)` and additionally break the result down by every
    named grouping array in `groups` (e.g. {"table_type": ds.table_type}).
    Returns `(overall_score, n_points, {group_name: {group_value: (score,
    n_points)}})` -- identical contract to
    models.phase3_features.score_breakdown, duplicated here (it is a
    small pure function with no phase3-specific coupling) so this module
    has no import-time dependency direction issue."""
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


__all__ = [
    "ANCHOR_CORNER_NAMES",
    "NEAREST_ANCHOR_BY_TARGET",
    "DELIVERY_CORNER_NAMES",
    "XTABLE_COMPANION_TABLE_TYPES",
    "build_power_to_timing_arc_map",
    "build_xtable_features",
    "xtable_feature_names",
    "CornerTopology",
    "infer_nearest_anchor_by_target",
    "ALPHA_TOPOLOGY",
    "BETA_ANCHOR_NAMES",
    "BETA_TARGET_NAMES",
    "BETA_TOPOLOGY",
    "FINAL_ANCHOR_NAMES",
    "FINAL_TARGET_NAMES",
    "FINAL_TOPOLOGY",
    "STAGE_TOPOLOGIES",
    "PHASE4_CELL_SPLIT_SEED",
    "PHASE4_DEV_SPLIT_SEED",
    "TRAIN_CELL_FRAC",
    "DEV_TRAIN_FRAC",
    "SCORE_THRESHOLD",
    "FEATURE_NAMES",
    "SENSITIVITY_FEATURE_NAMES",
    "GRADIENT_FEATURE_NAMES",
    "Phase4Dataset",
    "build_family_vocab_for_phase4",
    "build_base_dataset",
    "extract_raw_values",
    "make_label",
    "trainable_mask",
    "reconstruct_predictions",
    "unravel_predictions",
    "score_breakdown",
    "split_cells",
    "split_dev",
    "build_arc_attr_index",
]
