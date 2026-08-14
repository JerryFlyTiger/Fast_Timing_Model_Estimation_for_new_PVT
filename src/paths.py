"""Central path constants for this repo's test-case data layout.

docs/plan.md section 3 (2026-07-26 data migration): the original
`testcase/full` / `testcase/partial` directories (100 alpha cells) were
moved to `testcase/alpha_test/full` / `testcase/alpha_test/partial`
without any content change, and a new `testcase/training_set/` was added
(400 training cells x all 15 corners, `.tlib` format, zero cell overlap
with the alpha 100). Every module/script that needs one of these paths
imports the constants from here instead of hardcoding `"testcase/..."`
strings, so a future data-layout change only requires editing this one
file (docs/plan.md Phase 4 item 1, "集中成一個 config/常數模組").
"""

from __future__ import annotations

from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTCASE_DIR = REPO_ROOT / "testcase"

# 100 alpha cells x 5 standard-voltage corners, fully populated.
ALPHA_FULL_DIR = TESTCASE_DIR / "alpha_test" / "full"
# Same 100 cells x 10 delivery corners, values blanked out (the writer's
# fill-in template / the checker's scoring target).
ALPHA_PARTIAL_DIR = TESTCASE_DIR / "alpha_test" / "partial"

# The beta / final contest stages, same layout as alpha: 100 cells (a
# different, non-overlapping 100 each) x the 5 corners that stage gets
# fully populated, plus 10 blanked delivery corners. Which 5 are
# populated differs per stage -- alpha gets the standard-voltage
# corners, beta the boost corners, final the buck corners (see
# models.phase4_features.STAGE_TOPOLOGIES).
BETA_FULL_DIR = TESTCASE_DIR / "beta_test" / "full"
BETA_PARTIAL_DIR = TESTCASE_DIR / "beta_test" / "partial"
FINAL_FULL_DIR = TESTCASE_DIR / "final_test" / "full"
FINAL_PARTIAL_DIR = TESTCASE_DIR / "final_test" / "partial"

# Filename stem convention per stage: lib1_<corner>_<tag>_100.lib
STAGE_FULL_DIRS = {"alpha": ALPHA_FULL_DIR, "beta": BETA_FULL_DIR, "final": FINAL_FULL_DIR}


def stage_full_lib(stage: str, corner: str) -> Path:
    """Path to one fully-populated `<stage>` lib for `corner`."""
    return STAGE_FULL_DIRS[stage] / f"lib1_{corner}_{stage}_100.lib"

# 400 training cells x all 15 corners (5 standard-voltage "anchor" +
# 10 "delivery"), `.tlib` format, fully populated. Zero overlap with the
# alpha 100 cells.
TRAINING_SET_DIR = TESTCASE_DIR / "training_set"
TRAINING_SET_SUBDIRS = ("base_nom_0p8v", "base_nom_0p9v", "base_nom_1p0v")

OUTPUT_DIR = REPO_ROOT / "output"
DOCS_DIR = REPO_ROOT / "docs"


def training_set_files() -> List[Path]:
    """All 15 training-set `.tlib` file paths, sorted."""
    files: List[Path] = []
    for sub in TRAINING_SET_SUBDIRS:
        files.extend((TRAINING_SET_DIR / sub).glob("*.tlib"))
    return sorted(files)
