"""Shared test utilities (not itself a test module)."""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from paths import ALPHA_FULL_DIR, ALPHA_PARTIAL_DIR  # noqa: E402

# docs/plan.md 2026-07-26: these used to be testcase/full, testcase/partial;
# both directories moved (content unchanged) to testcase/alpha_test/{full,partial}
# when the official 400-cell training_set was added. Path constants now live
# in src/paths.py; kept as FULL_DIR/PARTIAL_DIR here since every existing test
# imports these two names.
FULL_DIR = ALPHA_FULL_DIR
PARTIAL_DIR = ALPHA_PARTIAL_DIR

FULL_LIBS = sorted(FULL_DIR.glob("*.lib"))
PARTIAL_LIBS = sorted(PARTIAL_DIR.glob("*.lib"))

# Matches an entire `values ( \...\n  );` block, used to build a
# structural "skeleton" of a .lib file with all values content erased --
# two files with the same skeleton are byte-identical everywhere except
# inside values(...) blocks.
_VALUES_BLOCK_RE = re.compile(r"values\s*\(\s*\\\r?\n.*?\n\s*\);", re.S)


def skeleton(text: str) -> str:
    return _VALUES_BLOCK_RE.sub("VALUES_BLOCK", text)


BLANK_ROW = ", ".join([" "] * 7)
