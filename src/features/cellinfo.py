"""Parse a cell name into a (function family, drive strength) pair.

Naming convention observed across all 100 cells in every released `.lib`
(see docs/phase2_results.md): ``<base>M<strength>``, e.g. ``AN2AM16`` ->
base ``AN2A``, strength ``16``; ``INVM12`` -> base ``INV``, strength
``12``. The "function family" used for the Phase 2 sensitivity-group
gain (docs/plan.md Phase 2 item 3, "cell type") is the leading alphabetic
run of the base (``AN2A`` -> ``AN``, ``ND2B1`` -> ``ND``), which coarsens
fan-in/variant suffixes away. Note: in this 100-cell library every base
is unique (no cell appears at multiple drive strengths under the same
base), so "cell type" and "cell instance" only diverge at the coarser
function-family level -- see docs/phase2_results.md for the empirical
count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

_CELL_RE = re.compile(r"^(?P<base>.+)M(?P<strength>\d+)$")
_PREFIX_RE = re.compile(r"^[A-Za-z]+")


@dataclass(frozen=True)
class CellInfo:
    name: str
    base: str
    family: str
    drive_strength: int


@lru_cache(maxsize=None)
def parse_cell_name(name: str) -> CellInfo:
    m = _CELL_RE.match(name)
    if not m:
        raise ValueError(f"cell name {name!r} does not match the <base>M<strength> convention")
    base = m.group("base")
    strength = int(m.group("strength"))
    prefix_m = _PREFIX_RE.match(base)
    family = prefix_m.group(0) if prefix_m else base
    return CellInfo(name=name, base=base, family=family, drive_strength=strength)
