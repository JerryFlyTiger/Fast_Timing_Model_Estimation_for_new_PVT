"""PVT corner metadata parsed from `.lib`/`.tlib` file names.

Filenames follow either ``lib1_<process><voltage>v<temp>c_alpha_100.lib``
(the 100-cell alpha test set, `testcase/alpha_test/{full,partial}/`), e.g.
``lib1_ff0p99vm40c_alpha_100.lib`` = fast-fast, 0.99V, -40C (``m`` = minus,
``0p99`` = 0.99), or ``lib1_<process><voltage>v<temp>c_base_400.tlib`` (the
400-cell official training set, `testcase/training_set/base_nom_*/`,
added 2026-07-26 -- see docs/plan.md section 3). Both suffixes encode the
same corner-naming convention, so a single regex with an alternation
handles both. See CLAUDE.md "Liberty 檔案要點".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union

_CORNER_RE = re.compile(
    r"lib1_(?P<proc>ss|ff|tt)(?P<v>\d+p\d+)v(?P<t>m?\d+)c_"
    r"(?:alpha_100\.lib|base_400\.tlib)$"
)

# Nominal (full-corner) supply voltage per process -- the single voltage
# each process has ground-truth data at in testcase/alpha_test/full/ (and,
# since 2026-07-26, also the 5 "anchor" corners in
# testcase/training_set/base_nom_0p9v/).
NOMINAL_VOLTAGE = {"ss": 0.81, "ff": 0.99, "tt": 0.9}


@dataclass(frozen=True)
class CornerMeta:
    process: str  # "ss" | "ff" | "tt"
    voltage: float
    temperature: float  # degrees C
    name: str  # short corner label, e.g. "ss0p81v125c"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"CornerMeta({self.name})"


def _parse_voltage(token: str) -> float:
    return float(token.replace("p", "."))


def _parse_temperature(token: str) -> float:
    if token.startswith("m"):
        return -float(token[1:])
    return float(token)


def parse_corner_filename(path: Union[str, Path]) -> CornerMeta:
    """Parse a `.lib` path/filename into a CornerMeta."""
    fname = Path(path).name
    m = _CORNER_RE.search(fname)
    if not m:
        raise ValueError(f"filename {fname!r} does not match the lib1_<p><v>v<t>c convention")
    process = m.group("proc")
    voltage = _parse_voltage(m.group("v"))
    temperature = _parse_temperature(m.group("t"))
    name = f"{process}{m.group('v')}v{m.group('t')}c"
    return CornerMeta(process=process, voltage=voltage, temperature=temperature, name=name)
