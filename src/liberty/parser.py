"""Liberty (.lib) parser, table-level granularity only.

This module does NOT contain any estimation/scaling logic (see CLAUDE.md
"parser 不含估算邏輯"). It scans a Liberty file with a structured
brace/regex scan (not a full Liberty grammar) and extracts:

- library name
- cells (name, pins, max_capacitance)
- timing / internal_power groups (related_pin, timing_sense, timing_type,
  when, related_pg_pin)
- the 6 table kinds we care about (cell_rise, cell_fall, rise_transition,
  fall_transition, rise_power, fall_power): index_1, index_2, and either a
  7x7 numpy array of values or a "blank" marker (``is_blank=True``).

Every :class:`ValueTable` also carries the exact character offsets of the
content of each of its 7 quoted value rows in the *original source text*
(``row_spans``). The writer uses these offsets to do byte-precise
template fill-in without ever re-serializing the file.

Braces/parens inside quoted strings are never treated as structural: we
skip over ``"..."`` literals verbatim while scanning for group
boundaries. Group bodies are found in a single linear pass (each byte of
the file is visited O(1) times), so a ~5.5MB file parses in well under a
second.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

TABLE_KINDS = (
    "cell_rise",
    "cell_fall",
    "rise_transition",
    "fall_transition",
    "rise_power",
    "fall_power",
)

# A "group statement" header: `name (params) {`. Params never contain a
# literal '(' / ')' / newline / '"' in this dataset (verified against all
# 5 full .lib files), so a simple non-greedy character class is enough.
_HEADER_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()\n]*)\)\s*\{")

_VALUES_RE = re.compile(r"values\s*\(\s*\\\r?\n(?P<rows>.*?)\n\s*\);", re.S)
_INDEX_RE_TMPL = r'{name}\s*\(\s*"([^"]*)"\s*\)\s*;'
_INDEX1_RE = re.compile(_INDEX_RE_TMPL.format(name="index_1"))
_INDEX2_RE = re.compile(_INDEX_RE_TMPL.format(name="index_2"))
_SCALAR_ATTR_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^;\n]*);")
_ROW_RE = re.compile(r'"([^"]*)"')

TableKey = tuple


@dataclass
class ValueTable:
    """One `cell_rise (...) { ... }`-style table group."""

    table_type: str  # one of TABLE_KINDS
    index_1: tuple
    index_2: tuple
    is_blank: bool
    values: Optional[np.ndarray]  # shape (7, 7), None iff is_blank
    row_spans: list  # 7 x (start, end) char offsets of each row's quoted content
    key: TableKey  # (cell, pin, group_type, arc_index, table_type)


@dataclass
class Arc:
    """A `timing () { ... }` or `internal_power () { ... }` group."""

    group_type: str  # "timing" | "internal_power"
    arc_index: int  # 0-based position among siblings of the same group_type/pin
    related_pin: Optional[str]
    when: Optional[str]
    timing_sense: Optional[str]
    timing_type: Optional[str]
    related_pg_pin: Optional[str]
    tables: dict = field(default_factory=dict)  # table_type -> ValueTable


@dataclass
class Pin:
    name: str
    direction: Optional[str]
    max_capacitance: Optional[float]
    attributes: dict = field(default_factory=dict)
    arcs: list = field(default_factory=list)  # list[Arc]


@dataclass
class Cell:
    name: str
    pins: dict = field(default_factory=dict)  # name -> Pin


@dataclass
class LibertyFile:
    path: Optional[str]
    text: str
    library_name: str
    cells: dict  # name -> Cell
    tables: list  # flat, in document order
    tables_by_key: dict  # TableKey -> ValueTable


def _find_matching_brace(text: str, pos: int) -> int:
    """`pos` points right after an opening '{'. Return index of the
    matching '}', skipping over quoted string literals."""
    depth = 1
    i = pos
    n = len(text)
    while depth:
        c = text[i]
        if c == '"':
            i = text.index('"', i + 1) + 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    return i - 1


def _iter_children(text: str, start: int, end: int):
    """Yield direct child group statements of text[start:end] as
    (name, params, header_start, body_start, body_end) tuples, where
    body_start/body_end delimit the region strictly inside the group's
    braces. Does not descend into children of children."""
    i = start
    while True:
        m = _HEADER_RE.search(text, i, end)
        if not m:
            return
        name, params = m.group(1), m.group(2)
        body_start = m.end()
        body_end = _find_matching_brace(text, body_start)
        yield name, params, m.start(), body_start, body_end
        i = body_end + 1


def _scalar_attrs(text: str, start: int, end: int, child_spans) -> dict:
    """Regex-scan text[start:end] for `key : value;` attributes, skipping
    over the character ranges occupied by nested child groups (so that
    attributes belonging to nested arcs/tables are not picked up)."""
    attrs = {}
    cursor = start
    spans = sorted(child_spans)
    boundaries = [(s, e) for s, e in spans] + [(end, end)]
    for child_start, child_end in boundaries:
        for m in _SCALAR_ATTR_RE.finditer(text, cursor, child_start):
            key = m.group(1)
            val = m.group(2).strip()
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            attrs[key] = val
        cursor = child_end + 1
    return attrs


def _parse_index(text: str, start: int, end: int, regex: re.Pattern) -> Optional[tuple]:
    m = regex.search(text, start, end)
    if not m:
        return None
    return tuple(float(x) for x in m.group(1).split(","))


def _parse_table_body(text: str, start: int, end: int):
    """Parse index_1/index_2/values out of a table group's body.
    Returns (index_1, index_2, is_blank, values, row_spans)."""
    index_1 = _parse_index(text, start, end, _INDEX1_RE)
    index_2 = _parse_index(text, start, end, _INDEX2_RE)

    m = _VALUES_RE.search(text, start, end)
    if not m:
        raise ValueError("no values(...) block found in table body")
    rows_start = m.start("rows")
    rows_text = m.group("rows")

    row_spans = []
    row_strs = []
    for rm in _ROW_RE.finditer(rows_text):
        abs_start = rows_start + rm.start(1)
        abs_end = rows_start + rm.end(1)
        row_spans.append((abs_start, abs_end))
        row_strs.append(rm.group(1))

    is_blank = not any(any(c.isdigit() for c in row) for row in row_strs)
    if is_blank:
        values = None
    else:
        values = np.array(
            [[float(x) for x in row.split(",")] for row in row_strs],
            dtype=float,
        )

    return index_1, index_2, is_blank, values, row_spans


def parse_text(text: str, path: Optional[str] = None) -> LibertyFile:
    lib_m = _HEADER_RE.search(text)
    if not lib_m or lib_m.group(1) != "library":
        raise ValueError("expected a top-level `library (...) { ... }` group")
    library_name = lib_m.group(2)
    lib_body_start = lib_m.end()
    lib_body_end = _find_matching_brace(text, lib_body_start)

    cells: dict = {}
    tables: list = []
    tables_by_key: dict = {}

    for name, params, _hstart, body_start, body_end in _iter_children(
        text, lib_body_start, lib_body_end
    ):
        if name != "cell":
            continue
        cell_name = params.strip()
        cell = Cell(name=cell_name, pins={})
        cells[cell_name] = cell

        for pname, pparams, _phstart, pbody_start, pbody_end in _iter_children(
            text, body_start, body_end
        ):
            if pname != "pin":
                continue
            pin_name = pparams.strip()

            pin_children = list(_iter_children(text, pbody_start, pbody_end))
            arc_children = [
                c for c in pin_children if c[0] in ("timing", "internal_power")
            ]
            child_spans = [(hs, be) for _n, _p, hs, _bs, be in pin_children]
            pin_attrs = _scalar_attrs(text, pbody_start, pbody_end, child_spans)
            direction = pin_attrs.get("direction")
            max_cap = pin_attrs.get("max_capacitance")
            max_cap = float(max_cap) if max_cap is not None else None

            pin = Pin(
                name=pin_name,
                direction=direction,
                max_capacitance=max_cap,
                attributes=pin_attrs,
                arcs=[],
            )
            cell.pins[pin_name] = pin

            arc_counters = {"timing": 0, "internal_power": 0}
            for aname, aparams, ahstart, abody_start, abody_end in arc_children:
                arc_index = arc_counters[aname]
                arc_counters[aname] += 1

                arc_body_children = list(_iter_children(text, abody_start, abody_end))
                table_children = [c for c in arc_body_children if c[0] in TABLE_KINDS]
                arc_child_spans = [(hs, be) for _n, _p, hs, _bs, be in arc_body_children]
                arc_attrs = _scalar_attrs(text, abody_start, abody_end, arc_child_spans)

                arc = Arc(
                    group_type=aname,
                    arc_index=arc_index,
                    related_pin=arc_attrs.get("related_pin"),
                    when=arc_attrs.get("when"),
                    timing_sense=arc_attrs.get("timing_sense"),
                    timing_type=arc_attrs.get("timing_type"),
                    related_pg_pin=arc_attrs.get("related_pg_pin"),
                    tables={},
                )
                pin.arcs.append(arc)

                for tname, tparams, thstart, tbody_start, tbody_end in table_children:
                    index_1, index_2, is_blank, values, row_spans = _parse_table_body(
                        text, tbody_start, tbody_end
                    )
                    key = (cell_name, pin_name, aname, arc_index, tname)
                    table = ValueTable(
                        table_type=tname,
                        index_1=index_1,
                        index_2=index_2,
                        is_blank=is_blank,
                        values=values,
                        row_spans=row_spans,
                        key=key,
                    )
                    arc.tables[tname] = table
                    tables.append(table)
                    tables_by_key[key] = table

    return LibertyFile(
        path=path,
        text=text,
        library_name=library_name,
        cells=cells,
        tables=tables,
        tables_by_key=tables_by_key,
    )


def parse_file(path: str) -> LibertyFile:
    with open(path, "r", encoding="ascii") as f:
        text = f.read()
    return parse_text(text, path=path)
