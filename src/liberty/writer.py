"""Template fill-in writer.

Strict requirement (see CLAUDE.md / docs/plan.md): the output is produced
by taking the *original bytes* of a partial (template) .lib file and only
overwriting the blank ``values(...)`` row slots with formatted numbers.
Every other byte -- whitespace, punctuation, comments, attribute values,
even the already-filled all-zero power rows some partial files ship with
-- is left untouched. We never re-serialize the file from a parsed
representation.

Number formatting: every value in the 5 reference `testcase/alpha_test/full/*.lib`
files matches Python's ``'%.6g' % value`` formatting exactly (verified:
0 mismatches across ~1.2M values, including negative numbers and
scientific notation such as ``8.07996e-05``). We use that same format
here, with the single normalization that an exact zero is always
rendered as ``"0"`` (never ``"-0"``, which never appears in the
reference data but is what ``%.6g`` would print for ``-0.0``).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .parser import LibertyFile, TableKey


def format_value(x: float) -> str:
    """Render a single scalar exactly the way the reference .lib files do."""
    if x == 0:
        return "0"
    return "%.6g" % x


def render_row(values) -> str:
    """Render one row (7 floats) the way it appears inside a values(...)
    block: comma-space separated, no surrounding whitespace."""
    return ", ".join(format_value(v) for v in values)


def fill_template(
    lib: LibertyFile,
    predictions: Mapping[TableKey, np.ndarray],
    *,
    require_all_blanks_filled: bool = True,
) -> str:
    """Return the filled-in .lib text for a parsed template `lib`.

    `predictions` maps a table key (as produced by
    `liberty.parser.parse_text`, i.e. `(cell, pin, group_type, arc_index,
    table_type)`) to a 7x7 array-like of predicted values. Only tables
    that are blank in the template are replaced; tables that already
    contain values (e.g. the known-zero invalid power arcs) are left
    byte-for-byte untouched, and predictions for them (if provided) are
    ignored.

    If `require_all_blanks_filled` is True (default), raises KeyError
    naming the first missing key when a blank table in the template has
    no corresponding entry in `predictions`.
    """
    text = lib.text
    pieces = []
    cursor = 0

    # Collect (start, end, replacement_text) for every blank table's rows,
    # in document order, then splice once over the original text.
    edits = []
    for table in lib.tables:
        if not table.is_blank:
            continue
        if table.key not in predictions:
            if require_all_blanks_filled:
                raise KeyError(f"no prediction supplied for blank table {table.key!r}")
            continue
        values = np.asarray(predictions[table.key], dtype=float)
        if values.shape != (7, 7):
            raise ValueError(
                f"prediction for {table.key!r} has shape {values.shape}, expected (7, 7)"
            )
        if not np.isfinite(values).all():
            raise ValueError(
                f"prediction for {table.key!r} contains NaN/Inf; refusing to write "
                "an invalid Liberty literal"
            )
        for row_idx, (start, end) in enumerate(table.row_spans):
            edits.append((start, end, render_row(values[row_idx])))

    edits.sort(key=lambda e: e[0])
    for start, end, replacement in edits:
        if start < cursor:
            raise ValueError("overlapping edits detected while filling template")
        pieces.append(text[cursor:start])
        pieces.append(replacement)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def fill_template_file(
    template_path: str,
    predictions: Mapping[TableKey, np.ndarray],
    output_path: str,
    *,
    require_all_blanks_filled: bool = True,
) -> None:
    """Parse `template_path`, fill in blanks from `predictions`, and write
    the result to `output_path`."""
    from .parser import parse_file

    lib = parse_file(template_path)
    filled = fill_template(
        lib, predictions, require_all_blanks_filled=require_all_blanks_filled
    )
    with open(output_path, "w", encoding="ascii") as f:
        f.write(filled)
