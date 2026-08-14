"""Post-delivery verification of `output/*.lib` (2026-08-09, direction F).

Checks, per delivered corner:

1. **Template fidelity**: the output file must be byte-identical to the
   official partial template EXCEPT inside `values(...)` quoted rows --
   verified by normalizing every values-row's quoted content to `#` in
   both files and comparing the results byte-for-byte; additionally every
   output values-row must parse as exactly 7 finite floats.
2. **Structural parse**: `liberty.parser` must read every filled table as
   a finite 7x7 array, and the filled key set must cover every blank key
   in the template.
3. **Rule 3 (all-zero power)**: any rise_power/fall_power table whose
   nearest-anchor table (in the alpha full libs) is all-zero must be
   delivered as exactly all-zero.
4. **Physical cross-corner audit** (`scoring.audits`): process-ordering
   inequalities (predicted-vs-true across process corners) and delay
   scaling-band checks -- the phase-2.5 review's delivered-output audit,
   now applied to the real delivery.

Exit code 0 iff every check passes.

Usage:
    python3 scripts/verify_delivery.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from liberty.parser import parse_file
from models.phase4_features import DELIVERY_CORNER_NAMES, NEAREST_ANCHOR_BY_TARGET
from paths import ALPHA_FULL_DIR, ALPHA_PARTIAL_DIR, OUTPUT_DIR
from scoring.audits import check_delay_scaling_bands, run_cross_corner_inequality_audit

VALUES_ROW = re.compile(r'^(\s*)"([^"]*)"(.*)$')


def normalized_lines_and_rows(path: Path):
    """File lines with every values-block quoted row replaced by `#`,
    plus the raw quoted contents of those rows (for float parsing)."""
    lines = []
    rows = []
    in_values = False
    for line in path.read_text().splitlines(keepends=True):
        if in_values:
            m = VALUES_ROW.match(line)
            if m:
                rows.append(m.group(2))
                lines.append(f'{m.group(1)}"#"{m.group(3)}\n' if line.endswith("\n")
                             else f'{m.group(1)}"#"{m.group(3)}')
                if ");" in line:
                    in_values = False
                continue
            if ");" in line:
                in_values = False
            lines.append(line)
            continue
        if "values" in line and "(" in line:
            in_values = ");" not in line
        lines.append(line)
    return lines, rows


def main() -> int:
    failures = []

    print("=== 1+2. template fidelity + structural parse ===")
    predicted = {}
    for corner in DELIVERY_CORNER_NAMES:
        name = f"lib1_{corner}_alpha_100.lib"
        out_path, tpl_path = OUTPUT_DIR / name, ALPHA_PARTIAL_DIR / name
        if not out_path.exists():
            failures.append(f"{corner}: missing {out_path}")
            continue
        out_norm, out_rows = normalized_lines_and_rows(out_path)
        tpl_norm, tpl_rows = normalized_lines_and_rows(tpl_path)
        if out_norm != tpl_norm:
            ndiff = sum(1 for a, b in zip(out_norm, tpl_norm) if a != b) + abs(len(out_norm) - len(tpl_norm))
            failures.append(f"{corner}: {ndiff} non-values line(s) differ from template")
        if len(out_rows) != len(tpl_rows):
            failures.append(f"{corner}: values-row count {len(out_rows)} != template {len(tpl_rows)}")
        bad_rows = 0
        for row in out_rows:
            vals = [v.strip() for v in row.split(",")]
            if len(vals) != 7 or not all(v and np.isfinite(float(v)) for v in vals):
                bad_rows += 1
        if bad_rows:
            failures.append(f"{corner}: {bad_rows} values row(s) not 7 finite floats")

        lib = parse_file(str(out_path))
        n_filled = sum(1 for t in lib.tables_by_key.values() if t.values is not None)
        n_bad_shape = sum(1 for t in lib.tables_by_key.values()
                          if t.values is not None and (t.values.shape != (7, 7) or not np.isfinite(t.values).all()))
        if n_bad_shape:
            failures.append(f"{corner}: {n_bad_shape} table(s) not finite 7x7")
        predicted[corner] = lib
        print(f"  {corner:16s} rows={len(out_rows):6d}  filled_tables={n_filled:5d}  "
              f"{'OK' if not any(corner in f for f in failures) else 'FAIL'}")

    print("\n=== 3. rule-3 all-zero power tables ===")
    anchor_libs = {}
    for corner, lib in predicted.items():
        anchor = NEAREST_ANCHOR_BY_TARGET[corner]
        if anchor not in anchor_libs:
            anchor_libs[anchor] = parse_file(str(ALPHA_FULL_DIR / f"lib1_{anchor}_alpha_100.lib"))
        alib = anchor_libs[anchor]
        n_zero = n_viol = 0
        for key, atab in alib.tables_by_key.items():
            if key[-1] not in ("rise_power", "fall_power") or atab.values is None:
                continue
            if np.all(atab.values == 0):
                n_zero += 1
                ptab = lib.tables_by_key.get(key)
                if ptab is None or ptab.values is None or not np.all(ptab.values == 0):
                    n_viol += 1
        if n_viol:
            failures.append(f"{corner}: {n_viol}/{n_zero} all-zero-anchor power tables not delivered as 0")
        print(f"  {corner:16s} all-zero-anchor power tables={n_zero:3d}  violations={n_viol}")

    print("\n=== 4. physical cross-corner audit ===")
    truth = {}
    for path in sorted(ALPHA_FULL_DIR.glob("*.lib")):
        cname = path.name.replace("lib1_", "").replace("_alpha_100.lib", "")
        truth[cname] = anchor_libs.get(cname) or parse_file(str(path))
    for res in run_cross_corner_inequality_audit(predicted, truth):
        print(f"  {res.summary_line()}")
        if not res.passed:
            failures.append(f"inequality audit failed: {res.name}")
    for chk in check_delay_scaling_bands(predicted, truth, NEAREST_ANCHOR_BY_TARGET):
        print(f"  {chk.summary_line()}")
        if not chk.passed:
            failures.append(f"scaling band failed: {chk.name}")

    print("\n=== verdict ===")
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        return 1
    print("  ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
