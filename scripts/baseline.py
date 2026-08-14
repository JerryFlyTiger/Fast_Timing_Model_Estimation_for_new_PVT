"""Phase 1 baseline: fill each partial-corner template by directly copying
the same-process, same-temperature standard-voltage full corner's table
values, grid-point-for-grid-point (docs/plan.md step 6). This ignores
that index_2 (load) actually differs per corner -- a known, documented
Phase 1 simplification; Phase 2 replaces this with a physical scaling
model.

Usage: python3 scripts/baseline.py
Writes one filled .lib per testcase/alpha_test/partial/*.lib into output/,
with the same filename.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from liberty.parser import parse_file
from liberty.writer import fill_template
from paths import ALPHA_FULL_DIR, ALPHA_PARTIAL_DIR, OUTPUT_DIR

FULL_DIR = ALPHA_FULL_DIR
PARTIAL_DIR = ALPHA_PARTIAL_DIR

# partial corner filename -> source full corner filename (same process,
# same temperature, standard voltage).
SOURCE_MAP = {
    "lib1_ss0p72v125c_alpha_100.lib": "lib1_ss0p81v125c_alpha_100.lib",
    "lib1_ss0p9v125c_alpha_100.lib": "lib1_ss0p81v125c_alpha_100.lib",
    "lib1_ss0p72vm40c_alpha_100.lib": "lib1_ss0p81vm40c_alpha_100.lib",
    "lib1_ss0p9vm40c_alpha_100.lib": "lib1_ss0p81vm40c_alpha_100.lib",
    "lib1_ff0p88v125c_alpha_100.lib": "lib1_ff0p99v125c_alpha_100.lib",
    "lib1_ff1p1v125c_alpha_100.lib": "lib1_ff0p99v125c_alpha_100.lib",
    "lib1_ff0p88vm40c_alpha_100.lib": "lib1_ff0p99vm40c_alpha_100.lib",
    "lib1_ff1p1vm40c_alpha_100.lib": "lib1_ff0p99vm40c_alpha_100.lib",
    "lib1_tt0p8v25c_alpha_100.lib": "lib1_tt0p9v25c_alpha_100.lib",
    "lib1_tt1p0v25c_alpha_100.lib": "lib1_tt0p9v25c_alpha_100.lib",
}


def build_baseline(partial_name: str, source_name: str) -> None:
    partial_path = PARTIAL_DIR / partial_name
    source_path = FULL_DIR / source_name
    output_path = OUTPUT_DIR / partial_name

    partial_lib = parse_file(str(partial_path))
    source_lib = parse_file(str(source_path))

    predictions = {}
    missing = []
    for table in partial_lib.tables:
        if not table.is_blank:
            continue
        source_table = source_lib.tables_by_key.get(table.key)
        if source_table is None or source_table.values is None:
            missing.append(table.key)
            continue
        predictions[table.key] = source_table.values

    if missing:
        raise RuntimeError(
            f"{partial_name}: {len(missing)} blank tables have no source value, "
            f"e.g. {missing[0]!r}"
        )

    filled_text = fill_template(partial_lib, predictions)
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path.write_text(filled_text, encoding="ascii")
    print(f"{partial_name:40s} <- {source_name:40s} ({len(predictions)} tables filled)")


def main():
    for partial_name, source_name in SOURCE_MAP.items():
        build_baseline(partial_name, source_name)


if __name__ == "__main__":
    main()
