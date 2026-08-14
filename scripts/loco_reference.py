"""Phase 1 LOCO reference score (docs/plan.md step 7).

We have no ground truth for the 10 partial (target) corners, so we can't
score our baseline output directly. Instead we quantify how good a naive
"just copy another full corner's values" strategy is among the 5 full
corners we *do* have ground truth for: hold out one full corner, predict
it by copying grid-point values from a different full corner, and score
with scoring.scorer.compare_libs. This calibrates expectations for the
Phase 1 baseline (which does exactly this kind of direct copy, just
same-process/same-temperature) and gives a floor that Phase 2/3 should
clear.

Usage: python3 scripts/loco_reference.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from liberty.parser import parse_file
from scoring.scorer import compare_libs
from paths import ALPHA_FULL_DIR

FULL_DIR = ALPHA_FULL_DIR

PAIRS = [
    (
        "ss0p81v125c -> ff0p99v125c (cross-process, same temperature)",
        "lib1_ss0p81v125c_alpha_100.lib",
        "lib1_ff0p99v125c_alpha_100.lib",
    ),
    (
        "ss0p81v125c -> ss0p81vm40c (same process, cross-temperature)",
        "lib1_ss0p81v125c_alpha_100.lib",
        "lib1_ss0p81vm40c_alpha_100.lib",
    ),
    (
        "ff0p99v125c -> ff0p99vm40c (same process, cross-temperature)",
        "lib1_ff0p99v125c_alpha_100.lib",
        "lib1_ff0p99vm40c_alpha_100.lib",
    ),
    (
        "ss0p81v125c -> tt0p9v25c (cross-process, cross-temperature)",
        "lib1_ss0p81v125c_alpha_100.lib",
        "lib1_tt0p9v25c_alpha_100.lib",
    ),
]


def main():
    for label, source_name, target_name in PAIRS:
        source = parse_file(str(FULL_DIR / source_name))
        target = parse_file(str(FULL_DIR / target_name))
        report = compare_libs(target, source)  # reference=ground truth, predicted=copy
        print(f"=== {label} ===")
        print(report)
        print()


if __name__ == "__main__":
    main()
