"""Concatenate per-corner `--dump-errors` .npz files into one dump.

`scripts/phase4_final_validate.py` writes one dump per invocation, and
docs/round_20260810.md section 4.1 requires the huber config to be run one
corner at a time (batch runs get killed non-deterministically). So a full
10-corner huber run produces 10 dumps where the downstream audits expect
one. This joins them.

The per-run scalar metadata (config / stage / fold / seeds) must agree
across every input -- a mismatch means the parts came from different
experiments and merging them would silently produce a dump that
describes no real run. Duplicate corners are likewise rejected rather
than silently double-counted in the pooled score.

Usage:
    python3 scripts/merge_error_dumps.py OUT.npz IN1.npz IN2.npz ...
    python3 scripts/merge_error_dumps.py OUT.npz dir/*.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ARRAY_FIELDS = ("corner", "cell", "table_type", "y_true", "y_pred", "nearest_anchor")
META_FIELDS = ("meta_config", "meta_stage", "meta_fold", "meta_n_seeds")


def main(out_path: str, in_paths: list[str]) -> None:
    if not in_paths:
        raise SystemExit("no input dumps given")

    parts = []
    meta = None
    seen_corners: dict[str, str] = {}
    for p in in_paths:
        d = np.load(p, allow_pickle=False)
        this_meta = {f: str(d[f][0]) for f in META_FIELDS}
        if meta is None:
            meta = this_meta
        elif this_meta != meta:
            raise SystemExit(
                f"metadata mismatch: {p} has {this_meta}, earlier inputs have {meta}. "
                f"These are different experiments; refusing to merge."
            )
        for c in sorted(set(d["corner"].tolist())):
            if c in seen_corners:
                raise SystemExit(f"corner {c!r} appears in both {seen_corners[c]} and {p}")
            seen_corners[c] = p
        parts.append({f: d[f] for f in ARRAY_FIELDS})
        print(f"  {Path(p).name:28s} n={d['y_true'].size:9d}  corners={sorted(set(d['corner'].tolist()))}")

    merged = {f: np.concatenate([part[f] for part in parts]) for f in ARRAY_FIELDS}
    assert meta is not None
    for f, v in meta.items():
        merged[f] = np.array([v])

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **merged)
    print(f"\nmerged {len(parts)} dumps -> {out}")
    print(f"  n={merged['y_true'].size}  corners={len(seen_corners)}  "
          f"config={meta['meta_config']} stage={meta['meta_stage']} seeds={meta['meta_n_seeds']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output .npz path")
    ap.add_argument("inputs", nargs="+", help="per-corner .npz dumps to concatenate")
    a = ap.parse_args()
    main(a.out, a.inputs)
