"""(slew, load) coordinate-space alignment between corners' value tables.

docs/plan.md's Phase 2 design assumes index_2 (output load) is rescaled
per corner by the driven pin's max_capacitance, so grid-point-for-grid-point
transfer across corners would be wrong and an interpolation step is
needed before taking cross-corner log-ratios.

**Empirical finding** (see docs/phase2_results.md section "index_2 大方向
發現"): across all 15 released `.lib` files (5 full + 10 partial), for
every one of the 5804 shared table keys, ``index_1`` *and* ``index_2``
are byte-identical -- including across processes and voltages. The
documented per-corner load rescaling is not observed in this dataset
release. Grid alignment is therefore a no-op here.

We still implement real 1-D interpolation (index_1/slew is always
identical across corners per Phase 1, so only the load axis can ever
need it) so the pipeline is correct if a future release does rescale
index_2, and so the "座標空間插值對齊" deliverable named in the task is a
real, tested code path rather than an assumption.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def align_table_to_grid(values, src_index_2, dst_index_2) -> np.ndarray:
    """Return `values` (shape (n_rows, len(src_index_2))) resampled along
    its last axis from `src_index_2` onto `dst_index_2` coordinates.

    Uses linear interpolation with linear extrapolation outside the
    source range (rather than numpy.interp's edge-clamping) since a
    target corner's load grid could in principle extend past the source
    corner's range. Returns `values` unchanged (well, as a float array)
    when the two grids already match, which is the common case in this
    dataset.
    """
    src = np.asarray(src_index_2, dtype=float)
    dst = np.asarray(dst_index_2, dtype=float)
    values = np.asarray(values, dtype=float)
    if src.shape == dst.shape and np.array_equal(src, dst):
        return values
    f = interp1d(src, values, axis=-1, kind="linear", fill_value="extrapolate", assume_sorted=True)
    return f(dst)
