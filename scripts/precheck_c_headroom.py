"""Follow-up: (1) direction-C headroom via optimal multiplicative shading
of a median-ratio predictor on the near-zero group; (2) what fraction of
near-zero points live in tables that contain a sign change at all
(direction-B coverage)."""
import sys, numpy as np
from pathlib import Path

REPO = Path("/Users/jerrychen/My_Projects/Fast_Timing_Model_Esti_for_new_ PVT")
sys.path.insert(0, str(REPO / "src"))
from liberty.parser import parse_file  # noqa: E402

T = REPO / "testcase" / "training_set"
FILES = {
    "ss0p72vm40c": T / "base_nom_0p8v" / "lib1_ss0p72vm40c_base_400.tlib",
    "ss0p81vm40c": T / "base_nom_0p9v" / "lib1_ss0p81vm40c_base_400.tlib",
}
NZ = 1e-4

def load(corner):
    lf = parse_file(str(FILES[corner]))
    return {k[:4]: tb.values for k, tb in lf.tables_by_key.items()
            if k[4] == "fall_power" and tb.values is not None}

def group_score(e):
    return 100.0 * (1.0 - np.sqrt(np.mean(np.square(np.minimum(np.abs(e), 1.0)))))

fa, ft = load("ss0p81vm40c"), load("ss0p72vm40c")
keys = [k for k in ft if k in fa and fa[k].any() and ft[k].any()]

ya, yt, table_has_flip = [], [], []
for k in keys:
    a, t = fa[k].ravel(), ft[k].ravel()
    ya.append(a); yt.append(t)
    signs = np.sign(t[t != 0])
    table_has_flip.append(np.full(49, len(set(signs)) > 1))
ya, yt = np.concatenate(ya), np.concatenate(yt)
table_has_flip = np.concatenate(table_has_flip)

flip = ya * yt < 0
nz = (np.abs(yt) < NZ) & ~flip
print(f"near-zero points: {nz.sum()}; in mixed-sign tables: "
      f"{100*table_has_flip[nz].mean():.1f}%  <- direction-B max coverage")
print(f"sign-flip points: {flip.sum()}; in mixed-sign tables: "
      f"{100*table_has_flip[flip].mean():.1f}%")

# Direction C headroom: naive median-ratio predictor, then sweep a global
# multiplicative shading factor m (what a score-aware decision layer would
# effectively learn) on the near-zero group, oracle sign.
lr = np.log(np.abs(yt[nz]) / np.abs(ya[nz]))
base_pred = np.abs(ya[nz]) * np.exp(np.median(lr))
y = np.abs(yt[nz])
print("\nshading sweep on near-zero group (median-ratio base, oracle sign):")
best = None
for m in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25):
    s = group_score((y - m * base_pred) / y)
    tag = ""
    if best is None or s > best[1]:
        best = (m, s); tag = "  <-"
    print(f"  m={m:4.2f}: score={s:.2f}{tag}")
print(f"\nbulk reference: same sweep on bulk group")
bulk = ~flip & ~nz
lrb = np.log(np.abs(yt[bulk]) / np.abs(ya[bulk]))
bp = np.abs(ya[bulk]) * np.exp(np.median(lrb))
yb = np.abs(yt[bulk])
for m in (0.9, 0.95, 1.0, 1.05):
    print(f"  m={m:4.2f}: score={group_score((yb - m*bp)/yb):.2f}")
