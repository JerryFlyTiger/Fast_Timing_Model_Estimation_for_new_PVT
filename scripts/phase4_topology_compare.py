"""Reproduce the cross-topology numbers in docs/round_20260810.md section 8.

Section 8's derived figures were computed ad hoc while the round was being
written. docs/current_status.md section 8 lesson 1 is about exactly that
failure mode -- a calculation that lives only in a session transcript
disappears, and the rebuild discovers the method had freedom the prose
never pinned down. So this file exists to make section 8 re-runnable.

It reproduces:

  8.2/8.3  pooled score per topology, recomputed from the merged error
           dumps rather than read off the per-corner logs
  8.3      each topology's subgroup share of squared-error mass (the flip
           row: 63.4 / 69.2 / 71.8 %)
  8.4      the paired direction statistics -- one-step opposite-direction
           (beta vs final on the same nominal corners) and the marginal
           cost of the second step (alpha's one-step targets are beta's
           and final's two-step targets)
  8.8      the back-solve: what the two unfinished corners would have to
           score for pooled(10) to land in each pre-registered band
           (skipped once the control is complete -- it was scaffolding)
  8.10     the verdict on 8.7's pre-registered mechanism claim: Huber's
           gain per topology against the tail-mass proxy it was supposed
           to track, plus the corner-paired gain differences between
           topologies

The back-solve is the reason this file matters most. The contest score is
100*(1 - sqrt(mean e^2)), so pooling is linear in mean SQUARED error, not
in per-corner scores. Re-deriving it from per-corner means gives a
materially different answer (+0.24 rather than +0.307 -- an error made
once already, by hand, while writing 8.8).

Usage:
    python3 scripts/phase4_topology_compare.py
    python3 scripts/phase4_topology_compare.py --cache-dir output/_phase4_cache
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Dict, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scoring.audits import assign_subgroups, subgroup_stats  # noqa: E402
from scoring.scorer import point_errors, score_from_errors  # noqa: E402

# The merged huber dumps, one per topology (section 8.3's table). The
# alpha and beta ones predate this round; final's was merged on
# 2026-08-13 from output/_phase4_cache/final_huber_s1/.
MERGED = {
    "alpha": "alpha_validate_huber_s1_errors.npz",
    "beta": "beta_validate_huber_s1_errors.npz",
    "final": "final_validate_huber_s1_errors.npz",
}

# Per-corner dump directories for the huber/mse pair on the final
# topology (section 8.8). The control is incomplete by design -- two
# corners were abandoned after eight environment aborts, see 8.9.
FINAL_HUBER_DIR = "final_huber_s1"
FINAL_CONTROL_DIR = "final_control_s1"

# Section 8.10 needs the huber/mse pair on ALL THREE topologies, but only
# final's runs left per-corner error dumps; the alpha and beta pairs exist
# only as the per-corner run logs, which are tracked in git. Parsing those
# is what makes 8.10 re-runnable instead of a table typed out by hand.
# Each entry is (log directory, filename prefix).
RUN_LOGS = {
    ("alpha", "mse"): ("round_20260810", "control"),
    ("alpha", "huber"): ("round_20260811", "alpha_huber"),
    ("beta", "mse"): ("round_20260811", "beta_control"),
    ("beta", "huber"): ("round_20260811", "beta_huber"),
    ("final", "mse"): ("round_20260811", "final_control"),
    ("final", "huber"): ("round_20260811", "final_huber"),
}

# Topologies share exactly 5 target corners pairwise (8.4), so every
# cross-topology paired test here has df=4. Spelled out because |t|>2
# reads as "significant" only when df is large, and it is not here.
T_CRIT_05_DF4 = 2.776

SCORE_HEADER = "=== summary: per-corner overall scores ==="
SCORE_RE = re.compile(r"^ {2}(\S+) +([0-9]+\.[0-9]+)\s*$")

# The three pre-registered verdict bands from section 8.7. They do NOT
# tile the line -- that gap is a recorded flaw, not an oversight here
# (8.8), so it is reproduced as written rather than silently patched.
BANDS = (
    (0.110, "機制成立"),
    (0.080, "無資訊帶下緣"),
    (0.060, "推翻門檻"),
)


def load_merged(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return d["y_true"], d["y_pred"], d["corner"], d["nearest_anchor"], d["table_type"]


def load_per_corner(directory: str) -> Dict[str, np.ndarray]:
    """corner name -> point errors, from one dump file per corner."""
    out = {}
    for f in sorted(glob.glob(os.path.join(directory, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        out[os.path.basename(f)[:-4]] = point_errors(z["y_true"], z["y_pred"])
    return out


def scores_from_logs(logs_root: str, subdir: str, prefix: str) -> Dict[str, float]:
    """corner -> overall score, read from one run log per corner.

    Only the per-corner summary block is read. A log without that block
    is an aborted run (section 8.9) and is skipped, which is the same
    success test run_corner_sweep.sh uses.
    """
    out: Dict[str, float] = {}
    pattern = os.path.join(logs_root, subdir, f"{prefix}_*.log")
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
        try:
            start = lines.index(SCORE_HEADER)
        except ValueError:
            continue
        for line in lines[start + 1:]:
            if not line.strip():
                break
            m = SCORE_RE.match(line)
            if m:
                out[m.group(1)] = float(m.group(2))
    return out


def pool(scores) -> float:
    """Pool per-corner scores. Every target corner carries the same point
    count (196588), so pooled mean e^2 is the plain mean of the per-corner
    mean e^2 -- never the mean of the scores themselves."""
    return 100.0 - 100.0 * float(np.sqrt(np.mean([mean_sq(s) for s in scores])))


def mean_sq(score: float) -> float:
    """Invert score = 100*(1 - sqrt(mean e^2)). Pooling is linear here,
    not in the score itself -- the whole point of the back-solve."""
    return ((100.0 - score) / 100.0) ** 2


def paired(name: str, deltas: np.ndarray) -> None:
    se = deltas.std(ddof=1) / np.sqrt(deltas.size)
    print(f"  {name:36s} 平均 {deltas.mean():+.4f}  sd {deltas.std(ddof=1):.4f}  "
          f"SE {se:.4f}  t {deltas.mean()/se:+.2f}  n={deltas.size}")


def preregistered_verdict(final_gain: float, top_stage: str,
                          threshold: float = BANDS[0][0]) -> str:
    """Rule on 8.7's claim, which is a CONJUNCTION: final's gain is
    >= +0.11 AND is the largest of the three topologies.

    Split out of verdict() so it can be tested. The first version
    checked only the "largest" conjunct, so a future re-run landing at,
    say, +0.09-but-largest would have been reported as upholding the
    claim -- exactly the after-the-fact latitude a pre-registration
    exists to remove. That slip survived a cold read of verdict() as a
    whole; it is small enough to only be visible on its own.
    """
    why = ([] if final_gain >= threshold else
           [f"{final_gain:+.4f} < {threshold:+.2f}"])
    if top_stage != "final":
        why.append(f"最大的是 {top_stage}")
    ruling = "成立" if not why else "不成立（" + "；".join(why) + "）"
    return f"預測（8.7）：final ≳ {threshold:+.2f} 且為三者最大 → {ruling}"


def verdict(logs_root: str, huber_ref: Dict[str, Dict[str, float]],
            mse_ref: Dict[str, np.ndarray], flip_share: Dict[str, float]) -> int:
    """8.10 -- rule on 8.7's pre-registered claim now that final's control
    is complete. The claim: the Huber gain grows with topology difficulty
    / residual tail weight, proxied by the flip subgroup's share of e^2.
    Predicted final >= +0.11 AND largest of the three."""
    print("\n=== 8.10 機制判讀：Huber 增益 vs 尾巴質量（8.7 預先登記）===")
    # Two implicit preconditions, neither enforceable from inside this
    # function, both recorded because a cold read found them:
    #  - the mse side of alpha/beta has no independent cross-check; only
    #    final's can be checked against its dumps (done below), because
    #    only final's mse run left per-corner .npz files.
    #  - the RUN_LOGS globs also match sweep-driver and sensitivity logs
    #    in the same directories. They are harmless only because no other
    #    script prints SCORE_HEADER. The n != 10 guard catches an extra
    #    or a missing corner, but NOT a same-named overwrite.
    runs: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, (subdir, prefix) in RUN_LOGS.items():
        runs[key] = scores_from_logs(logs_root, subdir, prefix)
        if len(runs[key]) != 10:
            print(f"  {key} 只有 {len(runs[key])}/10 個 corner 的 log，跳過本節。")
            return 1

    # The huber side exists twice over -- parsed from logs here, and
    # recomputed from the merged dumps in 8.2. They must agree, or one of
    # the two paths is reading something other than what it claims to.
    # Compare the corner SETS before the values. The dump side is a
    # gitignored hand-managed cache; one stray .npz there would
    # otherwise make this a bare KeyError instead of a diagnosis.
    for stage in ("alpha", "beta", "final"):
        log_side, dump_side = runs[(stage, "huber")], huber_ref[stage]
        if set(log_side) != set(dump_side):
            print(f"  huber 的 corner 集合 log 與 dump 不一致（{stage}）："
                  f"僅 log {sorted(set(log_side) - set(dump_side))}、"
                  f"僅 dump {sorted(set(dump_side) - set(log_side))}，中止。")
            return 1
        gap = max(abs(log_side[c] - dump_side[c]) for c in dump_side)
        if gap > 5e-4:
            print(f"  log 與 dump 的 huber 分數不符（{stage}，最大差 {gap:.4f}），中止。")
            return 1
    print("  [check] 三拓樸的 huber 逐 corner 分數，log 與 merged dump 相符 (<5e-4)")

    # final is the one topology whose mse control also left per-corner
    # dumps, so its mse column gets the same treatment. alpha's and
    # beta's mse columns remain log-only -- stated, not silently assumed.
    ctl_log = runs[("final", "mse")]
    if set(ctl_log) != set(mse_ref):
        print(f"  final mse 的 corner 集合 log 與 dump 不一致："
              f"僅 log {sorted(set(ctl_log) - set(mse_ref))}、"
              f"僅 dump {sorted(set(mse_ref) - set(ctl_log))}，中止。")
        return 1
    ctl_gap = max(abs(ctl_log[c] - score_from_errors(e)) for c, e in mse_ref.items())
    if ctl_gap > 5e-4:
        print(f"  log 與 dump 的 final mse 分數不符（最大差 {ctl_gap:.4f}），中止。")
        return 1
    print("  [check] final 的 mse 逐 corner 分數，log 與 dump 相符 (<5e-4)；"
          "alpha/beta 的 mse 側只有 log，無獨立來源")

    gains: Dict[str, Dict[str, float]] = {}
    print(f"\n  {'拓樸':6s} {'翻轉佔e²':>9s} {'mse':>9s} {'huber':>9s} {'Δ':>9s}   逐corner")
    for stage in ("alpha", "beta", "final"):
        mse, hub = runs[(stage, "mse")], runs[(stage, "huber")]
        gains[stage] = {c: hub[c] - mse[c] for c in mse}
        d = np.array(list(gains[stage].values()))
        s_m, s_h = pool(mse.values()), pool(hub.values())
        print(f"  {stage:6s} {flip_share[stage]:8.1f}% {s_m:9.4f} {s_h:9.4f} "
              f"{s_h - s_m:+9.4f}   平均 {d.mean():+.4f} ± {d.std(ddof=1):.4f} "
              f"(SE {d.std(ddof=1)/np.sqrt(d.size):.4f}), {(d > 0).sum()}/10 為正")

    # Topologies share corners pairwise (8.4), so the gains can be
    # compared paired-by-corner instead of as two independent samples --
    # that removes the corner-to-corner spread, which is the dominant
    # variance here.
    print("\n  增益的逐 corner 配對比較（共同 target）：")
    for a, b in (("final", "beta"), ("final", "alpha"), ("beta", "alpha")):
        shared = sorted(set(gains[a]) & set(gains[b]))
        paired(f"{a} − {b} 的 Δ  (共同 {len(shared)} corner)",
               np.array([gains[a][c] - gains[b][c] for c in shared]))
    print(f"  （n=5 → df=4，雙尾 α=.05 的臨界 t 是 {T_CRIT_05_DF4}。|t|>2 在這裡"
          "**不是**顯著門檻。）")

    pooled_gain = {s: pool(runs[(s, "huber")].values()) - pool(runs[(s, "mse")].values())
                   for s in gains}
    order_mass = sorted(flip_share, key=lambda s: flip_share[s])
    order_gain = sorted(pooled_gain, key=lambda s: pooled_gain[s])
    print(f"\n  尾巴質量由小到大: {' < '.join(order_mass)}")
    print(f"  增益   由小到大: {' < '.join(order_gain)}")

    print("  " + preregistered_verdict(pooled_gain["final"], order_gain[-1]))
    return 0


def main(cache_dir: str, logs_root: str) -> int:
    # ---- 8.2 / 8.3: pooled score and subgroup mass per topology --------
    print("=== 8.2/8.3 三拓樸 pooled 分數與子群 e² 質量 (huber, 1-seed) ===")
    per_corner: Dict[str, Dict[str, float]] = {}
    flip_share: Dict[str, float] = {}
    missing = []
    for stage, fname in MERGED.items():
        path = os.path.join(cache_dir, fname)
        if not os.path.exists(path):
            missing.append(path)
            continue
        yt, yp, corner, anchor, table = load_merged(path)
        e = point_errors(yt, yp)
        stats = subgroup_stats(e, assign_subgroups(yt, anchor, table))
        total = sum(s.e2_mass for s in stats)
        flip = next(s for s in stats if s.name == "fall_power:flip")
        flip_share[stage] = 100 * flip.e2_mass / total
        print(f"  {stage:6s} pooled={score_from_errors(e):.4f}  "
              f"翻轉佔 e² {flip_share[stage]:.1f}%  (n={e.size})")
        per_corner[stage] = {
            c: score_from_errors(e[corner == c]) for c in sorted(set(corner.tolist()))
        }
    if missing:
        print("  缺少 dump（跳過相關區段）:")
        for m in missing:
            print(f"    {m}")
    if not {"alpha", "beta", "final"} <= per_corner.keys():
        print("\n三個拓樸的 dump 不齊，8.4 與 8.8 需要全部三個。")
        return 1

    alpha, beta, final = per_corner["alpha"], per_corner["beta"], per_corner["final"]

    # ---- 8.4: paired direction statistics -----------------------------
    # beta targets one step DOWN onto the nominal corners; final targets
    # the same corners one step UP. alpha's one-step targets are exactly
    # beta's and final's two-step targets, which gives the marginal cost
    # of the second step on an identical corner.
    nominal = sorted(set(beta) & set(final))
    buck = sorted(set(alpha) & set(beta))
    boost = sorted(set(alpha) & set(final))
    print(f"\n=== 8.4 配對統計 ===")
    print(f"  一步、方向相反的共同 target: {nominal}")
    paired("final(一步上) − beta(一步下)", np.array([final[c] - beta[c] for c in nominal]))
    print(f"  兩步 target（alpha 為一步）: 降壓 {buck} / 升壓 {boost}")
    d_dn = np.array([beta[c] - alpha[c] for c in buck])
    d_up = np.array([final[c] - alpha[c] for c in boost])
    paired("第二步（向下）beta − alpha", d_dn)
    paired("第二步（向上）final − alpha", d_up)
    se = np.sqrt(d_dn.var(ddof=1) / d_dn.size + d_up.var(ddof=1) / d_up.size)
    print(f"  {'上 − 下':36s} 差 {d_up.mean()-d_dn.mean():+.4f}  SE {se:.4f}  "
          f"t {(d_up.mean()-d_dn.mean())/se:+.2f}")
    print(f"  半邊平均: 一步 beta={np.mean([beta[c] for c in nominal]):.4f} "
          f"final={np.mean([final[c] for c in nominal]):.4f} | "
          f"兩步 beta={np.mean([beta[c] for c in buck]):.4f} "
          f"final={np.mean([final[c] for c in boost]):.4f}")

    # ---- 8.8: the huber/mse pair on final, and the back-solve ---------
    hub = load_per_corner(os.path.join(cache_dir, FINAL_HUBER_DIR))
    ctl = load_per_corner(os.path.join(cache_dir, FINAL_CONTROL_DIR))
    if not hub or not ctl:
        print("\n=== 8.8 === 缺少 final 的逐 corner dump，跳過。")
        return 0
    shared = sorted(set(hub) & set(ctl))
    absent = sorted(set(hub) - set(ctl))
    print(f"\n=== 8.8 final: huber vs mse（配對 {len(shared)}/{len(hub)} corner）===")
    for c in shared:
        a, b = score_from_errors(ctl[c]), score_from_errors(hub[c])
        print(f"  {c:13s} mse {a:8.4f}  huber {b:8.4f}  Δ {b-a:+.4f}")
    ds = np.array([score_from_errors(hub[c]) - score_from_errors(ctl[c]) for c in shared])
    s_ctl = score_from_errors(np.concatenate([ctl[c] for c in shared]))
    s_hub = score_from_errors(np.concatenate([hub[c] for c in shared]))
    print(f"  {'pooled':13s} mse {s_ctl:8.4f}  huber {s_hub:8.4f}  Δ {s_hub-s_ctl:+.4f}")
    print(f"  逐 corner 平均 {ds.mean():+.4f} ± {ds.std(ddof=1):.4f}, "
          f"{(ds > 0).sum()}/{ds.size} 為正")

    if not absent:
        print("\n  對照組已完整（2026-08-14 補完），8.8 的回推不再適用"
              "——直接引用上面的 pooled Δ。")
        return verdict(logs_root, per_corner, ctl, flip_share)

    # Back-solve. Corners carry equal point counts, so pooled mean e^2 is
    # the plain average of the per-corner mean e^2 -- but of the SQUARED
    # error, never of the scores.
    n_all = len(hub)
    n_have = len(shared)
    n_gap = len(absent)
    hub_all = score_from_errors(np.concatenate([hub[c] for c in sorted(hub)]))
    hub_gap = score_from_errors(np.concatenate([hub[c] for c in absent]))
    m_have = mean_sq(s_ctl)
    print(f"\n  尚缺 {absent}（huber 側 pooled {hub_gap:.4f}）")
    print(f"  要讓 pooled({n_all}) 的 Δ 落在各判讀帶，缺的 {n_gap} 個 corner 需要：")
    for target, label in BANDS:
        m_all = mean_sq(hub_all - target)
        m_gap = (n_all * m_all - n_have * m_have) / n_gap
        if m_gap <= 0:
            print(f"    Δ={target:+.3f} ({label}): 不可能（已被前 {n_have} 個鎖死）")
            continue
        s_gap = 100.0 - 100.0 * float(np.sqrt(m_gap))
        print(f"    Δ={target:+.3f} ({label:12s}): mse pooled {s_gap:.4f}"
              f"  即 Δ {hub_gap - s_gap:+.4f}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="output/_phase4_cache",
                    help="誤差 dump 所在目錄（預設 output/_phase4_cache）")
    ap.add_argument("--logs-root", default="logs",
                    help="逐 corner 執行 log 的根目錄（8.10 用，預設 logs）")
    _a = ap.parse_args()
    raise SystemExit(main(_a.cache_dir, _a.logs_root))
