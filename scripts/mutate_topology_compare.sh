#!/bin/bash
# Mutation matrix for scripts/phase4_topology_compare.py.
#
# That script rules on a pre-registered hypothesis (docs/round_20260810.md
# section 8.10), so "the tests pass" is not the interesting question --
# "would the tests notice if it were wrong" is. Each mutation below is a
# way the script has actually been wrong, or was one cold read away from
# being wrong. All nine must turn the suite red.
#
# CACHE TRAP (docs/current_status.md section 8 lesson 6): this
# interpreter puts bytecode OUTSIDE the repo
# (sys.pycache_prefix = ~/Library/Caches/com.apple.python), so
# `find . -name __pycache__` does not clear it, and invalidation is by
# (source mtime, source size). A mutation that keeps the byte length
# identical -- several below do, they only move a character -- applied
# and reverted inside the same second is INVISIBLE, and the run silently
# reports the pristine result. That produced one false failure already.
# Hence: delete the pyc before every run, and pass -B.
#
# Usage:  scripts/mutate_topology_compare.sh
set -u

cd "$(dirname "$0")/.." || exit 1
S=scripts/phase4_topology_compare.py
BAK="$(mktemp -t tc_pristine)"
CACHE="$(.venv/bin/python3 -c 'import sys, os
p = sys.pycache_prefix
print(os.path.join(p, os.path.abspath("scripts").lstrip("/")) if p else "scripts/__pycache__")')"
PYC_GLOB="$CACHE/phase4_topology_compare.cpython-*.pyc"

cp "$S" "$BAK"
# The pristine copy is the only way back: a mutated script left behind
# would be a silently wrong verdict next time someone runs it.
trap 'cp "$BAK" "$S"; rm -f "$BAK"' EXIT

run() {
    rm -f $PYC_GLOB
    .venv/bin/python3 -B -m pytest tests/test_topology_compare.py -q \
        -p no:cacheprovider 2>&1 | tail -1
}

n_survived=0
mutate() {  # <description> <perl expression>
    printf '%-56s ' "$1"
    perl -0pi -e "$2" "$S"
    if cmp -s "$S" "$BAK"; then
        # A pattern that stopped matching after a refactor would
        # otherwise look like a passing mutation run.
        echo "!! MUTATION DID NOT APPLY (pattern miss) !!"
        n_survived=$((n_survived + 1))
    else
        result="$(run)"
        echo "$result"
        case "$result" in *failed*) ;; *) n_survived=$((n_survived + 1)) ;; esac
    fi
    cp "$BAK" "$S"
}

printf '%-56s ' "baseline (pristine)"; run

mutate "M1 拿掉區塊結束的空行 break" \
    's/            if not line\.strip\(\):\n                break\n//'
mutate "M2 RUN_LOGS 的 alpha/mse 指向 beta 對照組" \
    's/\("alpha", "mse"\): \("round_20260810", "control"\)/("alpha", "mse"): ("round_20260811", "beta_control")/'
mutate "M3 pool() 改成直接平均分數" \
    's/    return 100\.0 - 100\.0 \* float\(np\.sqrt\(np\.mean\(\[mean_sq\(s\) for s in scores\]\)\)\)/    return float(np.mean(list(scores)))/'
mutate "M4 SCORE_RE 的 ^ {2} 放寬成 ^ +" \
    's/\^ \{2\}\(\\S\+\) \+/^ +(\\S+) +/'
mutate "M5 量值門檻恆真（第一版的原始 bug）" \
    's/    why = \(\[\] if final_gain >= threshold else\n           \[f"\{final_gain:\+\.4f\} < \{threshold:\+\.2f\}"\]\)/    why = []/'
mutate "M6 「三者最大」恆真" \
    's/    if top_stage != "final":/    if False:/'
mutate "M7 負增益的符號改回手動加 +" \
    's/\[f"\{final_gain:\+\.4f\} < \{threshold:\+\.2f\}"\]/[f"+{final_gain:.4f} < +{threshold:.2f}"]/'
mutate "M8 final mse 交叉驗證的閾值放寬到吞掉一切" \
    's/    if ctl_gap > 5e-4:/    if ctl_gap > 1e9:/'
mutate "M9 huber 交叉驗證的閾值放寬到吞掉一切" \
    's/        if gap > 5e-4:/        if gap > 1e9:/'

printf '%-56s ' "restored (pristine)"; run
if [ "$n_survived" -gt 0 ]; then
    echo "[FAIL] $n_survived 個 mutation 存活——測試沒有釘住那些行為"
    exit 1
fi
echo "[ ok ] 9 個 mutation 全部被測試抓到"
