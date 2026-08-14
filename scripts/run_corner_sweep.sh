#!/bin/bash
# Run scripts/phase4_final_validate.py over one stage's 10 target corners,
# ONE CORNER PER INVOCATION.
#
# Why one at a time: docs/round_20260810.md 4.1 -- batch runs of the huber
# config get killed non-deterministically, single-corner runs succeed.
# Aborts leave no log (scores are written once at the end), so the
# presence of the score marker in the log is the success test, and each
# corner gets up to 3 attempts.
#
# Usage:
#   scripts/run_corner_sweep.sh <stage> <config> <log-prefix> [dump-dir]
#
# Invocations (all logs kept under logs/round_20260811/):
#   beta  mlp_w256_b4_huber beta_huber   output/_phase4_cache/beta_huber_s1
#       -> the beta-topology measurement, docs/round_20260810.md 7.4
#   alpha mlp_w256_b4_huber alpha_huber  output/_phase4_cache/alpha_huber_s1
#       -> gives the current default config an alpha error dump, so the
#          official-composition audit can run on huber instead of mse
#   beta  mlp_w256_b4_full  beta_control
#       -> re-runs the 2026-07-28 mse baseline under current code, so
#          section 7.5's +0.111 rests on two same-code runs rather than on
#          extrapolating alpha's bit-identity check to the beta topology.
#          NOTE: the first attempt at this one (2026-08-11 20:26) never
#          ran a single corner -- it is the no-dump-dir path, which hit
#          the bash 3.2 empty-array bug fixed below.
set -u

cd "$(dirname "$0")/.." || exit 1
# Overridable so the argument handling can be smoke-tested without
# starting a 5-minute training run: PY=echo scripts/run_corner_sweep.sh ...
PY="${PY:-.venv/bin/python3}"
STAGE="${1:?usage: run_corner_sweep.sh <stage> <config> <log-prefix> [dump-dir]}"
CONFIG="${2:?missing config}"
PREFIX="${3:?missing log prefix}"
DUMPDIR="${4:-}"

LOGDIR=logs/round_20260811
mkdir -p "$LOGDIR"
[ -n "$DUMPDIR" ] && mkdir -p "$DUMPDIR"

# Ask the topology for its target corners rather than hardcoding them --
# the alpha and beta target sets differ and are easy to transpose by hand.
# Always the real interpreter: this is a metadata query, not the
# workload, so it must still work when PY is overridden for a smoke test.
CORNERS=$(.venv/bin/python3 -c "
import sys; sys.path.insert(0, 'src')
from models.phase4_features import STAGE_TOPOLOGIES
print(' '.join(STAGE_TOPOLOGIES['$STAGE'].target_names))
") || exit 1

MARKER="=== pooled overall across"
echo "[plan] stage=$STAGE config=$CONFIG corners: $CORNERS"

n_ok=0
n_failed=0
failed_corners=""

for corner in $CORNERS; do
    log="$LOGDIR/${PREFIX}_${corner}.log"
    dump_args=()
    if [ -n "$DUMPDIR" ]; then
        dump_args=(--dump-errors "$DUMPDIR/${corner}.npz")
    fi
    if grep -q "$MARKER" "$log" 2>/dev/null; then
        echo "[skip] $corner already done"
        n_ok=$((n_ok + 1))
        continue
    fi
    corner_ok=0
    for attempt in 1 2 3; do
        echo "[run ] $corner attempt $attempt $(date +%H:%M:%S)"
        # "${dump_args[@]+...}" rather than a bare "${dump_args[@]}":
        # macOS ships bash 3.2, where expanding an EMPTY array under
        # `set -u` is an unbound-variable error that kills the script.
        # That bug silently skipped an entire 10-corner sweep on
        # 2026-08-11 (the no-dump-dir invocation died in 0 seconds and
        # the caller read the surviving next phase as "it's running").
        "$PY" scripts/phase4_final_validate.py \
            --stage "$STAGE" --config "$CONFIG" --seeds 1 \
            --corners "$corner" ${dump_args[@]+"${dump_args[@]}"} > "$log" 2>&1
        if grep -q "$MARKER" "$log"; then
            echo "[ ok ] $corner  $(grep -E "^  ${corner} " "$log" | tail -1)"
            corner_ok=1
            break
        fi
        echo "[fail] $corner attempt $attempt (silent abort, retrying)"
        [ "$attempt" = 3 ] && echo "[GIVE UP] $corner after 3 attempts"
    done
    if [ "$corner_ok" = 1 ]; then
        n_ok=$((n_ok + 1))
    else
        n_failed=$((n_failed + 1))
        failed_corners="$failed_corners $corner"
    fi
done

# A sweep that gave up on some corners must not look like a clean run to
# whatever chains merge/audit after it.
echo "[done] $STAGE/$CONFIG $(date +%H:%M:%S) -- $n_ok ok, $n_failed failed"
if [ "$n_failed" -gt 0 ]; then
    echo "[FAILED CORNERS]$failed_corners"
    exit 1
fi
