#!/usr/bin/env bash
# Run one bench_release_ort configuration under an RSS watchdog.
#
# The TensorRT execution provider builds engines for the release pipeline graph
# in host memory; an unbounded run reached 68.8 GB RSS on this machine. This
# wrapper kills the run and records "too large" once RSS passes the budget, so
# the machine stays usable for other workers.
#
# Usage: run_release_ort.sh --height 1024 --width 1024 --pair-count 1
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RSS_BUDGET_KB=${RSS_BUDGET_KB:-$((60 * 1024 * 1024))}
LOG_DIR=${LOG_DIR:-/tmp/raco_speed}
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run-$(date +%s).log"

cd "$REPO_ROOT" || exit 1
pixi run -e bench python -m tools.audit.raco_speed.bench_release_ort "$@" >"$LOG_FILE" 2>&1 &
runner_pid=$!

runner_pgid=$(ps -o pgid= -p "$runner_pid" | tr -d ' ')

peak_kb=0
while kill -0 "$runner_pid" 2>/dev/null; do
    # `pixi run` forks, so sum the whole process group to catch the python child.
    # `ps -g` selects by *session*, not by process group, and silently returns
    # nothing here — it reported "peak RSS 0 MB" for every run and would never
    # have fired the budget. Match the pgid column explicitly instead.
    rss_kb=$(ps -eo pgid=,rss= | awk -v pgid="$runner_pgid" '$1 == pgid {total += $2} END {print total + 0}')
    if [ "$rss_kb" -gt "$peak_kb" ]; then peak_kb=$rss_kb; fi
    if [ "$rss_kb" -gt "$RSS_BUDGET_KB" ]; then
        echo "TOO LARGE: RSS ${rss_kb} kB exceeded budget ${RSS_BUDGET_KB} kB; killing" | tee -a "$LOG_FILE"
        # Kill the group members, not this watchdog, which shares the group.
        for victim in $(ps -eo pgid=,pid= | awk -v pgid="$runner_pgid" '$1 == pgid {print $2}'); do
            [ "$victim" != "$$" ] && kill -9 "$victim" 2>/dev/null
        done
        exit 2
    fi
    sleep 2
done
wait "$runner_pid"
status=$?
echo "peak group RSS: $((peak_kb / 1024)) MB   exit: $status   log: $LOG_FILE"
tail -n 6 "$LOG_FILE"
exit "$status"
