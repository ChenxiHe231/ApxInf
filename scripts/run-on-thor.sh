#!/bin/bash
# Serialize GPU work on Thor-U and refuse to start when it cannot succeed.
#
# Written after a run of self-inflicted failures:
#   - two engines competing for the one 40 GB hugepage pool, which surfaced as
#     nvmap "alloc_handle_from_va failed -12" and as TensorRT reporting a
#     corrupt engine when the real problem was no free pages;
#   - a diagnostic launched while a matrix was running, which OOMed itself;
#   - `pkill -f <script>` killing the bench child while the parent loop
#     promptly started the next configuration, so nine results were recorded as
#     null and the memory was never actually released;
#   - orphaned processes holding 29 GB indefinitely.
#
# usage: run-on-thor.sh <tag> <command...>
set -u
LOCK=/tmp/apxinf-gpu.lock
TAG=${1:?usage: run-on-thor.sh <tag> <command...>}
shift
NEED_PAGES=${NEED_PAGES:-13000}          # ~26 GB, enough for weights plus KV
WAIT_IDLE=${WAIT_IDLE:-1}                # 0 to fail fast instead of waiting

note() { echo "[run-on-thor:$TAG] $*" >&2; }

# Processes that are not ours but will fight us for the pool. The lock only
# binds participants, so a non-participant has to be waited out.
foreign() {
  pgrep -af "qwen35_mtp_bench|qwen35_aios|nsys|experimental.serve[r]" 2>/dev/null \
    | grep -v "run-on-thor" | grep -v "^$$ " || true
}

free_pages() { awk '/HugePages_Free/{print $2}' /proc/meminfo; }

preflight() {
  local waited=0
  while :; do
    local f; f=$(foreign)
    local p; p=$(free_pages)
    if [ -z "$f" ] && [ "$p" -ge "$NEED_PAGES" ]; then
      note "idle, HugePages_Free=$p, starting"
      return 0
    fi
    if [ "$WAIT_IDLE" = "0" ]; then
      note "busy (free=$p need=$NEED_PAGES); foreign: $(echo "$f" | head -1)"
      return 1
    fi
    [ $((waited % 300)) -eq 0 ] && \
      note "waiting: free=$p need=$NEED_PAGES foreign=$(echo "$f" | wc -l)"
    sleep 30; waited=$((waited + 30))
  done
}

# flock holds for the whole command, so a long-lived server and a short bench
# cannot overlap. setsid puts the payload in its own process group, so it can
# be stopped as a unit instead of leaving a loop to respawn children.
exec 9>"$LOCK"
note "acquiring $LOCK"
flock 9 || { note "could not lock"; exit 1; }
preflight || exit 1

setsid "$@" &
PAYLOAD=$!
echo "$PAYLOAD" > /tmp/apxinf-gpu.pgid
note "pgid=$PAYLOAD (stop with: kill -- -$PAYLOAD)"
trap 'note "signalled, stopping group $PAYLOAD"; kill -- -"$PAYLOAD" 2>/dev/null' INT TERM
wait "$PAYLOAD"; RC=$?
note "exit=$RC; HugePages_Free=$(free_pages)"
# A payload that exits without releasing pages means an orphan survived.
LEFT=$(foreign)
[ -n "$LEFT" ] && note "WARNING orphan survived: $(echo "$LEFT" | head -2)"
exit $RC
