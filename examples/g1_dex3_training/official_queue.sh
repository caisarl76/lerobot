#!/usr/bin/env bash
# Run a per-GPU training queue inside a training container, one job at a time.
#
# Usage (in the container): official_queue.sh QUEUE_FILE [WAIT_FOR_FILE ...]
#   QUEUE_FILE  one job per line, "<root>/<name>" with <root> relative to $BASE (e.g. "." for the
#               Unitree runs, "humanoid_everyday_g1_20260923" for HE); the config is
#               <root>/configs/<name>.json and the log <root>/logs/<name>.{log,exit}. The file is
#               re-read after every job, so lines can be appended while the queue runs. Blank lines
#               and lines starting with # are skipped.
#   WAIT_FOR_FILE  optional files (e.g. .exit files of jobs started outside the queue); the queue
#               starts once all of them exist.
# A job whose .exit file exists is done. A *_full job is skipped (exit "skip") when its *_smoke job
# exited non-zero.
set -u
Q=$1
shift
BASE=${BASE:-/run-output}
PY=${PY:-/run-output/environment/venv/bin/python}
for f in "$@"; do
  while [ ! -f "$f" ]; do sleep 60; done
done
while true; do
  next=""
  while read -r job; do
    [[ -z "$job" || "$job" == \#* ]] && continue
    [ -f "$BASE/$(dirname "$job")/logs/$(basename "$job").exit" ] && continue
    next=$job
    break
  done <"$Q"
  [ -z "$next" ] && break
  root=$BASE/$(dirname "$next")
  name=$(basename "$next")
  smoke_exit="$root/logs/${name%_full}_smoke.exit"
  if [[ "$name" == *_full && -f "$smoke_exit" && "$(cat "$smoke_exit")" != "0" ]]; then
    echo skip >"$root/logs/$name.exit"
    continue
  fi
  # A container can lose its GPU (NVML "Unknown Error" after a host cgroup reset); LeRobot would then
  # silently train on the CPU. Stop the queue instead; recreate the container and restart the runner.
  if ! "$PY" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    echo "no CUDA in this container; queue stopped before $next" >&2
    exit 2
  fi
  "$PY" -m lerobot.scripts.lerobot_train --config_path="$root/configs/$name.json" >"$root/logs/$name.log" 2>&1
  echo $? >"$root/logs/$name.exit"
done
