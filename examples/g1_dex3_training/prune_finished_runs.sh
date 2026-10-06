#!/usr/bin/env bash
# Free disk from finished training runs: keep only the final checkpoint's weights.
#
# Usage: prune_finished_runs.sh [--dry-run] RUN_DIR [RUN_DIR ...]
# A run qualifies when <root>/logs/<name>.exit is 0 and <root>/logs/<name>.log contains "End of training"
# (<root> is the run's grandparent, <name> its basename). For a qualifying run it deletes every
# checkpoints/<step>/ other than the one `checkpoints/last` points to, and the final step's resume-only
# files (training_state/*.safetensors, *.pt). pretrained_model/ and pretrained_model_ema/ are kept,
# and so are the small training_state/*.json files. Other runs are skipped. Idempotent.
set -u
dry=0
[ "${1:-}" = "--dry-run" ] && dry=1 && shift
rm_() { if [ $dry = 1 ]; then echo "would delete $(du -sh "$1" | cut -f1) $1"; else du -sh "$1" | sed 's/^/deleted /'; rm -rf "$1"; fi; }
for run in "$@"; do
  run=${run%/}
  name=$(basename "$run")
  logs=$(dirname "$(dirname "$run")")/logs
  if [ "$(cat "$logs/$name.exit" 2>/dev/null)" != "0" ] || ! grep -aq "End of training" "$logs/$name.log" 2>/dev/null; then
    echo "skip (not finished): $run"
    continue
  fi
  final=$(readlink "$run/checkpoints/last")
  if [ -z "$final" ] || [ ! -d "$run/checkpoints/$final/pretrained_model" ]; then
    echo "skip (no final checkpoint): $run"
    continue
  fi
  for step in "$run"/checkpoints/*/; do
    step=${step%/}
    [ -L "$step" ] && continue
    [ "$(basename "$step")" = "$final" ] && continue
    rm_ "$step"
  done
  for f in "$run/checkpoints/$final"/training_state/*.safetensors "$run/checkpoints/$final"/training_state/*.pt; do
    [ -e "$f" ] && rm_ "$f"
  done
done
