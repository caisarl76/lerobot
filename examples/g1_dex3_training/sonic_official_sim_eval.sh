#!/bin/bash
# Official NVIDIA-deploy evaluation of one policy on one recorded episode, run from a workstation over SSH on the
# H100 host ($SIM_HOST, default alias "h100"; everything runs in containers there). Same split as the real robot: a CPU-only streamer
# asks sonic_policy_server.py (GPU) for chunks; the sim host adds an optional table (TABLE_GAP_CM) once the robot
# stands in planner mode. Scores the run with sonic_stream_eval.py; see sonic_stream_phases.py for per-phase table
# clearance and balance.
#
# Usage: sonic_official_sim_eval.sh OUT RUN_NAME DATASET EPISODE GPU TABLE_GAP_CM [extra streamer args...]
#   e.g. sonic_official_sim_eval.sh FINAL_groot_ep6 groot_sonic78nolimit_ho5_full sonic78_nolimit 6 2 5 \
#          --startup-tokens /audit/table_startup/r20/startup.npz
# Needs: /audit/code on H100 holding this directory's current scripts (copy them there through a container).
# Env: SIM_HOST (ssh alias of the GPU host; default h100), HF_CACHE (host HF cache mounted for the policy server and streamer, default /mnt/data01/huggingface; on h100_174 use /mnt/data01/jhkim/huggingface, which holds CLIP and Qwen3-VL for Psi0/XR-1), LOG_DIR (local streamer logs, default ./sim_eval_logs), CODE_DIR (H100 script copy, default $A/code), JOINT28_DIR (joint28 dataset for scoring; HE: /run-output/humanoid_everyday_g1_20260923/datasets/joint28), SERVER_EXTRA (extra policy server args, e.g. --noise-seed 0 or --noise-scale 0), POLICY_SUBDIR (checkpoint subfolder, e.g. pretrained_model_ema), VPN_ON (command to re-establish the VPN, optional), BACKEND (sonic | decoupled; default sonic), HOST_ARGS (extra host args, decoupled only, e.g. --waist-location lower_and_upper_body), REPLAY (1: stream the episode's recorded actions; no policy server, RUN_NAME is ignored), SONIC_DIR (stored-token dataset for scoring; default the episode dataset), LEROBOT_DIR (lerobot code mounted at /workspace/lerobot for the policy server and streamer; default the official-recipe copy lerobot-g1-official-20260929-ceb40b77).
if [ "${SIM:-mujoco}" = isaac ]; then
  # Isaac mode (closed-loop handover sessions, docs/research/2026-10-17-g1-isaac-handover-eval.md): Isaac + deploy +
  # streamer on this workstation, one full stack per trial (the deploy's planner keeps its heading across a reset), each
  # in a private network namespace (its own lo: no DDS cross-talk with other sims on the machine) with a fresh gate dir;
  # the policy server on $SIM_HOST GPU $GPU in a container, reached over an ssh tunnel bridged into each namespace
  # through a unix socket. Streamer flags are a real robot session's args (ROBOT_ARGS: its trial_000.npz) except the
  # policy server, gate / session dirs, the trial count, --dex3-right-order swap (the sim bridge's right-hand order) and
  # --policy-timeout-s 5.
  # Usage: SIM=isaac TRIALS=10 SESSION_DIR=DIR POLICY_PATH=DIR_ON_SIM_HOST ROBOT_ARGS=trial_000.npz \
  #          GROUND_USD=default_environment.usd ISAAC_PY=python-of-the-isaac-env \
  #          sonic_official_sim_eval.sh OUT - - - GPU TABLE_GAP_CM [extra streamer args...]
  # Env: SERVER_EXTRA (e.g. "--backbone-dtype bfloat16 --noise-scale 0" for GR00T), ISAAC_GPU (default 0), SEED, MAX_ATTEMPTS
  # (stacks per trial, default 5: a trial is redone when the sim lagged), CODE_DIR, LEROBOT_DIR, HF_CACHE as below.
  : "${TRIALS:?}" "${SESSION_DIR:?}" "${POLICY_PATH:?policy dir on SIM_HOST}" "${ROBOT_ARGS:?robot session trial_000.npz}"
  : "${GROUND_USD:?local copy of Isaac Grid/default_environment.usd}" "${ISAAC_PY:?python of the Isaac Lab env}"
  OUT=$1; GPU=$5; GAP=$6; shift 6; EXTRA="$*"; SIM_HOST=${SIM_HOST:-h100}
  A=/mnt/data01/jhkim/model_weight/sonic_roundtrip_20260923; C=${CODE_DIR:-$A/code}
  LEROBOT_DIR=${LEROBOT_DIR:-/mnt/data01/jhkim/code/lerobot-g1-official-20260929-ceb40b77}
  LOG_DIR=${LOG_DIR:-./sim_eval_logs}; OUTD=$(realpath -m "$LOG_DIR/$OUT"); mkdir -p "$OUTD" "$SESSION_DIR"
  SESSION_DIR=$(realpath "$SESSION_DIR"); HERE=$(cd "$(dirname "$0")" && pwd)
  P=jihun-isaac-policy-$OUT; RPORT=$((15560 + GPU)); LPORT=${LPORT:-25560}; SOCK=$OUTD/policy.sock
  FLAGS=$(python3 -c "
import json, sys, numpy as np
a = json.loads(str(np.load(sys.argv[1], allow_pickle=True)['args']))
skip = {'policy_path', 'policy_server', 'device', 'gate_dir', 'log', 'trials', 'session_dir', 'dataset_root', 'episode',
        'replay', 'replay_horizon', 'synthetic_waist', 'encoder_model', 'observation_config', 'robot_xml', 'dex3_right_order'}
out = []
for k, v in a.items():
    if k in skip or v is None or v is False:
        continue
    out += [f'--{k.replace(\"_\", \"-\")}'] + ([] if v is True else [str(v)])
print(json.dumps(out))" "$ROBOT_ARGS") || exit 1
  echo "$FLAGS" > "$OUTD/streamer_flags.json"
  ssh -n $SIM_HOST "docker rm -f $P >/dev/null 2>&1; docker run -d --name $P --gpus device=$GPU -p 127.0.0.1:$RPORT:5560 -e HF_HUB_OFFLINE=1 \
    -e PYTHONPATH=/audit/pylib_stream -w /workspace/lerobot/examples/g1_dex3_training -v $LEROBOT_DIR:/workspace/lerobot:ro \
    -v $C:/workspace/lerobot/examples/g1_dex3_training:ro -v /mnt/data01/jhkim/model_weight/g1_dex3_20260922:/run-output:ro \
    -v ${HF_CACHE:-/mnt/data01/huggingface}:/root/.cache/huggingface -v $A:/audit:ro -v $POLICY_PATH:/policy:ro \
    --entrypoint /run-output/environment/venv/bin/python 4cbe2a3f7fc6 -u sonic_policy_server.py --policy-path /policy \
    --port 5560 $SERVER_EXTRA >/dev/null && until docker logs $P 2>&1 | grep -q 'ready on port'; do
      [ -n \"\$(docker ps -q -f name=^$P\$)\" ] || { docker logs --tail 20 $P; exit 1; }; sleep 5; done" || { echo "policy server failed" >&2; exit 1; }
  ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -L 127.0.0.1:$LPORT:127.0.0.1:$RPORT $SIM_HOST & TUNNEL=$!
  rm -f "$SOCK"; socat UNIX-LISTEN:"$SOCK",fork TCP:127.0.0.1:$LPORT & BRIDGE=$!
  trap 'kill $TUNNEL $BRIDGE 2>/dev/null; ssh -n $SIM_HOST "docker rm -f $P >/dev/null 2>&1"' EXIT
  trap 'exit 130' INT TERM
  # the server's first chunk (CUDA warm-up) takes longer than the robot sessions' 1.5 s request timeout: warm it once
  (cd "$HERE/../.." && env -u PYTHONPATH uv run python - "$LPORT" <<'PY'
import sys, time
import numpy as np
sys.path.insert(0, "examples/g1_dex3_training")
from sonic_policy_streamer import RemotePolicy
p = RemotePolicy(f"tcp://127.0.0.1:{sys.argv[1]}", 300.0)
img = {k: np.zeros((480, 640, 3), np.uint8) for k in p.image_keys}  # camera size: warms the server's resize path too
for i in range(3):
    t = time.time()
    p.chunk([np.zeros(28, np.float32)] * p.n_obs, [img] * p.n_obs, "warm-up")
    print(f"[policy] warm-up chunk {i}: {time.time() - t:.2f} s", flush=True)
PY
  ) || { echo "policy server warm-up failed" >&2; exit 1; }
  done_n() { [ -f "$SESSION_DIR/results.jsonl" ] && wc -l < "$SESSION_DIR/results.jsonl" || echo 0; }
  for i in $(seq 0 $((TRIALS - 1))); do
    for attempt in $(seq "${MAX_ATTEMPTS:-5}"); do
      [ "$(done_n)" -gt "$i" ] && break
      G=$OUTD/gate_${i}_$attempt; mkdir -p "$G" "$OUTD/sim"
      unshare -rn bash -c '
        SOCK=$0 HERE=$1 N=$2 SESS=$3 GAP=$4 OUTD=$5 G=$6 TAG=$7 EXTRA=$8
        mapfile -t FL < <(python3 -c "import json, sys; print(*json.load(open(sys.argv[1])), sep=chr(10))" "$OUTD/streamer_flags.json")
        ip link set lo up
        socat TCP-LISTEN:5560,bind=127.0.0.1,fork,reuseaddr UNIX-CONNECT:"$SOCK" & BR=$!
        cd "$HERE" && SIM=isaac TRIALS=$N SESSION_DIR="$SESS" SEED=${SEED:-0} TABLE_GAP_CM=$GAP CUDA_VISIBLE_DEVICES=${ISAAC_GPU:-0} \
          timeout -k 30 400 env -u PYTHONPATH "$ISAAC_PY" -u sonic_official_sim_host.py "$OUTD/sim" "$G" > "$OUTD/host_$TAG.log" 2>&1 & HOST=$!
        # the streamer grabs a camera frame at start (10 s timeout): wait until Isaac is up and the deploy is ready.
        # 127.0.0.1, not localhost: resolving localhost in the user namespace costs ~2 s per connection (warm-up timeout)
        # --policy-timeout-s 5 (robot: 1.5): the first chunk of a fresh streamer through the namespace bridge took 1.5-2.1 s
        # (later ones 0.15-0.4 s); the per-trial chunk latency is logged
        until [ -f "$G/deploy_ready" ] || ! kill -0 $HOST 2>/dev/null; do sleep 2; done
        cd "$HERE/../.." && timeout -k 10 380 env -u PYTHONPATH uv run python "$HERE/sonic_policy_streamer.py" \
          --policy-server tcp://127.0.0.1:5560 --gate-dir "$G" --trials $N --session-dir "$SESS" "${FL[@]}" \
          --dex3-right-order swap --policy-timeout-s 5 $EXTRA > "$OUTD/streamer_$TAG.log" 2>&1 & STR=$!
        wait -n $HOST $STR; sleep 20; kill $STR 2>/dev/null; wait $HOST; kill $BR 2>/dev/null' \
        "$SOCK" "$HERE" $((i + 1)) "$SESSION_DIR" "$GAP" "$OUTD" "$G" "${i}_$attempt" "$EXTRA"
      grep -h "chunk inference s" "$OUTD/streamer_${i}_$attempt.log" | tail -1 | sed "s/^/[trial $((i + 1)) attempt $attempt] /"
    done
    [ "$(done_n)" -gt "$i" ] || { echo "trial $((i + 1)) not saved after ${MAX_ATTEMPTS:-5} stacks" >&2; exit 1; }
  done
  echo "[$(TZ=Asia/Seoul date +%H:%M) KST] $OUT: $(python3 -c "
import json, sys; r = [json.loads(l) for l in open(sys.argv[1])]
print(f\"{sum(x['verdict'] == 'success' for x in r)}/{len(r)} success\")" "$SESSION_DIR/results.jsonl")"
  exit 0
fi
LOG_DIR=${LOG_DIR:-./sim_eval_logs}; mkdir -p "$LOG_DIR"
SIM_HOST=${SIM_HOST:-h100}
OUT=$1; RUN=$2; DS=$3; EP=$4; GPU=$5; GAP=$6; shift 6; EXTRA="$*"
A=/mnt/data01/jhkim/model_weight/sonic_roundtrip_20260923; C=${CODE_DIR:-$A/code}; J28=${JOINT28_DIR:-/run-output/datasets/joint28}; G=/home/kube/sonic_vla_integration/20260906; R=/run-output/runs; D=/run-output/datasets
LEROBOT_DIR=${LEROBOT_DIR:-/mnt/data01/jhkim/code/lerobot-g1-official-20260929-ceb40b77}
BACKEND=${BACKEND:-sonic}; REPLAY=${REPLAY:-0}; SONIC=${SONIC_DIR:-$D/$DS}
case $BACKEND in
  sonic) HOST_PY=sonic_official_sim_host.py; HOST_MNT=""; HOST_ARGS=""; STREAMER_FLAGS="--dex3-right-order swap" ;;
  decoupled) HOST_PY=decoupled_wbc_sim_host.py; STREAMER_FLAGS="--backend decoupled --dex3-right-order dataset"
    HOST_MNT="-v $A/upstream_b042411fae/decoupled_wbc:/upstream/decoupled_wbc:ro -v $A/pylib_ort310:/pylib_ort:ro -e PYTHONPATH=/pylib_ort:/upstream" ;;
  *) echo "BACKEND must be sonic or decoupled" >&2; exit 2 ;;
esac
if [ "$REPLAY" = 1 ]; then SRC="--replay"; else SRC="--policy-server tcp://localhost:5560"; fi
H="jihun-sonic-simhost-$OUT"; S="jihun-sonic-streamer-$OUT"
ssh_() { local rc; while :; do ssh -n -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 $SIM_HOST "$@"; rc=$?; [ $rc -eq 255 ] || return $rc; ${VPN_ON:-true} >/dev/null 2>&1 </dev/null; sleep 20; done; }
ssh_ "docker rm -f $H $S P-$OUT >/dev/null 2>&1; docker run --rm -v $A:/a --entrypoint bash 4cbe2a3f7fc6 -c 'rm -rf /a/$OUT && mkdir -p /a/$OUT/sim /a/$OUT/gate'
M=$G/models/sonic/sonic_v1_1; P=\$(readlink -f \$(ls -d /mnt/data01/huggingface/hub/models--nvidia--GEAR-SONIC/snapshots/*/planner_sonic.onnx)); W=/workspace/GR00T-WholeBodyControl/gear_sonic_deploy; SD=$A/$OUT
docker run -d --name $H --gpus device=$GPU -e NVIDIA_DRIVER_CAPABILITIES=all -e MUJOCO_GL=egl -e TABLE_GAP_CM=$GAP -w \$W -v \$M/model_encoder.onnx:\$W/policy/sonic_v1_1/model_encoder.onnx:ro -v \$M/model_decoder.onnx:\$W/policy/sonic_v1_1/model_decoder.onnx:ro -v \$P:\$W/planner/target_vel/V2/planner_sonic.onnx:ro -v $C:/code:ro -v \$SD/sim:/out -v \$SD/gate:/gate $HOST_MNT --entrypoint /opt/sonic-sim/bin/python jihun/sonic-vla-sim:startupfix-20260907 -u /code/$HOST_PY /out /gate $HOST_ARGS >/dev/null &&
until [ -f \$SD/gate/deploy_ready ] || [ -z \"\$(docker ps -q -f name=$H)\" ]; do sleep 5; done &&
{ [ $REPLAY = 1 ] || { docker run -d --name P-$OUT --network container:$H --gpus device=$GPU -e PYTHONPATH=/audit/pylib_stream -e HF_HUB_OFFLINE=1 -w /workspace/lerobot/examples/g1_dex3_training -v $LEROBOT_DIR:/workspace/lerobot:ro -v $C:/workspace/lerobot/examples/g1_dex3_training:ro -v /mnt/data01/jhkim/model_weight/g1_dex3_20260922:/run-output:ro -v ${HF_CACHE:-/mnt/data01/huggingface}:/root/.cache/huggingface -v $A:/audit:ro --entrypoint /run-output/environment/venv/bin/python 4cbe2a3f7fc6 -u sonic_policy_server.py --policy-path $R/$RUN/checkpoints/last/${POLICY_SUBDIR:-pretrained_model} $SERVER_EXTRA >/dev/null &&
until docker logs P-$OUT 2>&1 | grep -q 'ready on port'; do [ -n \"\$(docker ps -q -f name=^P-$OUT\$)\" ] || { echo \"policy server P-$OUT exited:\"; docker logs --tail 20 P-$OUT 2>&1; exit 1; }; sleep 5; done; }; } &&
docker run -d --name $S --network container:$H -e PYTHONPATH=/audit/pylib_stream -e HF_HUB_OFFLINE=1 -w /workspace/lerobot -v $LEROBOT_DIR:/workspace/lerobot:ro -v $C:/workspace/lerobot/examples/g1_dex3_training:ro -v /mnt/data01/jhkim/model_weight/g1_dex3_20260922:/run-output:ro -v /mnt/data01/jhkim/datasets/unitreerobotics:/source-datasets:ro -v ${HF_CACHE:-/mnt/data01/huggingface}:/root/.cache/huggingface -v $A:/audit -v \$SD/gate:/gate -v \$M:/sonic-model:ro --entrypoint /run-output/environment/venv/bin/python 4cbe2a3f7fc6 -u examples/g1_dex3_training/sonic_policy_streamer.py $SRC --dataset-root $D/$DS --episode $EP --state-source robot --max-chunk-age-s 2.0 --gate-dir /gate --log /audit/$OUT/streamer.npz $STREAMER_FLAGS $EXTRA >/dev/null" && echo "[$(TZ=Asia/Seoul date +%H:%M) KST] $OUT launched" || { echo "launch of $OUT failed (policy server or sim host exited)" >&2; ssh_ "docker logs P-$OUT 2>&1" > "$LOG_DIR/$OUT.server.log" 2>&1; ssh_ "docker rm -f $H $S P-$OUT >/dev/null 2>&1 || true"; exit 1; }
while :; do r=$(ssh -n -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 $SIM_HOST "docker ps -q -f name=$H" 2>/dev/null) || { ${VPN_ON:-true} >/dev/null 2>&1 </dev/null; sleep 20; continue; }; [ -z "$r" ] && break; sleep 30; done
ssh -n -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 $SIM_HOST "docker logs $S 2>&1" > "$LOG_DIR/$OUT.streamer.log" 2>&1
ssh -n -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 $SIM_HOST "docker logs $H 2>&1" > "$LOG_DIR/$OUT.host.log" 2>&1
ssh_ "docker logs $S 2>&1 | grep -E 'chunk inference|no valid chunk|rejected|Error|old: ending' | tail -3; docker rm -f $H $S P-$OUT >/dev/null 2>&1 || true"
ssh_ "docker run --rm -e PYTHONPATH=/audit/pylib -e MUJOCO_GL=disable -v $A:/audit -v $C:/code:ro -v /mnt/data01/jhkim/model_weight/g1_dex3_20260922:/run-output:ro -v $G/source/gear_sonic:/gear_sonic:ro -v $G/source/gear_sonic_deploy:/gear_sonic_deploy:ro -v $G/models/sonic/sonic_v1_1:/sonic-model:ro --entrypoint sh 4cbe2a3f7fc6 -c '/run-output/environment/venv/bin/python /code/sonic_stream_eval.py /audit/$OUT $J28 $SONIC $EP > /audit/$OUT/stream_eval.json 2>/dev/null'; grep -E 'frames_scored|\"p50\"|\"p95\"|tilt_max|contacts_min' $A/$OUT/stream_eval.json | head -8" && echo "[$(TZ=Asia/Seoul date +%H:%M) KST] $OUT scored"
