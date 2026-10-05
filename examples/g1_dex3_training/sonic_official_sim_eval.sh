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
# Env: SIM_HOST (ssh alias of the GPU host; default h100), LOG_DIR (local streamer logs, default ./sim_eval_logs), CODE_DIR (H100 script copy, default $A/code), JOINT28_DIR (joint28 dataset for scoring; HE: /run-output/humanoid_everyday_g1_20260923/datasets/joint28), SERVER_EXTRA (extra policy server args, e.g. --noise-seed 0), VPN_ON (command to re-establish the VPN, optional), BACKEND (sonic | decoupled; default sonic), HOST_ARGS (extra host args, decoupled only, e.g. --waist-location lower_and_upper_body), REPLAY (1: stream the episode's recorded actions; no policy server, RUN_NAME is ignored), SONIC_DIR (stored-token dataset for scoring; default the episode dataset), LEROBOT_DIR (lerobot code mounted at /workspace/lerobot for the policy server and streamer; default the official-recipe copy lerobot-g1-official-20260929-ceb40b77).
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
{ [ $REPLAY = 1 ] || { docker run -d --name P-$OUT --network container:$H --gpus device=$GPU -e PYTHONPATH=/audit/pylib_stream -e HF_HUB_OFFLINE=1 -w /workspace/lerobot/examples/g1_dex3_training -v $LEROBOT_DIR:/workspace/lerobot:ro -v $C:/workspace/lerobot/examples/g1_dex3_training:ro -v /mnt/data01/jhkim/model_weight/g1_dex3_20260922:/run-output:ro -v /mnt/data01/huggingface:/root/.cache/huggingface -v $A:/audit:ro --entrypoint /run-output/environment/venv/bin/python 4cbe2a3f7fc6 -u sonic_policy_server.py --policy-path $R/$RUN/checkpoints/last/pretrained_model $SERVER_EXTRA >/dev/null &&
until docker logs P-$OUT 2>&1 | grep -q 'ready on port'; do [ -n \"\$(docker ps -q -f name=^P-$OUT\$)\" ] || { echo \"policy server P-$OUT exited:\"; docker logs --tail 20 P-$OUT 2>&1; exit 1; }; sleep 5; done; }; } &&
docker run -d --name $S --network container:$H -e PYTHONPATH=/audit/pylib_stream -e HF_HUB_OFFLINE=1 -w /workspace/lerobot -v $LEROBOT_DIR:/workspace/lerobot:ro -v $C:/workspace/lerobot/examples/g1_dex3_training:ro -v /mnt/data01/jhkim/model_weight/g1_dex3_20260922:/run-output:ro -v /mnt/data01/jhkim/datasets/unitreerobotics:/source-datasets:ro -v /mnt/data01/huggingface:/root/.cache/huggingface -v $A:/audit -v \$SD/gate:/gate -v \$M:/sonic-model:ro --entrypoint /run-output/environment/venv/bin/python 4cbe2a3f7fc6 -u examples/g1_dex3_training/sonic_policy_streamer.py $SRC --dataset-root $D/$DS --episode $EP --state-source robot --max-chunk-age-s 2.0 --gate-dir /gate --log /audit/$OUT/streamer.npz $STREAMER_FLAGS $EXTRA >/dev/null" && echo "[$(TZ=Asia/Seoul date +%H:%M) KST] $OUT launched" || { echo "launch of $OUT failed (policy server or sim host exited)" >&2; ssh_ "docker logs P-$OUT 2>&1" > "$LOG_DIR/$OUT.server.log" 2>&1; ssh_ "docker rm -f $H $S P-$OUT >/dev/null 2>&1 || true"; exit 1; }
while :; do r=$(ssh -n -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 $SIM_HOST "docker ps -q -f name=$H" 2>/dev/null) || { ${VPN_ON:-true} >/dev/null 2>&1 </dev/null; sleep 20; continue; }; [ -z "$r" ] && break; sleep 30; done
ssh -n -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 $SIM_HOST "docker logs $S 2>&1" > "$LOG_DIR/$OUT.streamer.log" 2>&1
ssh -n -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 $SIM_HOST "docker logs $H 2>&1" > "$LOG_DIR/$OUT.host.log" 2>&1
ssh_ "docker logs $S 2>&1 | grep -E 'chunk inference|no valid chunk|rejected|Error|old: ending' | tail -3; docker rm -f $H $S P-$OUT >/dev/null 2>&1 || true"
ssh_ "docker run --rm -e PYTHONPATH=/audit/pylib -e MUJOCO_GL=disable -v $A:/audit -v $C:/code:ro -v /mnt/data01/jhkim/model_weight/g1_dex3_20260922:/run-output:ro -v $G/source/gear_sonic:/gear_sonic:ro -v $G/source/gear_sonic_deploy:/gear_sonic_deploy:ro -v $G/models/sonic/sonic_v1_1:/sonic-model:ro --entrypoint sh 4cbe2a3f7fc6 -c '/run-output/environment/venv/bin/python /code/sonic_stream_eval.py /audit/$OUT $J28 $SONIC $EP > /audit/$OUT/stream_eval.json 2>/dev/null'; grep -E 'frames_scored|\"p50\"|\"p95\"|tilt_max|contacts_min' $A/$OUT/stream_eval.json | head -8" && echo "[$(TZ=Asia/Seoul date +%H:%M) KST] $OUT scored"
