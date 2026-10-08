#!/bin/bash
# Real-robot run of the HE GR00T (official recipe) with its reference setting: noise scale 0 and a 0.3 s chunk blend
# (zero-noise flow sampling: err 0.149 -> 0.131, seam 0.140 -> 0.091; 2026-10-01, see
# docs/research/2026-09-30-official-retraining-handover.md). RTC stays off (GR00T RTC drifts, err ~0.50).
# Run from the workstation, each step in its own terminal, in this order; the PC2 steps go over `ssh -t pc2_222`.
#
#   g1_groot_real_run.sh server                  # workstation GPU 1, port 5560, waits for "ready on port"
#   g1_groot_real_run.sh deploy                  # PC2: NVIDIA SONIC deploy; wait for "Init Done"
#   g1_groot_real_run.sh camera                  # PC2: D435i head camera, 640x480, stream "egocentric"
#   g1_groot_real_run.sh stream RUN [TASK] [extra streamer args...]
#                                                # PC2: streamer; Enter at "Init Done", Enter once in position
#   g1_groot_real_run.sh stop                    # stop the workstation server
#
# e.g. g1_groot_real_run.sh stream run20_he_groot_official_laptop_t0 "close a laptop g1"
# Env: MODEL (checkpoint dir), GPU, PORT, NOISE_SCALE (default 0; empty = unscaled sampling), BLEND_S (streamer
#      --chunk-blend-s, default 0.3; 0 = runs 20-21), CAM_FPS (default 30),
#      POLICY_FPS (50 for the G1 WBT models; needs the PC2 streamer copy with --policy-fps),
#      POLICY_SUBDIR (default pretrained_model), BACKBONE_DTYPE (default bfloat16, GR00T only; empty for others).
set -e
MODEL=${MODEL:-/mnt/data/jihun/g1_models/he_groot_sonic78sonicstate_ho5_official_full}
# Robot 2026-10-08 (docs/research/2026-09-29-issue-groot-jerky-predictions.md): noise 0 + blend 0.3 s removed the chunk
# seams (runs 24-27, jerk p95 214-276 vs 315-514 without blend); blend 0.5 s similar (run 28); noise 0.5 was jerky inside
# the chunks (run 29, jerk p95 669), so the default is back to 0. At noise 0 the arm can stall after ~15 s; if that
# becomes the main problem, raise the noise only while the arm is stalled (impl log 10, option 4; not implemented).
GPU=${GPU:-1}; PORT=${PORT:-5560}; NOISE_SCALE=${NOISE_SCALE-0}; BLEND_S=${BLEND_S-0.3}
POLICY_SUBDIR=${POLICY_SUBDIR:-pretrained_model}  # pretrained_model_ema for Pi0.5
BACKBONE_DTYPE=${BACKBONE_DTYPE-bfloat16}  # GR00T only; set empty (BACKBONE_DTYPE=) for other policies
CAM_FPS=${CAM_FPS:-30}  # camera publish rate; capture is 30 Hz, so 30 is the max (server default 10)
WS=192.168.0.62  # workstation address seen from PC2
UNIT=groot-server-$PORT
cd "$(dirname "$0")"

case $1 in
server)
  LOG=$MODEL/server_gpu${GPU}_${PORT}_t${NOISE_SCALE:-1}.log
  # memory-capped user service: loading the fp32 checkpoint on the CPU first can trigger systemd-oomd otherwise
  systemd-run --user --unit=$UNIT -p MemoryMax=32G -p MemorySwapMax=0 -p WorkingDirectory="$PWD" \
    -E PATH="$PATH" -E HOME="$HOME" -E CUDA_VISIBLE_DEVICES=$GPU -E HF_HUB_OFFLINE=1 \
    bash -c "exec uv run --project ../.. python -u sonic_policy_server.py --policy-path $MODEL/$POLICY_SUBDIR \
      --port $PORT ${BACKBONE_DTYPE:+--backbone-dtype $BACKBONE_DTYPE} ${NOISE_SCALE:+--noise-scale $NOISE_SCALE} > $LOG 2>&1"
  until grep -q "ready on port" "$LOG" 2>/dev/null; do
    systemctl --user is-active -q $UNIT || { tail -20 "$LOG"; exit 1; }
    sleep 5
  done
  grep "ready on port" "$LOG" ;;
deploy)
  # A conda env's DDS library clashes with the SDK's ("corrupted size vs. prev_size" at "Creating G1Deploy object",
  # 2026-10-06): leave conda and preload the SDK's own libddsc/libddscxx (what worked for run22).
  ssh -t pc2_222 'cd ~/GR00T-WholeBodyControl/gear_sonic_deploy && { conda deactivate 2>/dev/null || true; } &&
    source scripts/setup_env.sh && L=$(pwd)/thirdparty/unitree_sdk2/thirdparty/lib/aarch64 &&
    export LD_PRELOAD="$L/libddsc.so:$L/libddscxx.so" &&
    bash deploy.sh --cp policy/sonic_v1_1/model --obs-config policy/sonic_v1_1/observation_config.yaml \
      --input-type zmq_manager --zmq-host localhost real' ;;
camera)
  ssh -t pc2_222 "cd ~/g1_sonic_eval && .venv/bin/python code/g1_head_camera_server.py \
    --device /dev/video4 --name egocentric --width 640 --height 480 --publish-fps $CAM_FPS" ;;
stream)
  RUN=${2:?run name, e.g. run20_he_groot_official_laptop_t0}; TASK=${3:-close a laptop g1}; shift $(($# < 3 ? $# : 3))
  ssh -t pc2_222 "cd ~/g1_sonic_eval/code && ../.venv/bin/python -u sonic_policy_streamer.py \
    --policy-server tcp://$WS:$PORT --images zmq --camera-host localhost --task '$TASK' \
    --start planner --end planner --duration-s 30 --replan-s 0.4 --max-token-step 0.1 ${BLEND_S:+--chunk-blend-s $BLEND_S} \
    ${POLICY_FPS:+--policy-fps $POLICY_FPS} --log ../runs/$RUN.npz $* 2>&1 | tee ../runs/$RUN.log" ;;
stop)
  # a stopped transient unit stays "failed" (non-zero exit) and blocks the next start under the same name
  systemctl --user stop $UNIT; systemctl --user reset-failed $UNIT 2>/dev/null || true ;;
*)
  sed -n 2,15p "$0"; exit 1 ;;
esac
