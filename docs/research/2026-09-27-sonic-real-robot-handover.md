# G1 Dex3 + SONIC: handover for the Ubuntu workstation (2026-09-27)

Work moves from a macOS laptop to an Ubuntu workstation. This page says what exists, where it lives, how to set up
the workstation, how to run the first real-robot evaluation, and what is still open. The detailed evidence and
every decision are in [`2026-09-23-sonic-roundtrip-audit.md`](./2026-09-23-sonic-roundtrip-audit.md) (the
knowledge note); read it next.

This repository is public: addresses, user names and VPN credentials are deliberately left out. Get them from the
team (or the previous workstation's `~/.ssh/config` and VPN config).

## Status in one paragraph

Unitree G1 Dex3 datasets are converted to SONIC v1.1 tokens (78D: 64 token + 14 Dex3 hands) with the converter's
speed limits off; SONIC reproduces the hand path within the 5 cm gate (Unitree p95 3.07 cm; Humanoid Everyday
accepted at ~5 cm). Seven policies were trained on both the original and the relabelled state (arm state replaced
by the pose SONIC reaches), 5% held out. In NVIDIA's deploy in MuJoCo, with an 80 cm table and a table-safe startup,
**GR00T** tracks best (palm error 6.1 / 12.7 cm on two held-out episodes); Diffusion and FastWAM lose balance.
The streamer is split for the real robot: a robot-side process (no GPU) keeps all real-time work and asks a
policy server on H100 for chunks over the VPN. Everything is merged (PRs #2–#6). **Next: the first real-robot run.**

## Where things are

Repository (`examples/g1_dex3_training/`):

| File                                                                                              | What it does                                                                                   |
| ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `sonic_policy_streamer.py`                                                                        | Robot side: 50 Hz tokens to NVIDIA's deploy, table startup/shutdown, planner hand-back, gates. |
| `sonic_policy_server.py`                                                                          | H100 side: serves policy chunks (ZMQ REQ/REP, msgpack, JPEG frames).                           |
| `sonic_table_startup.py`                                                                          | Plans and checks the table-safe arm path (planner stance ↔ fixed initial pose).               |
| `sonic_official_sim_host.py`, `sonic_official_sim_eval.sh`                                        | NVIDIA deploy + MuJoCo (optional table) and the end-to-end sim evaluation driver.              |
| `sonic_stream_eval.py`, `sonic_stream_phases.py`                                                  | Score a sim run: palm/wrist error, balance; per-phase table clearance and balance.             |
| `sonic_token_stream.py`, `sonic_targets.py`, `prepare_sonic_dataset.py`, `sonic_state_relabel.py` | Token resampling, converter, relabelling.                                                      |

H100 host (`/run-output` = `/mnt/data01/jhkim/model_weight/g1_dex3_20260922`):

- Datasets: `/run-output/datasets/{joint28,sonic78_nolimit,sonic78_nolimit_sonicstate}` and the Humanoid Everyday
  equivalents under `/run-output/humanoid_everyday_g1_20260923/datasets/` (HE trains on `sonic78_nolimit_g2`).
- Models (final `pretrained_model` only, not resumable): `/run-output/runs/{act,diffusion,groot,vla_jepa,pi05,molmoact2,fastwam}_sonic78{nolimit,sonicstate}_ho5_full`,
  same under `humanoid_everyday_g1_20260923/runs/`. Real-robot candidate: `groot_sonic78nolimit_ho5_full`;
  backups MolmoAct2 / Pi0.5 `sonic78sonicstate`.
- Audit / sim outputs: `/mnt/data01/jhkim/model_weight/sonic_roundtrip_20260923/` (kept on purpose). Its `code/`
  folder is the script copy that containers mount as `/code` or as the examples directory; keep it in sync with
  this repository (copy through a container, never write to the host directly).
- Table startup file used in the final runs (1 rad/s in, 2 rad/s out):
  `/mnt/data01/jhkim/model_weight/sonic_roundtrip_20260923/table_startup/r20/startup.npz`.
- Images: lerobot `4cbe2a3f7fc6` (Python `/run-output/environment/venv/bin/python`; pyzmq/msgpack in
  `/audit/pylib_stream`, MuJoCo in `/audit/pylib`), NVIDIA deploy + sim `jihun/sonic-vla-sim:startupfix-20260907`.
- SONIC v1.1 models and gear_sonic source: `/home/kube/sonic_vla_integration/20260906/{models/sonic/sonic_v1_1,source}`.

## Workstation setup (Ubuntu)

1. Repository: `git clone` this fork, then `uv sync --locked --extra test --extra dev` (see `AGENTS.md`).
2. VPN: `sudo apt install openfortivpn`, a config file with the gateway, user and `trusted-cert` (from the team),
   `sudo openfortivpn -c <config>`. The tunnel often stays "up" while its routes vanish; when `ssh h100` times out,
   restart it (`sudo pkill openfortivpn`, then connect again).
3. SSH alias `h100` in `~/.ssh/config`. Add `ServerAliveInterval 15` / `ServerAliveCountMax 4`: without them a
   VPN drop leaves SSH sessions hanging forever.
4. H100 rules: Docker containers only; install nothing on the host and write to host paths only through container
   mounts. GPU 0 runs a Qwen vLLM container (`jihun-lerobot-qwen36-gpu0`) that may be stopped when needed and must be
   restarted afterwards. Another working session may be using GPUs 2/3/7; check `nvidia-smi` first.

## First real-robot run

Procedure (decided 2026-09-27): start SONIC on the robot about 30 cm from the table; move the robot into position
in planner mode; press Enter to run the episode. The robot then switches to POSE mode, runs the 8.2 s table-safe
startup, the policy, the 5.6 s path back to the arms-down stance, and returns to planner mode, leaving the deploy
running (stop it with NVIDIA's own procedure).

1. **H100: policy server** (container, a free GPU, port published):

   ```bash
   A=/mnt/data01/jhkim/model_weight/sonic_roundtrip_20260923
   docker run -d --name sonic-policy-server --gpus device=<N> -p 5560:5560 \
     -e PYTHONPATH=/audit/pylib_stream -e HF_HUB_OFFLINE=1 -w /workspace/lerobot/examples/g1_dex3_training \
     -v /mnt/data01/jhkim/code/lerobot-g1-dex3-20260922:/workspace/lerobot:ro \
     -v $A/code:/workspace/lerobot/examples/g1_dex3_training:ro \
     -v /mnt/data01/jhkim/model_weight/g1_dex3_20260922:/run-output:ro \
     -v /mnt/data01/huggingface:/root/.cache/huggingface -v $A:/audit:ro \
     --entrypoint /run-output/environment/venv/bin/python 4cbe2a3f7fc6 -u sonic_policy_server.py \
     --policy-path /run-output/runs/groot_sonic78nolimit_ho5_full/checkpoints/last/pretrained_model
   docker logs -f sonic-policy-server   # wait for "ready on port 5560" (GR00T loads in ~15 s, MolmoAct2 ~2 min)
   ```

   Not yet verified: that port 5560 on H100 is reachable from a VPN client (SSH is). Test from the robot PC first
   (e.g. `nc -vz <H100> 5560`).

2. **Robot PC** (reaches H100 only through the VPN):
   - This repository plus `pyzmq`, `msgpack`, `numpy`, `opencv-python-headless` (the streamer imports lerobot
     utilities; torch is not needed).
   - Copy the table startup file from H100 (path above).
   - Head camera: `python -m lerobot.robots.unitree_g1.run_g1_server --camera ...` (LeRobot ImageServer, port 5555).
     A side-by-side stereo frame is split into `cam_left_high` / `cam_right_high` and resized to 640×480; two
     cameras named like those keys are used as is. Check the stereo layout of your head camera before the first run.
   - NVIDIA `g1_deploy_onnx_ref` per NVIDIA's real-robot instructions, with
     `--input-type zmq_manager --output-type zmq --zmq-host localhost` (as in sim).

3. **Robot PC: streamer**

   ```bash
   python examples/g1_dex3_training/sonic_policy_streamer.py \
     --policy-server tcp://<H100>:5560 --images zmq --camera-host localhost \
     --task "<instruction, as in the dataset>" --duration-s 30 \
     --startup-tokens startup.npz --log run.npz
   ```

   - It prints "policy ready: X s per chunk" (the real VPN round trip; 0.22 s on H100 locally). If X is near 1 s,
     raise `--policy-timeout-s` (default 1.5).
   - Gates: Enter after the deploy prints "Init Done" (starts planner mode); Enter once the robot is in position.
   - Keep `--dex3-right-order dataset` (the default): the real Dex3 right hand follows the dataset order. `swap` is
     only for NVIDIA's MuJoCo bridge.
   - Safety behaviour: no new chunk for `--max-chunk-age-s` (default 1.0 s; the sim runs used 2.0 s) or robot state
     older than 0.2 s ends the episode and runs the shutdown path. Killing the streamer mid-episode skips the
     shutdown path; use NVIDIA's stop / e-stop then.

## Re-running the sim checks from the workstation

```bash
cd examples/g1_dex3_training
# copy this directory's scripts to /audit/code on H100 first (through a container)
LOG_DIR=./sim_eval_logs ./sonic_official_sim_eval.sh MYRUN groot_sonic78nolimit_ho5_full sonic78_nolimit 6 2 5 \
  --startup-tokens /audit/table_startup/r20/startup.npz
ssh h100 docker run --rm -v /mnt/data01/jhkim/model_weight/sonic_roundtrip_20260923:/audit \
  -v /mnt/data01/jhkim/model_weight/g1_dex3_20260922:/run-output:ro \
  --entrypoint /run-output/environment/venv/bin/python 4cbe2a3f7fc6 /audit/code/sonic_stream_phases.py /audit/MYRUN
```

Expect for GR00T, episode 6, table 5 cm: no table contact during startup and shutdown, a brief brush during the
last 2 s hand-back (accepted), palm ~3.5 / 12–14 cm, all feet down. A run takes ~6 min (the TensorRT engine is
rebuilt at every container start).

## Open items

1. Real VPN round trip and H100 port reachability (step 1 above).
2. Live head camera: stereo layout, colour, and exposure versus the training images.
3. Per-task initial poses (the fixed pose cost the policies a few cm in sim); a table with an apron is untested.
4. VLA-JEPA checkpoints fail to load (`embed_tokens` missing); the Humanoid Everyday models were not ranked.
5. With the table 5 cm away, the final hand-back into NVIDIA's planner stance may brush the table (planner-stance
   wrists sit ~2 cm under the edge). Accepted; at ≥ 8 cm there is room.

## Pitfalls met (so you don't repeat them)

- Long-running training containers can lose the GPU ("Failed to initialize NVML") and silently train on CPU:
  `docker restart` them.
- In the official sim, start the streamer / policy server only after the deploy prints "Init Done": loading a large
  VLA during the TensorRT build killed the sim host.
- GR00T and MolmoAct2 are slow on the first call of every new thread; the streamer and server use one long-lived
  inference thread warmed up before the robot moves.
- The sim table appears only once the robot stands in planner mode (the elastic-band hang and reset drop the hands
  onto a table that is already there).
- Build every dataset's quantile statistics with `augment_joint_quantiles.py`; aggregated per-episode quantiles
  were off by up to 0.7. That script is not in this repository yet (it was uncommitted on the laptop); a copy is in
  `/mnt/data01/jhkim/model_weight/sonic_roundtrip_20260923/code/` on H100.
- Stopping control at the end of an episode made the robot collapse in sim; always hand back to planner mode.
