# G1 Dex3 → SONIC v1.1 conversion: round-trip audit, decisions, and how to reproduce

Status as of 2026-09-26. Question: does converting joint-action datasets (28D: arms 14 + Dex3 hands 14) into
SONIC tokens (78D: token 64 + hands 14) keep the hand where the original action put it? Gate: p95 palm
position error vs. the original command > 5 cm in the pelvis frame ⇒ unusable for VLA fine-tuning.
Evidence is simulation-only (MuJoCo); no real-robot claim.

## Decisions (user-confirmed)

| Topic                       | Decision                                                                                                                                                                                                                                                                                                                                                   |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Converter speed limits      | **Off.** `prepare_sonic_dataset.py --arm-speed-limit none --hand-speed-limit none`. The old 1 rad/s arm / 2 rad/s hand slew limit caused most of the error.                                                                                                                                                                                                |
| Episode start in evaluation | 1 s linear token blend from `LATENT_INITIAL_MOTION_TOKEN` to the episode's first token, 1 s hold.                                                                                                                                                                                                                                                          |
| Token rate                  | Train on the 30 Hz dataset. At deploy, linearly interpolate the policy's 30 Hz tokens to 50 Hz (resample the predicted action chunk; interpolation needs the next token). Re-encoding at 50 Hz gave no gain (≤0.1 cm).                                                                                                                                     |
| POSE handoff (deploy)       | **Gradual (2026-09-25).** Send the planner's current `token_state` first, then blend to `LATENT_INITIAL_MOTION_TOKEN` over 2 s (`--handoff-blend-s 2`, `HANDOFF_BLEND_S=2`). An instant switch unloaded a foot in 5/5 runs (arm speed up to 30.7 rad/s); 2 s kept all 8 contacts in 3/3 (1.2 rad/s); 4 s swayed in 1/3.                                    |
| Dex3 right-hand order       | **Dataset order (thumb, index, middle) on the real robot.** NVIDIA's MuJoCo bridge maps index↔middle slots, so sim runs use `--dex3-right-order swap`.                                                                                                                                                                                                    |
| Policy state input          | **Relabelled state** (`sonic78_nolimit_sonicstate`): `observation.state` arms replaced by the arm pose SONIC reaches. Best closed loop and open loop in the held-out A/B below.                                                                                                                                                                            |
| Dataset quantile stats      | **Exact (2026-09-25).** `joint28` / `sonic78` / `sonic78_nolimit` shipped approximate global q01/q99 (state off by up to 0.55 rad, action tokens up to 0.375; mean/std were exact). Run `augment_joint_quantiles.py --root <dataset>` after building any dataset. Only quantile-normalized policies (MolmoAct2, Pi0.5) were affected; they were retrained. |
| Table startup (tabletop)    | **Skip the ready pose at the table (2026-09-26).** Planner stance (arms down) → table-safe arm path → fixed initial pose → policy; see "Tabletop startup" below. In progress.                                                                                                                                                                              |
| Humanoid Everyday dataset   | **Accepted as-is (2026-09-24)** despite p95 5.21 cm, 0.2 cm over the 5 cm gate: 20% of sampled episodes exceed 5 cm on their own p95 (worst 8.6 cm); no episodes are excluded. The miss comes from SONIC tracking fast manipulation, not from source glitches.                                                                                             |

Datasets built with limits off (existing `sonic78` datasets are unchanged):
`/run-output/datasets/sonic78_nolimit` (3152 ep, 2,587,515 frames) and
`/run-output/humanoid_everyday_g1_20260923/datasets/sonic78_nolimit` (4064 ep, 1,779,287 frames).
`/run-output` = `/mnt/data01/jhkim/model_weight/g1_dex3_20260922` on the H100 host.

## Results

Offline harness (validated against the official deploy to within 0.2–0.8 cm), stored tokens, same episode picks
(4 random per task + most clipped + most speed-limited; seed 0):

| Dataset                                      | Palm p50 / p95 / max (cm) | Episodes > 5 cm | Lag  | Verdict                         |
| -------------------------------------------- | ------------------------- | --------------- | ---- | ------------------------------- |
| Unitree `sonic78` (limited, tokens held)     | 1.45 / 3.40 / 17.6        | 14%             | 2 fr | pass                            |
| **Unitree `sonic78_nolimit` (interpolated)** | 1.42 / **3.07** / 9.3     | 7%              | 1 fr | **pass**                        |
| Humanoid Everyday `sonic78` (limited)        | 2.07 / 8.82 / 68.4        | 77%             | 3 fr | fail                            |
| **Humanoid Everyday `sonic78_nolimit`**      | 1.92 / **5.21** / 53.8    | 20%             | 1 fr | **accepted** (0.2 cm over gate) |

The Humanoid Everyday p95 depends on which episodes are sampled: 5.21 cm on the 81-episode sample above, 4.59 cm on an 83-episode sample drawn in a two-dataset run (same seed). Treat it as about 4.6–5.2 cm, i.e. at the gate.

Token feed to the 50 Hz decoder, compared on the same episodes (no-limit datasets, offline harness):

| Feed                                                   | Unitree p95 (56 ep) | HE p95 (83 ep) | HE episodes > 5 cm | Lag  |
| ------------------------------------------------------ | ------------------- | -------------- | ------------------ | ---- |
| Hold each 30 Hz token (current robot code)             | 3.10 cm             | 4.72 cm        | 27%                | 2 fr |
| Look-ahead linear interpolation (needs the next token) | 3.07 cm             | 4.59 cm        | 19%                | 1 fr |
| Delayed interpolation (causal, controller-only)        | 3.19 cm             | 5.08 cm        | 35%                | 2 fr |

Causal interpolation inside the controller adds 33 ms of latency and is worse than holding; only look-ahead interpolation helps.

Official deploy (NVIDIA `g1_deploy_onnx_ref`, decoder only), six episodes × four variants in one session:
no-limit cut the worst episode from 26.9 → 5.0 cm and wrist-orientation p95 from 61.5° → 13°, with no change
in stability (tilt ≤ 4°, all feet in contact). One balance step (tilt 19°, recovered) occurred once in the
first episode after the POSE handoff and did not reproduce in the repeat run.

## Facts worth knowing before touching this pipeline

- **Dex3 fingers never go through SONIC.** They are copied into the 78D action; the token only encodes the 29D body.
- **The decoder's output is a PD setpoint, not a pose.** With the soft official gains (arm kp ≈ 14 Nm/rad) it
  commands 0.2–0.9 rad past the target to fight gravity. Compare the pose the robot reaches, never the setpoint.
- **SONIC picks its own arm posture.** Shoulder joints can differ from the dataset by up to ~0.8 rad while the palm
  lands within ~2 cm (elbow swivel). Check wrist orientation too, not only position.
- Encoder mode 0 (`g1`) takes **absolute** joint targets in IsaacLab order (`ISAAC_FROM_MOTOR`), 10 future frames
  × 5 ticks at 50 Hz. Mode 1 (`teleop`, 3-point wrist/torso targets) is the untested alternative.
- Decoder input (994D): token + 10-frame history (gyro, `q − default_angles`, `dq`, last raw action) + gravity,
  oldest→newest, IsaacLab order; target = `default_angles + action[isaaclab_to_mujoco] × action_scale`
  (constants in `gear_sonic_deploy/.../policy_parameters.hpp`; `default_angles` = `NOMINAL_BODY`).
- Converter joint-limit clipping only touches Dex3 fingers and by ≤0.001 rad (float jitter); arms are never clipped.
- Humanoid Everyday contains single-frame pose jumps (up to ~1 rad in one frame); they explain the max error,
  not the p95.
- Model pair: encoder `fb97de22…`, decoder `34bae857…`, config `4a67713b…` (`/sonic-model` = SONIC v1.1).
  `lerobot/sonic_decoder` and the HF-cache `nvidia/GEAR-SONIC` ONNX files have **different** hashes.

## Official sim2sim procedure (confirmed)

1. Start `jihun/sonic-vla-sim:startupfix-20260907` with the v1.1 ONNX mounted into
   `gear_sonic_deploy/policy/sonic_v1_1/` and `planner_sonic.onnx` into `planner/target_vel/V2/`.
2. The robot hangs on the elastic band; wait for the deploy to print `Init Done`
   (the first TensorRT engine build takes about 190 s).
3. Start control in planner mode (`]` then Enter with keyboard input, or a ZMQ `command` with `planner=true`).
4. Sim key `9` releases the band, then `Backspace` resets the robot onto the ground; it settles in about 3 s
   (pelvis 0.787 m, 8 foot contacts).
5. Token replay (like `gear_sonic/scripts/run_vla_inference.py`, `initial_pose=standing`):
   `--input-type zmq_manager`, PUB on tcp 5556, send `LATENT_INITIAL_MOTION_TOKEN`, then `command` with
   `planner=false` (POSE mode), then protocol-v4 `pose` messages (`token_state`, `frame_index`, hand joints) at 50 Hz.
   The deploy logs `[Token Flow] Copied external tokens` when its encoder is bypassed.

## Deploying a trained policy (decided 2026-09-24)

Runtime: **NVIDIA's C++ `g1_deploy_onnx_ref`** (v1.1 decoder, TensorRT at 50 Hz, token watchdog, startup ramp,
emergency stop), fed over ZMQ by a Python streamer, as in `gear_sonic/scripts/run_vla_inference.py`.
LeRobot's `SonicWholeBodyController` is not used: it loads `lerobot/sonic_decoder`, which is not the v1.1 decoder
paired with our tokens, and it holds the last token with no watchdog.

- The streamer publishes at 50 Hz. For each tick it takes `ChunkResampler.token_at(t)` (`sonic_token_stream.py`),
  which linearly interpolates the policy's 30 Hz chunk with look-ahead and holds the last token when a chunk runs out.
- Do not use LeRobot's `interpolation_multiplier` for SONIC tokens: `ActionInterpolator` is the causal form
  (blend from the previous action after a new one arrives), which measured worse than holding.
- Hands: send the chunk's 14 Dex3 values (at the same source frame) in the protocol-v4 message.
- Verified in the official deploy (2026-09-24, same six episodes, no-limit tokens): the chunked stream (40-token
  chunks every 0.4 s from 0.1 s old observations, `ChunkResampler`) matched ideal look-ahead within 0.02 cm on every
  episode. Median palm p95: hold 4.49 cm, ideal 4.05 cm, chunked 4.05 cm; tilt ≤ 4.0°, all feet in contact.
  The stand-in chunks come from the dataset, so consecutive chunks agree; chunks from a real policy that disagree
  at boundaries still need testing with a trained policy.

## Streaming a trained policy (2026-09-25)

`sonic_policy_streamer.py` feeds a LeRobot SONIC-token policy (78D, 30 Hz chunks) into the official deploy;
`sonic_official_sim_host.py` runs MuJoCo + deploy and performs the confirmed startup (Init Done → start/planner →
key 9 → Backspace → settle) through gate files; `sonic_stream_eval.py` scores a run against FK of the original
joint action. Images come from the recorded episode (`--images dataset`), so there is no sim-camera gap.

Held-out episodes 2155 and 6 (5% held-out split, seed 1000), palm p50 / p95 in cm, pelvis frame:

| Policy                                | Closed loop (robot state) | Recorded state (`--state-source dataset`) |
| ------------------------------------- | ------------------------- | ----------------------------------------- |
| ACT, `sonic78_nolimit`                | 10.7/21.6 · 7.6/17.7      | 2.6/6.9 · 2.6/12.9                        |
| **ACT, `sonic78_nolimit_sonicstate`** | **7.4/16.6 · 6.7/20.4**   | **2.1/5.7 · 2.3/9.7**                     |
| Diffusion, `sonic78_nolimit`          | 11.5/21.8 · 12.1/24.4     | –                                         |

- SONIC executes the policy's tokens well (recorded-state runs are near the round-trip error). The closed-loop
  drift comes from the policy: it copies its state input (open-loop ablation `sonic_state_ablation.py`), and in
  closed loop the robot state disagrees with the recorded images. Relabelling the state reduces the drift.
- Diffusion (2-frame state history) drifts as much as ACT and was the least stable run (ep 6: tilt 4.7°,
  contacts down to 5). DDPM inference takes ~0.7 s per chunk; use `--max-chunk-age-s 2`.
- All streamed episodes stayed standing (10/10 with the 2 s handoff). Expect the real robot between the
  closed-loop and recorded-state columns, depending on how far its state departs from the training state.

## Retrained models on relabelled state (2026-09-26)

All seven policies were retrained on relabelled state for both datasets (5% held-out split, same configs as the
`*_sonic78nolimit_ho5_full` runs except dataset and names), 40K steps each, all exit 0:
`/run-output/runs/{act,diffusion,groot,vla_jepa,pi05,molmoact2,fastwam}_sonic78sonicstate_ho5_full` and the same
names under `/run-output/humanoid_everyday_g1_20260923/runs/`. The HE relabelled dataset is
`humanoid_everyday_g1_20260923/datasets/sonic78_nolimit_sonicstate` (built from `sonic78_nolimit_g2`, 4064 ep,
1,779,287 frames, no falls, exact quantiles). Queue: `/run-output/queue_sonicstate` (one `worker.py` per GPU
container, jobs claimed with `*.claim` files).

H100 run storage (2026-09-26, user-approved): each run keeps only `checkpoints/last/pretrained_model` (halfway
checkpoints and optimizer state deleted; runs cannot be resumed). Remaining: `{policy}_sonic78nolimit_ho5_full`
(baselines; HE ACT baseline is `act_sonic78nolimit_ho5_g2_full`) and `{policy}_sonic78sonicstate_ho5_full`.

Pitfalls met on the way:

- Long-running training containers can lose the GPU (`Failed to initialize NVML`); PyTorch then silently falls
  back to CPU (158 s/step). `docker restart` the container; the queue worker refuses to start a job without CUDA.
- The disk filled (98%) mid-queue; the worker waits while free space is below 150 GB.

## Tabletop startup (in progress, 2026-09-26)

Real-robot evaluation is at a table: top 80 cm above the floor, near edge 5–12 cm from the torso. Measured in the
official deploy sim (12/12 runs):

| Pose                                                   | Hand tips above floor | In front of pelvis |
| ------------------------------------------------------ | --------------------- | ------------------ |
| Planner stance (after SONIC starts)                    | 56–59 cm              | 13–19 cm           |
| After the 2 s handoff to `LATENT_INITIAL_MOTION_TOKEN` | 77–85 cm              | 28–36 cm           |
| Fixed initial pose (median Unitree first frame)        | 85–92 cm              | 38–41 cm           |

NVIDIA's standing token lifts the hands into the table, so at the table the streamer must skip it: planner
stance → 2 s gradual switch → table-safe joint-space arm path → fixed initial pose → policy. In the planner stance
the wrists are already ~2.6 cm from the table's underside edge, and the initial pose clears the top by only
~2.7 cm (kinematic, collision geoms).

`sonic_table_startup.py`: `plan` searches two arm waypoints (hands back behind the edge, then raised above the
top) keeping every arm collision geom ≥ 3 cm from the table (no closer than the stance at the start, down to the
initial pose's own clearance at the end) and clear of the robot's body, smoothstep segments capped at 0.5 rad/s,
then encodes the path with the dataset converter; `check` replays the tokens through the v1.1 decoder in MuJoCo
with a physical table at 5 / 8.5 / 12 cm and reports reached clearance, contacts, tilt, and foot contacts.
Status: first planner run in progress; not yet wired into the streamer.

## Scripts (`examples/g1_dex3_training/`)

Written to run on H100 inside containers; `/code` = this directory, `/audit` = the output directory.

| Script                                                                                 | Purpose                                                                                                                                                                                       |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sonic_targets.py`, `prepare_sonic_dataset.py`                                         | Converter; speed limits and source fps are now parameters (defaults unchanged, bit-exact with `sonic78`).                                                                                     |
| `sonic_roundtrip_audit.py`                                                             | Offline harness: encoder → stored token → decoder closed loop in MuJoCo → palm FK. Options: `--conv-suffix _nolimit`, `--interp`, `--start placed/hang/hang_leadin`, `--noslew`, `summarize`. |
| `sonic_roundtrip_{breakdown,temporal,paired,init,export}.py`                           | Error sources, error over time, with/without limiter, initial pose, viewer export.                                                                                                            |
| `sonic_official_startup.py`                                                            | Records the official startup (video + telemetry).                                                                                                                                             |
| `sonic_official_replay.py`, `sonic_replay_{extract,variants,compare,variant_table}.py` | Stream tokens into the official deploy and compare with the original actions.                                                                                                                 |
| `sonic_token_stream.py`                                                                | `ChunkResampler`: 30 Hz policy token chunks → 50 Hz deploy ticks (look-ahead). Tested in `tests/datasets/test_g1_dex3_sonic_token_stream.py`.                                                 |
| `sonic_policy_streamer.py`, `sonic_official_sim_host.py`                               | Stream a trained policy into the official deploy; sim host with the confirmed startup and recording.                                                                                          |
| `sonic_stream_eval.py`, `sonic_policy_openloop.py`, `sonic_state_ablation.py`          | Score a streamed run; open-loop policy check; which state part throws the policy off.                                                                                                         |
| `sonic_state_relabel.py`                                                               | `simulate` + `build`: relabel `observation.state` arms with the SONIC-reached pose (`sonic78_nolimit_sonicstate`, 3152 ep, no falls).                                                         |
| `sonic_table_startup.py`                                                               | Table-safe startup path (planner stance → fixed initial pose) and its MuJoCo table check. In progress.                                                                                        |
| `sonic_nolimit_verify.py`                                                              | Checks a published no-limit dataset against its source and the replay-tested tokens.                                                                                                          |

Offline audit of a no-limit dataset (lerobot image `4cbe2a3f7fc6`, python `/run-output/environment/venv/bin/python`,
`pip install --target /audit/pylib mujoco==3.3.0`, `PYTHONPATH=/audit/pylib MUJOCO_GL=disable`):

```bash
python /code/sonic_roundtrip_audit.py --out /audit/results_nolimit --sets unitree humanoid_everyday \
  --per-task 4 --extremes 4 --conv-suffix _nolimit --interp
python /code/sonic_roundtrip_audit.py summarize /audit/results_nolimit
```

Outputs live in `/mnt/data01/jhkim/model_weight/sonic_roundtrip_20260923/` on the H100 host.
Reports: `2026-09-23-sonic-roundtrip-audit.html` (in this directory).

## Open items (carry on from here)

1. **Tabletop startup:** finish `sonic_table_startup.py plan`, pass `check` at 5 / 8.5 / 12 cm (no table contact,
   reached clearance > 0, balance), add `--startup-tokens` to the streamer (planner token → 2 s blend to the path's
   first token → path → 1 s blend to the first policy chunk), verify in the official deploy sim with a table and a
   video. Then expand the fixed initial pose to per-task poses.
2. **Held-out sim test of the 14 Unitree models** (7 baselines vs 7 relabelled, episodes 2155 and 6, closed loop):
   rank by palm error, stability, and chunk latency vs the 0.4 s replan to pick real-robot candidates. A smoke test
   of every model through `ChunkPolicy` is running.
3. Real-robot test of the chosen Unitree model at the table (needs 1 and 2; decide where the policy runs: H100 over
   the network or a GPU at the robot).
4. Dex3 finger tracking is only indicative in sim; NVIDIA flags its hand model as unstable.
