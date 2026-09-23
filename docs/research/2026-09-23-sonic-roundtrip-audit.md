# G1 Dex3 → SONIC v1.1 conversion: round-trip audit, decisions, and how to reproduce

Status as of 2026-09-24. Question: does converting joint-action datasets (28D: arms 14 + Dex3 hands 14) into
SONIC tokens (78D: token 64 + hands 14) keep the hand where the original action put it? Gate: p95 palm
position error vs. the original command > 5 cm in the pelvis frame ⇒ unusable for VLA fine-tuning.
Evidence is simulation-only (MuJoCo); no real-robot claim.

## Decisions (user-confirmed)

| Topic                       | Decision                                                                                                                                                                                                                                                       |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Converter speed limits      | **Off.** `prepare_sonic_dataset.py --arm-speed-limit none --hand-speed-limit none`. The old 1 rad/s arm / 2 rad/s hand slew limit caused most of the error.                                                                                                    |
| Episode start in evaluation | 1 s linear token blend from `LATENT_INITIAL_MOTION_TOKEN` to the episode's first token, 1 s hold.                                                                                                                                                              |
| Token rate                  | Train on the 30 Hz dataset. At deploy, linearly interpolate the policy's 30 Hz tokens to 50 Hz (resample the predicted action chunk; interpolation needs the next token). Re-encoding at 50 Hz gave no gain (≤0.1 cm).                                         |
| Humanoid Everyday dataset   | **Accepted as-is (2026-09-24)** despite p95 5.21 cm, 0.2 cm over the 5 cm gate: 20% of sampled episodes exceed 5 cm on their own p95 (worst 8.6 cm); no episodes are excluded. The miss comes from SONIC tracking fast manipulation, not from source glitches. |

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

## Scripts (`examples/g1_dex3_training/`)

Written to run on H100 inside containers; `/code` = this directory, `/audit` = the output directory.

| Script                                                                                 | Purpose                                                                                                                                                                                       |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sonic_targets.py`, `prepare_sonic_dataset.py`                                         | Converter; speed limits and source fps are now parameters (defaults unchanged, bit-exact with `sonic78`).                                                                                     |
| `sonic_roundtrip_audit.py`                                                             | Offline harness: encoder → stored token → decoder closed loop in MuJoCo → palm FK. Options: `--conv-suffix _nolimit`, `--interp`, `--start placed/hang/hang_leadin`, `--noslew`, `summarize`. |
| `sonic_roundtrip_{breakdown,temporal,paired,init,export}.py`                           | Error sources, error over time, with/without limiter, initial pose, viewer export.                                                                                                            |
| `sonic_official_startup.py`                                                            | Records the official startup (video + telemetry).                                                                                                                                             |
| `sonic_official_replay.py`, `sonic_replay_{extract,variants,compare,variant_table}.py` | Stream tokens into the official deploy and compare with the original actions.                                                                                                                 |
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

## Open items

- Robot-side 30 → 50 Hz token interpolation: only the look-ahead form helps (see the feed table), so it needs the next token from the policy's action chunk. Not implemented yet.
- The balance step after the POSE handoff needs repeated trials before any real-robot run.
- Dex3 finger tracking is only indicative in sim; NVIDIA flags its hand model as unstable.
