# G1 closed-loop task evaluation in Isaac Lab with the SONIC deploy (design)

Date: 2026-10-14. Status: approved in chat (goal, architecture, components, data flow, errors, testing); revised
2026-10-15 after a code review against `sonic_policy_streamer.py`, `eval_session.py`, `sonic_official_sim_host.py`,
`sim_scene.py`, `wbc_common.py`, `wbc_compare.py` and gear_sonic `base_sim.py` / `unitree_sdk2py_bridge.py`. Written
spec under review.

## Goal

Score sonic78 checkpoints (GR00T, Pi0.5, Psi0, XR-1, ...) on the **handover_bottle** task in simulation with an
automatic success / fail verdict, so that a sim session can be compared with the real-robot sessions of the same
checkpoints (handover, 41 episodes: GR00T he 5/5, Pi0.5 he 3/5).

Success: one command runs N trials of one checkpoint in Isaac Lab and produces the same session directory
(`trial_{i:03d}.npz`, `trial_{i:03d}_frames.npz`, `results.jsonl`) as a real-robot `--trials` session, with an automatic
verdict and failure reason per trial; `eval_session_to_lerobot.py` and the dataset visualizer open it unchanged.

## Decisions

- **Policies: sonic78 only** (64 SONIC tokens + 14 Dex3 hand targets, one head camera
  `observation.images.egocentric`, 28-D state). joint28/joint31 and Inspire are out of scope.
- **Body controller: the unchanged NVIDIA SONIC C++ deploy** (`g1_deploy_onnx_ref`, `sonic_v1_1`), the binary used on
  the real robot and in the MuJoCo backend A. Only the physics behind its Unitree DDS interface changes.
- **Isaac Lab 2.3.2 / Isaac Sim 5.1**, the version GR00T-WholeBodyControl pins. The existing `sonic` conda env already
  has both. Isaac Lab 3.0 / IsaacLab-Arena: not now; port when GR00T-WholeBodyControl moves off 2.3.2.
- **Runs on the workstation**: Isaac on GPU 0, `sonic_policy_server.py` on GPU 1 (GR00T he and Pi0.5 he in bfloat16
  already fit a 12 GB RTX 3060). No SSH, no container. The H100 path stays as it is for MuJoCo.
- **Automatic judge = the operator.** `eval_session.py` already takes verdicts from files (`--gate-dir`:
  `settled.i`, `stop.i`, `verdict.i`). The Isaac host writes those files.
- **One small streamer change**: in `--trials` mode with `--gate-dir`, the streamer writes `ended.{i}` (the
  `episode_end` string) before it waits in `gate_verdict()`. Today `write_done` runs only on the non-trials path
  (`sonic_policy_streamer.py` around L1069), so the host cannot learn a watchdog stop, the `--duration-s` end, "no
  valid chunk" or "robot state old", and the streamer would wait for `verdict.i` forever.
- **The trial time limit is the streamer's `--duration-s`** (40 s), not a judge timer: the host does not know when
  streaming starts (after `settled.i` come the handoff blend, `--hold-rest-s` and the first inference).
- **Success = pick and carry the bottle into a handover zone** (no human in the sim to take it), not pick alone, so the
  sim measures the task the robot sessions were judged on.
- **Task: handover_bottle**, held-out episodes 27 and 35 of `handover_bottle_sonic78_50hz_261002`.

## Architecture

```
sonic_policy_server.py (GPU 1)  <-ZMQ->  sonic_policy_streamer.py --trials N --gate-dir G --session-dir S --duration-s 40
                                              |  ZMQ command/pose (5556)              ^ head camera "egocentric" (ZMQ, as on the robot)
                                              v                                       |
                                     g1_deploy_onnx_ref (SONIC, unchanged)            |
                                              |  Unitree DDS on lo: rt/lowcmd, rt/lowstate,
                                              |  rt/dex3/{left,right}/{cmd,state}, rt/secondary_imu, rt/odostate
                                              v                                       |
                         sonic_official_sim_host.py SIM=isaac  ->  isaac_scene.Scene (GPU 0)
                                              |  gate files: host writes settled.i / stop.i / verdict.i,
                                              |              streamer writes ended.i
                                              v
                                       handover_success.Judge
```

Everything above the DDS line is what runs today against MuJoCo. The DDS messages are packed by gear_sonic's
`UnitreeSdk2Bridge` (`gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py`): `PublishLowState(obs)` takes a plain dict
and `LowCmdHandler` stores the command; MuJoCo appears there only in a debug printout, so the Isaac scene feeds it the
same dict.

### Bridge dict: Isaac source per key

| Key | MuJoCo source (gear_sonic) | Isaac source |
| --- | --- | --- |
| `body_q`, `body_dq` | `qpos` / `qvel` of the 29 body joints | articulation `joint_pos` / `joint_vel`, body joints in `sim_scene.BODY` order |
| `body_ddq` | `qacc` of the body joints | `joint_acc` |
| `body_tau_est` | applied (clipped) torques | the clipped torques this scene applied in the last step |
| `floating_base_pose`, `floating_base_vel` | root `qpos[:7]` / `qvel[:6]` | `root_state_w` (pose, lin/ang velocity; quaternion order converted to the bridge's) |
| `floating_base_acc` | `qacc[:6]`, no gravity term | finite difference of the root velocity over one step (no gravity term) |
| `secondary_imu_quat`, `secondary_imu_vel` | torso IMU site | `torso_link` body orientation and angular velocity |
| odometry (`rt/odostate`) | root position / velocity | root position / velocity |
| `left/right_hand_q`, `_dq` | the 14 Dex3 joints | articulation Dex3 joints, in the bridge's order |
| `time` | sim time | sim time |

## Components

All in `examples/g1_dex3_training/` unless noted.

| File | Change | What it does |
| --- | --- | --- |
| `isaac_scene.py` | new | `Scene` with the methods of `sim_scene.Scene` (`step`, `mark`, `place_table`, `record`, `check_fall`, `pace`, `close`, `floor_contacts`, `table_clearance`) plus `reset_trial(seed)`, `bottle_state()` and `cut_trial(i)`. Builds an Isaac Lab scene: G1 43-dof (29 body + Dex3) from GR00T-WholeBodyControl's Isaac Lab robot config; the table of `sim_scene.py` (center z 0.785 m, near edge `TABLE_GAP_CM` from the torso); a bottle (rigid cylinder, default r 3.3 cm, h 20 cm, 0.5 kg); the head camera. Physics dt 0.005 s (`SIMULATE_DT`). Actuators in effort mode (stiffness 0, damping 0); the scene computes `tau = kp (q* - q) + kd (dq* - dq) + tau_ff` from `rt/lowcmd` and the Dex3 commands, clipped to `MOTOR_EFFORT_LIMIT_LIST` (body, as in `base_sim.py` L423) and `SimLoopConfig.hand_torque_limit` (hands, `configs.py`); per-joint armature, damping and friction copied from gear_sonic's G1 MJCF. Camera rendered at 30 Hz (`sim_scene.FPS`), not every physics step, and published with the protocol of `g1_head_camera_server.py` under the name `egocentric`. |
| `tasks/handover_bottle.json` | new | Task config: bottle size and mass, bottle nominal position (table-relative), handover zone (robot-relative), jitter, judge constants, and the replay run IDs these values come from (below). |
| `sonic_official_sim_host.py` | edit | `SIM=mujoco` (default, current behaviour) or `SIM=isaac` selects the Scene class. New env `TRIALS`, `SESSION_DIR`, `SEED` (it keeps its positional `OUT GATE`). In trials mode it starts at `eval_session.next_trial(SESSION_DIR)` and, per trial: `reset_trial(seed + i)`, wait settled, write `settled.i`; feed the judge every physics step; write `stop.i` on an early verdict; on `ended.i` finalize the verdict, write `verdict.i`, call `cut_trial(i)`. `i` advances only on a non-redo verdict, as in the streamer. |
| `sonic_policy_streamer.py` | edit | `--trials` + `--gate-dir`: write `ended.{i}` with `episode_end` before `gate_verdict()` (few lines; nothing else changes). |
| `handover_success.py` | new | `Judge` (pure numpy, no Isaac import): fed per step with bottle pose, table top height, palm positions, robot fallen flag; `early()` returns a verdict that ends the trial now (success, knocked over, on floor, fall) or `None`; `final(episode_end)` classifies the trial when `ended.i` arrives. Rules below. |
| `cameras/d435i_head.json` | new | Head camera: parent link `d435_link`, 640 x 480 colour, `fx fy cx cy` and distortion. Starts with nominal D435i colour values; **calibration knob**: replace with `rs-enumerate-devices -c` from the robot's camera at 640 x 480. Loaded through the OpenCV-intrinsics path of GR00T-WholeBodyControl's Isaac Lab env (`_load_opencv_params_from_usd` / `OpenCVCameraCfg`). The recordings' camera is a D435 (`ego_view`, renamed to `egocentric` by `prepare_wbt_sonic78.py`). |
| `wbc_compare.py` | edit | New gate `GI` (Isaac parity, below), next to G0-G2. |
| `sonic_official_sim_eval.sh` | edit | `SIM=isaac` selects the local mode: no SSH; host in the `sonic` env on `ISAAC_GPU` (default 0) with `TRIALS`, `SESSION_DIR`, `SEED`; policy server on `POLICY_GPU` (default 1); streamer with `--trials`, `--gate-dir`, `--session-dir`, `--duration-s 40`. |
| `test_handover_success.py` | new | Assert-based check of the judge on synthetic trajectories. |

### Bottle position and handover zone (source of the task config)

From the existing MuJoCo backend A replays (`REPLAY=1`) of held-out episodes 27 and 35:

- **Grasping hand**: the hand whose Dex3 targets close (cross 50 % of their range) first.
- **Bottle nominal position**: that hand's palm position at the close frame, projected to the table top, stored
  relative to the table (x from the near edge, y from the torso line). Mean of the two episodes.
- **Handover zone**: sphere of radius 8 cm around the grasping palm's position at the episode's last frame, stored
  relative to the pelvis at that frame. Mean of the two episodes.
- The JSON keeps both per-episode values and the run IDs, so it can be recomputed.

### Judge rules

Constants are in `tasks/handover_bottle.json` (calibration knob; tune against robot sessions). Palm = mean of the
index_0 / middle_0 link origins (as `sim_scene.py`). Reach distance = palm to the bottle's axis segment; threshold
`r + 4 cm` (7.3 cm), checked in test 3 against the grasp frames of the replays.

- **lifted**: bottle bottom >= 10 cm above the table top.
- **success** (early, ends the trial): lifted and bottle centre inside the handover zone continuously for 1.0 s, robot
  not fallen.
- **early fails** (end the trial at once), with the `eval_session.DEFAULT_FAILURE_REASONS` index:
  - 5 "collision (table, knocked over)": before any lift, the bottle tilts > 60 deg or leaves the table top.
  - 3 "dropped the object": after a lift, the bottle is on the floor.
  - 8 "other", note `fell`: the existing `FallDetector`.
- **final classification** on `ended.i` without an early verdict:
  - `watchdog: ...` -> 7 "safety stop (watchdog)".
  - `no valid chunk ...` / `robot state ... old` -> **redo**.
  - no palm ever within the reach distance -> 1 "did not reach the object".
  - reached, never lifted -> 2 "grasp failed / missed".
  - lifted, now within 2 cm of the table top -> 3 "dropped the object".
  - lifted and still lifted at the end, never 1.0 s in the zone -> 4 "wrong object / target", note `not in handover zone`.
  - otherwise (lifted, then between 2 and 10 cm above the table, never held) -> 6 "stalled / froze". Reason 6 is
    reachable only by this case.

### Redo (decided live, before `verdict.i`)

`wbc_common.run_validity` needs the termination record, frame coverage and wall times, which exist only after the
trial; `verdict.i` is needed before `trial_{i}.npz` is written. Redo is therefore decided from what the host knows
live: the deploy hung (existing timeout), or `ended.i` reports "no valid chunk" / "robot state old". `run_validity` is
still computed afterwards per trial and reported; a trial that passes the live checks but fails `run_validity` is
flagged in the summary, not re-judged.

## Data flow per trial

1. `reset_trial(seed + i)`: robot joints and root to the deploy's standing pose, velocities zero; bottle at its nominal
   position + uniform +-3 cm in x and y.
2. The deploy holds planner standing; after 5 s settled, the host writes `settled.i`.
3. The streamer blends to POSE, holds rest, then streams the policy at 50 Hz on the head image and the 28-D state, up
   to `--duration-s`.
4. The judge runs every physics step. On an early verdict the host writes `stop.i`. The streamer ends the episode
   (operator stop, duration, watchdog or error) and writes `ended.i`; the host writes `verdict.i` (early verdict or
   `final(episode_end)`).
5. `eval_session.py` writes `trial_{i:03d}.npz`, the frames and `results.jsonl`. The host's `cut_trial(i)` writes
   `sim_state_{i:03d}.npz` (the 50 Hz `REC_KEYS` record plus bottle pose) and `video_{i:03d}.mp4` and clears its
   buffers; a redo of trial i overwrites both.

Default: 10 trials per checkpoint, session seed 0. Resume: rerun with the same `SESSION_DIR`; host and streamer both
start at `next_trial(SESSION_DIR)`.

## Errors

- Startup checks fail loudly: the Isaac articulation contains `sim_scene.BODY` and the Dex3 joints in the bridge's
  order; the robot has `d435_link` (gear_sonic uses it only when present); the camera publishes 640 x 480 under
  `egocentric`; the deploy prints "Init Done" within its existing timeout.
- A trial reset that leaves the robot fallen within the 5 s settle is retried once, then the session stops with
  `termination.json` reason `reset_failed` (no silent skipping).
- Everything else reuses the existing paths: `FallDetector`, the deploy-hang timeout, `--policy-timeout-s`.

## Testing (in this order)

0. **Real-time factor**: the deploy and the streamer run on wall clock at 50 Hz; `Scene.pace` only waits when the sim
   is ahead and cannot slow the deploy. With DDS running and the camera rendering at 30 Hz, robot standing in planner
   mode for 60 s: sim time / wall time >= 1.0 for every 1 s window. If this fails on the 3060 the closed loop is
   invalid here; stop and move to the H100 before anything else.
1. **Judge unit check** (`test_handover_success.py`): lift, carry, hold in zone -> success; lift then floor -> 3
   (early); lift then back on table -> 3 (final); never near -> 1; near, never lifted -> 2; tipped before lift -> 5;
   lifted, held outside the zone to the end -> 4; lifted, dipped to 5 cm, ended -> 6; `watchdog: ...` -> 7;
   `no valid chunk` -> redo.
2. **Replay parity, gate `GI`**: `REPLAY=1` episodes 27 and 35, 3 repeats each, `SIM=isaac` vs MuJoCo backend A.
   Pass: every run valid; recorded-action palm error |Isaac - MuJoCo| <= 1.0 cm at p50 and <= 2.0 cm at p95 per
   episode; mean max tilt <= MuJoCo + 2 deg (G1-style balance). G0 and G2 do not apply (G0 compares A with stored A,
   G2 compares backend C with A).
3. **Task config and reach threshold**: derive `tasks/handover_bottle.json` from the replays; in the Isaac replays the
   grasping palm passes the reach distance at the close frame.
4. **Reset stability**: 10 consecutive `reset_trial` calls with the policy idle (planner standing): no fall, and the
   planner's heading and position (odometry) do not cause the robot to step or turn after the teleport. If they do,
   use the fallback below.
5. **Closed loop**: GR00T he and Pi0.5 he (41-episode handover checkpoints), 10 trials each. Pass: both sessions
   complete and open in the visualizer, and the verdicts are not trivial (at least one success over the two sessions,
   and not all trials reason 1, which would point to a broken camera or wiring). Success rates are reported next to
   the robot results; agreement of the ranking is reported, not required.

## Out of scope (add when)

- Inspire hands: when an Inspire action format for SONIC exists.
- Randomization beyond bottle position (lighting, textures, camera pose): if sim and robot rankings disagree.
- Other tasks (pnp_bottle, pick_cup): after handover parity; each adds a task config and judge rules.
- Isaac Lab 3.0 / IsaacLab-Arena: when GR00T-WholeBodyControl supports it.
- H100 / container runs: if test 0 fails locally or throughput is too slow.

## Risks and fallback

- **Real time**: see test 0. Rendering only the policy camera at 30 Hz, headless, with GPU 0 dedicated, is the
  budget; there is no slow-motion mode, because the deploy and streamer cannot be slowed.
- **Physics parity**: PD law, torque clip or joint dynamics mismatches show up in test 2; fix there before running
  policies.
- **Mid-session resets** under a running deploy are new (test 4). Fallback without further streamer changes: run the
  whole stack (deploy, host, streamer) once per trial with `--trials i+1` on the same `SESSION_DIR`; the streamer
  resumes from `results.jsonl`. Slower, same outputs. A deploy restart under a running streamer is not an option: the
  streamer sends start / planner only once.
