# G1 closed-loop task evaluation in Isaac Lab with the SONIC deploy (design)

Date: 2026-10-14. Status: approved in chat (goal, architecture, components, data flow, errors, testing), written spec
under review.

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
  `settled.i`, `stop.i`, `verdict.i`). The Isaac host writes those files; the streamer and the session format do not
  change.
- **Task: handover_bottle**, held-out episodes 27 and 35 of `handover_bottle_sonic78_50hz_261002` as references.

## Architecture

```
sonic_policy_server.py (GPU 1)  <-ZMQ->  sonic_policy_streamer.py --trials N --gate-dir G --session-dir S
                                              |  ZMQ command/pose (5556)              ^ head camera (ZMQ, as on the robot)
                                              v                                       |
                                     g1_deploy_onnx_ref (SONIC, unchanged)            |
                                              |  Unitree DDS on lo: rt/lowcmd, rt/lowstate,
                                              |  rt/dex3/{left,right}/{cmd,state}, rt/secondary_imu, rt/odostate
                                              v                                       |
                         sonic_official_sim_host.py SIM=isaac  ->  isaac_scene.Scene (GPU 0)
                                              |  gate files settled.i / stop.i / verdict.i
                                              v
                                       handover_success.Judge
```

Everything above the DDS line is what runs today against MuJoCo. The DDS messages are packed by gear_sonic's
`UnitreeSdk2Bridge` (`gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py`): `PublishLowState(obs)` takes a plain dict
and `LowCmdHandler` stores the command; MuJoCo appears there only in a debug printout, so the Isaac scene feeds it the
same dict.

## Components

All in `examples/g1_dex3_training/` unless noted.

| File | Change | What it does |
| --- | --- | --- |
| `isaac_scene.py` | new | `Scene` with the methods of `sim_scene.Scene` (`step`, `mark`, `place_table`, `record`, `check_fall`, `pace`, `close`, `floor_contacts`, `table_clearance`) plus `reset_trial(seed)` and `bottle_pose()`. Builds an Isaac Lab scene: G1 43-dof (29 body + Dex3) from GR00T-WholeBodyControl's Isaac Lab robot config, the table of `sim_scene.py` (center z 0.785 m, near edge `TABLE_GAP_CM` from the torso), a bottle (rigid cylinder, default r 3.3 cm, h 20 cm, 0.5 kg; values in the task config), the head camera. Converts `rt/lowcmd` and the Dex3 commands to joint torques with the same PD law and physics step as gear_sonic MuJoCo (`tau = kp (q* - q) + kd (dq* - dq) + tau_ff`, `SIMULATE_DT` from the same `SimLoopConfig`). Publishes the head image with the protocol of `g1_head_camera_server.py`. |
| `sonic_official_sim_host.py` | edit | `SIM=mujoco` (default, current behaviour) or `SIM=isaac` selects the Scene class. In `--trials` mode, after each trial: `reset_trial(seed)`, wait until settled, write `settled.i`; run the judge every step; write `stop.i` and `verdict.i` when the judge decides. Deploy launch, gates and termination logic stay shared. |
| `handover_success.py` | new | `Judge` (pure numpy, no Isaac import): fed per step with bottle pose, table top height, palm positions, fall state, watchdog state, elapsed time; returns `None` (running) or a verdict dict in `eval_session` gate format. Rules below. |
| `cameras/d435i_head.json` | new | Head camera: parent link `d435_link` (present in the G1 model), 640 x 480 colour, `fx fy cx cy` and distortion. Starts with nominal D435i colour values; **calibration knob**: replace with `rs-enumerate-devices -c` from the robot's camera at 640 x 480. Loaded through the OpenCV-intrinsics path of GR00T-WholeBodyControl's Isaac Lab env (`_load_opencv_params_from_usd` / `OpenCVCameraCfg`). |
| `sonic_official_sim_eval.sh` | edit | `SIM=isaac` selects the local mode: no SSH; host in the `sonic` env on `ISAAC_GPU` (default 0), policy server on `POLICY_GPU` (default 1); passes `--trials`, `--gate-dir`, `--session-dir`, `--seed`. |
| `test_handover_success.py` | new | Assert-based check of the judge on synthetic trajectories. |

### Judge rules

Constants are in one dict at the top of `handover_success.py` (calibration knob; tune against robot sessions).

- **success**: bottle bottom >= 10 cm above the table top continuously for 1.0 s with the robot not fallen; the
  trial stops at that moment (what happens after the hold is not judged).
- **fail**, with the `eval_session.DEFAULT_FAILURE_REASONS` index:
  - 1 "did not reach the object": no palm came within 5 cm of the bottle by the time limit.
  - 2 "grasp failed / missed": a palm reached the bottle but it never left the table by 10 cm.
  - 3 "dropped the object": lifted >= 10 cm, then back below 2 cm above the table or on the floor before 1.0 s of hold.
  - 5 "collision (table, knocked over)": bottle tilted > 60 deg while on the table.
  - 6 "stalled / froze": time limit (default 40 s of streamed policy time) with none of the above.
  - 7 "safety stop (watchdog)": the streamer's arm-speed watchdog ended the episode.
  - fall: the existing `FallDetector`; reason 8 "other" with note `fell`.
- **redo**: the run is invalid by `wbc_common`'s validity rule (termination not completed or < 98 % of frames
  covered) or the deploy hung.

## Data flow per trial

1. `reset_trial(seed)`: robot joints and root to the deploy's standing pose, velocities zero; bottle at its nominal
   position + uniform +-3 cm in x and y (seeded by session seed + trial index).
2. The deploy holds planner standing; after 5 s settled, the host writes `settled.i`.
3. The streamer switches to POSE and streams the policy at 50 Hz on the rendered head image and the 28-D state.
4. The judge runs every physics step; on a verdict the host writes `stop.i`, then `verdict.i`.
5. `eval_session.py` writes `trial_{i:03d}.npz`, the frames and `results.jsonl`; the host writes `sim_state_{i:03d}.npz`
   (the 50 Hz `REC_KEYS` record plus bottle pose) and `video_{i:03d}.mp4` next to them.

Default: 10 trials per checkpoint, session seed 0.

## Errors

- Startup checks fail loudly: the joint names of the Isaac articulation must contain `sim_scene.BODY` and the Dex3
  joint names in the order the bridge expects; the deploy must print "Init Done" within its existing timeout; the
  camera image must be 640 x 480.
- A trial reset that leaves the robot fallen within the 5 s settle is retried once, then the session stops with
  `termination.json` reason `reset_failed` (no silent skipping).
- Everything else reuses the existing paths: `FallDetector`, the deploy-hang timeout, `--policy-timeout-s`, redo.

## Testing

1. **Judge unit check** (`test_handover_success.py`): lift-and-hold -> success; lift-and-drop -> fail 3; never moved ->
   fail 1 or 2 by palm distance; tipped -> fail 5; timeout -> fail 6.
2. **Replay parity before any policy**: `REPLAY=1` episodes 27 and 35 through `SIM=isaac`; the run must be valid and
   pass the palm-tracking gates G0-G2 of `wbc_compare.py`, as MuJoCo backend A does. This checks the physics swap,
   PD law, joint order and DDS wiring.
3. **Reset stability**: 10 consecutive `reset_trial` calls with the policy idle (planner standing), no fall.
4. **Closed loop**: GR00T he and Pi0.5 he (41-episode handover checkpoints), 10 trials each. Pass criterion of this
   milestone: the sessions complete and open in the visualizer; the success rates are reported next to the robot
   results. Agreement of the ranking is reported, not required.

## Out of scope (add when)

- Inspire hands: when an Inspire action format for SONIC exists.
- Randomization beyond bottle position (lighting, textures, camera pose): if sim and robot rankings disagree.
- Other tasks (pnp_bottle, pick_cup): after handover parity; each adds a task config and judge rules.
- Isaac Lab 3.0 / IsaacLab-Arena: when GR00T-WholeBodyControl supports it.
- H100 / container runs: if local throughput is too slow.

## Risks

- The 3060 renders one 640 x 480 camera plus physics; if Isaac cannot keep 50 Hz policy time, the host paces sim time
  to the streamer (as `Scene.pace` does) rather than dropping frames.
- PD law or actuator mismatch between Isaac and MuJoCo shows up in test 2; fix there before running policies.
- Mid-session resets under a running deploy are new; test 3 covers it. If it cannot be made stable, fall back to one
  deploy + scene start per trial (slower, same outputs).
