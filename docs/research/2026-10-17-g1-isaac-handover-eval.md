# G1 handover_bottle in Isaac Lab, closed loop through the SONIC deploy (2026-10-17)

This page reports the Isaac Lab evaluation of sonic78 handover policies behind NVIDIA's unchanged SONIC C++ deploy, with
an automatic verdict per trial in the robot's eval-session format. Design:
`docs/superpowers/specs/2026-10-14-g1-isaac-sonic-eval-design.md`; plan:
`docs/superpowers/plans/2026-10-15-g1-isaac-sonic-eval.md`. Real-robot sessions:
[`2026-10-13-g1-eval-logger.md`](./2026-10-13-g1-eval-logger.md); MuJoCo backends:
[`2026-10-02-g1-wbc-backends.md`](./2026-10-02-g1-wbc-backends.md).

This repository is public: addresses, user names and credentials are left out on purpose. "Workstation" is the
2x RTX 3060 machine, "GPU host" the H100 server.

## Status in one paragraph

Isaac Lab 2.3.2 (Isaac Sim 5.1, CPU physics, RTX head camera) now replaces MuJoCo behind the SONIC deploy's DDS
interface and runs judged handover trials end to end, one full stack per trial, the policy server on a GPU host. It keeps
up with real time on the RTX 3060 when the shared machine is quiet (lag p99 10-61 ms per trial, most ~11 ms) and matches MuJoCo on two
held-out replays (gate GI). The scene and the task were fitted to the real robot sessions (black flight case, PET
bottle, black Dex3 hands; bottle position, handover zone and judge thresholds from 16-20 real successes), and the
policies act on its renders almost as on real frames (test 4b). At the case height estimated from one real head frame
(top 0.846 m), all 20 trials knocked the bottle over: the hand reached the bottle at cap height. With the case and bottle
raised 5 cm (0.896 m), **Pi0.5 he scored 8/10 and GR00T he 3/10**; every failure is a knock-over before a lift. The robot
baseline (GR00T he 6/11 on the same checkpoint) overlaps both intervals, so the sim does not yet rank the two policies
against the robot.

## Results

All Isaac numbers below were produced before the IMU frame fix (`325aae1f`: the base gyro and torso IMU velocities
were published in the world frame instead of gear_sonic's body / torso frames; the stacks start at yaw 1-3 deg and tilt
2-6 deg, so the effect is expected to be small). Re-run pending on a quiet machine.

### Closed-loop sessions (test 5)

10 trials each, seed 0, bottle jittered +-4 cm per trial; Wilson 95 % intervals.

| Case top | Policy (checkpoint) | Successes | 95 % CI | Failures |
| --- | --- | --- | --- | --- |
| 0.846 m (estimate) | GR00T he (`wbt_groot_he_full`) | 0/10 | 0.00-0.28 | 10 knocked over |
| 0.846 m (estimate) | Pi0.5 he (`wbt_pi05_he_full`, EMA) | 0/10 | 0.00-0.28 | 10 knocked over |
| 0.896 m (+5 cm) | GR00T he | 3/10 | 0.11-0.60 | 7 knocked over |
| 0.896 m (+5 cm) | Pi0.5 he | 8/10 | 0.49-0.94 | 2 knocked over |
| robot | GR00T he, sessions s05-s07 (model logged) | 6/11 | 0.28-0.79 | 4 grasp failed, 1 dropped |
| robot | GR00T he, + s03 (model not logged, 5/5) | 11/16 | 0.44-0.86 | |
| robot | Pi0.5 he, 41 episodes | runs 36-37: 1 success, 1 watchdog stop | | no session |

Successes lifted the bottle 9-14 cm (judge bar 3 cm). Per-trial lag p99 10-61 ms (most ~11 ms; the redo bar is 100 ms); policy chunk latency median 0.17 s
(GR00T) / 0.25 s (Pi0.5) over the tunnel. The spec's "Pi0.5 he 3/5" was the 20-episode Pi0.5 he, not this checkpoint.

**Why 0/20 at the estimated height.** Torso-frame comparison with one FK model on the logged joints: the policies bring
the right palm within 3-6 cm of the bottle axis sideways, as on the robot, but at z 0.14-0.19 m while the sim bottle
centre sat at 0.058-0.079 m, so the fingers met the cap. On the robot the palm comes in at about the same height and
drops to 0.09-0.12 m during the last second; the real bottle centre is at 0.10-0.12 m. Raising the case 5 cm put the sim
bottle centre at 0.117-0.128 m. The case top fitted on the real frame matches the torso at the planner stance; the 5 cm
offset is either a fit error (single frame, torso pitch assumed 6 deg) or a body-height difference during the reach.
Open: the case height measured on the real setup.

### Gates on the way

| Test | Result |
| --- | --- |
| 0a real-time factor, scene + camera, 60 s | GPU physics 0.49x (fails); CPU physics min 1.08 / median 1.37 (user: continue on CPU) |
| 0b paced lag, ep 27 replay with the deploy | max 18.4 ms, p99 10.7 ms, 0 of 12259 steps > 20 ms (quiet machine); scorer valid, palm err vs the recording p50 3.9 / p95 7.0 cm |
| GI Isaac vs MuJoCo replay parity, ep 27 / 35 x 3 | pass: palm p50 diff 0.17 / 0.93 cm (<= 1.0), p95 diff 0.43 / 1.30 cm (<= 2.0), tilt 9.3 vs 8.1 / 7.6 vs 7.3 deg |
| 3 task config | zone reached on replays (5.7-6.1 cm <= 8); replay reach check dropped (user): open-loop replays land 4-7 cm off |
| 4 reset stability | the planner keeps its world heading across a reset (pelvis twisted to the last heading, 15.9 deg): one stack per trial; fresh stacks settle at yaw 1.8-2.6 deg, xy <= 4 cm |
| 4b visual gap, 42 pre-grasp samples of s07, GR00T he | token gap render vs real 0.0117 (median), black frame 0.0305, real vs streamed 0.0086; render closer than black in 35/42; last 1.5 s before the grasp: render ~0.01, black 0.04-0.09 |

## Scene and task

Fitted to the real handover sessions (s03, s06, s07, s08), not to the training data room.

| Item | Value | Source |
| --- | --- | --- |
| Head camera | URDF d435 mount on `torso_link`, 640x480, nominal 615 px | FK-projected hands land on the real hand within ~15 px |
| Case (table) | black box 0.42 x 0.32 m, top 0.846 m (`CASE_TOP_Z`; 0.896 for the +5 cm runs) | s03 head frame, case top fitted rectangular at a 6 deg torso pitch |
| Table distance | `TABLE_GAP_CM` 20 (near edge 0.28 m ahead of the pelvis) | same fit |
| Bottle | r 3.3 cm, h 13.5 cm + 1.5 cm yellow-green cap, 0.5 kg; nominal (0.33, -0.065) m, jitter +-4 cm | 16 real right-hand grasps (MuJoCo FK of the logged state) |
| Handover zone | (0.359, -0.126, 0.143) m in the pelvis frame, r 0.10 m | the same trials' releases |
| Judge | lift >= 3 cm, palm within reach of the bottle and in the zone for 0.5 s | passes 18 of 20 real successes (10 cm / 8 cm / 1 s passed 0) |
| Look | grey floor without grid, dome light 1200, black Dex3 hand and wrist links | side-by-side renders |

Session JPEGs carry RGB bytes (the camera server encodes RGB as is): decoded as BGR they look channel-swapped.

## How to run a session

```bash
# workstation, Isaac env + lerobot; policy server on the GPU host (models under a dir the container can mount)
SIM=isaac TRIALS=10 SESSION_DIR=sessions/isaac_pi05_he POLICY_PATH=<model dir on the GPU host> \
  ROBOT_ARGS=<robot session>/raw/trial_000.npz GROUND_USD=<local Grid/default_environment.usd> \
  ISAAC_PY=<python of the Isaac Lab env> CODE_DIR=<script copy on the GPU host> SERVER_EXTRA="--noise-scale 0" \
  CASE_TOP_Z=0.896 bash examples/g1_dex3_training/sonic_official_sim_eval.sh isaac_pi05_he - - - <gpu> 20
```

GR00T adds `--backbone-dtype bfloat16` to `SERVER_EXTRA`. The launcher starts the policy server container, an ssh
tunnel bridged into each trial's private network namespace through a unix socket, warms the server, then runs one stack
(Isaac host + deploy + streamer) per trial with a fresh gate dir and redoes a trial whose sim lag p99 exceeds 100 ms.
A session resumes with the same `SESSION_DIR`. Score files: `results.jsonl` and `trial_*.npz` as on the robot; host
logs and `sim_state_*.npz` under `LOG_DIR/<name>`.

Streamer flags are the robot session's args (`task "pick drink bottle from the table and handover"`, `duration 30 s`,
`start/end planner`, `replan 0.4 s`, `policy fps 50`, `hold rest 1.5 s`, `chunk blend 0.3 s`, `max token step 0.05`,
`handoff blend 2.0 s`, `max chunk age 1.0 s`, `max state age 0.2 s`, `max arm speed 6.0`, `arm range margin 0.15`).
Differences from the robot: policy server `tcp://127.0.0.1:5560` (resolving localhost in the user namespace costs ~2 s),
`--policy-timeout-s 5` (robot 1.5: the first chunk of a fresh streamer through the bridge took 1.5-2.1 s, later
0.15-0.4 s), `--dex3-right-order swap` (the sim bridge's right-hand motor order), gate and session dirs.

Test 4b: `isaac_visual_gap.py samples <session> OUT` (lerobot env, `G1_SESSIONS` = session root), then
`isaac_render_states.py OUT` (Isaac env), then `isaac_visual_gap.py policy OUT <model>`.

## Known limits

- The case height is a single-frame estimate, and the successes need it 5 cm higher; measure the real case.
- No humans, room clutter, floor texture or case hardware in the scene; the bottle is an opaque light cylinder
  (OmniGlass renders black in real-time RTX).
- The bottle is a rigid 0.5 kg cylinder and the fingers rigid: a brush that a water-filled PET bottle survives on the
  robot can tip it in sim.
- Nominal camera intrinsics until the head camera is calibrated; table clearance from body origins; no third-person video.
- The 3060 keeps up only on a quiet machine: at a load average of ~9, most trials had to be redone (one took ~30 attempts).
- Sessions depend on the VPN to the GPU host; a drop ends the trial ("no valid chunk") and the trial is redone.

Footage, per-trial heights and the render pairs: a private Claude artifact (Isaac Handover Footage).
