# G1 Isaac handover: the sim-vs-real gap at the measured geometry (handover, 2026-10-20)

Continues [`2026-10-17-g1-isaac-handover-eval.md`](./2026-10-17-g1-isaac-handover-eval.md) (harness, how to run a
session, gates). This page is the state of the investigation into why both handover policies fail in Isaac at the real
case height while they succeed on the robot, and what to try next. Public repository: addresses, user names and
credentials are left out. "Workstation" is the 2x RTX 3060 machine, "GPU host" the H100 server, `<eval data>` the
workstation's evaluation data dir (sessions, logs, the analysis scripts named below).

## Status in one paragraph

With the real case and bottle measured (case 0.475 x 0.35 m, top 0.845 m; bottle 17 cm with the cap), both policies
fail every trial in Isaac by knocking the bottle over. On the robot GR00T he grasps 6/11 (Pi0.5 he 3/5 by operator
score), so the sim must be fixed before it can rank policies. Ruled out: pelvis height, body posture, head-camera pose,
policy latency, execution lag, sampling noise, the arm-joint input and the bottle's look. Found so far: (1) a
full-radius cylinder catches the hand where a real PET bottle narrows (tapered bottle committed); (2) the rendered
scene delays the policy: with renders GR00T's commands trail the robot's by ~0.3 s, with the real room composited
behind the live sim robot (`COMPOSITE_BG`) they match within 0.06 s and the hand follows the real hand path.
**Update: the sim reproduces the real grasp; placement decides the outcome (GR00T 4/9 so far with randomized placements, robot 6/11)** - see "Push force and bottle placement" and "Placement-randomized evaluation". Before that: - see "Push force and bottle placement". Earlier note: Even on that real path the sim hand meets the bottle's neck or cap 1.8-2.3 s into the
episode and the bottle slides 8-9 cm and tips, while in the real frames the hand also touches the bottle (thumb at the
cap, fingers at the base) and it stays upright until grasped. Next: bottle/case friction, the bottle's mass
distribution, and the sim arm's stiffness against contact.

## Results at the measured geometry

GR00T he (`wbt_groot_he_full`), Pi0.5 he (`wbt_pi05_he_full`, EMA), server `--noise-scale 0` (GR00T also
`--backbone-dtype bfloat16`), case top 0.845 m, gap 20 cm. Palm height is in the torso frame (MuJoCo FK of the
logged joints) at the closest horizontal approach to the bottle; the bottle centre sits at 0.07-0.09 m.

| Session | Change | GR00T | Pi0.5 | Palm height (m) |
| --- | --- | --- | --- | --- |
| `meas_*` | measured case + 17 cm cylinder bottle | 0/9 | 0/10 | 0.15-0.19 / 0.14-0.19 |
| `off_*` | + arm-state offset (policy input only) | 0/10 | 0/10 | 0.15-0.20 / 0.17-0.20 |
| `rf_*` | real head frames of s03 t0 / s07 t0, cylinder bottle | 0/6 | 0/6 | 0.15-0.17 / 0.17-0.20 |
| `taper_*` | tapered bottle (body, shoulder cone, neck, solid cap), renders | 0/10 | 1/10 | 0.15-0.18 / 0.14-0.17 |
| `rfoff_groot_s03e0` | real frames + tapered bottle + arm offset | 2/3 (lift 13 cm) | - | 0.17 |
| `rftaper_groot_s03e0` | real frames + tapered bottle | 2/3 | - | - |
| `taper_groot_glass` | tapered glass bottle with a water core, renders | 0/10 | - | 0.16-0.18 |
| `comp_*_s03` | composite: real s03 background + live sim robot and bottle (15 fps) | 0/5 | 0/5 | 0.17 / 0.12-0.17 |

Every failure is "collision (table, knocked over)". Real-frame runs pin the bottle at the trial's real spot
(`BOTTLE=0.332,-0.090` for s03 t0); the policy is deterministic, so trials of one run are near repeats.

## What is ruled out, and how

- **Pelvis height / posture.** The sim pelvis stands at 0.785-0.787 m (real recordings 0.785-0.792). Replaying 9
  dataset episodes (`handover_bottle_sonic78_50hz_261002`) through SONIC in Isaac puts the palm centre at
  0.922-0.945 m at the grasp, the real grasp height (0.92-0.95 m). Script: `<eval data>/analysis/replay_grasp.py`.
  Note: torso-frame palm agreement in replays is nearly automatic (arm joints are copied); the world height is the
  evidence.
- **Head-camera pose.** Case corners in the first-tick frame match real within 1-25 px (rows 4-17 px, under ~1.6 deg).
  A real frame and a sim frame with the bottle at the same spot blend almost exactly
  (`<eval data>/analysis/camcmp_same_bottle.jpg`). Back-projecting the cap through the sim camera recovers the sim
  bottle within 1 cm; real bottles stood at x 0.30-0.42, y -0.12..+0.11 m (sim task: 0.33, -0.065 +- 0.04).
  Scripts: `proj_cap.py`, `capcmp.py`. Caveat: the case distance was originally fitted from one real frame with this
  camera model; depth and width are measured, and they still line up.
- **Arm-joint input.** At the planner stance the sim arm differs from the real one mostly at the right wrist
  (roll real 13.9 vs sim -2.8 deg, pitch -0.5 vs 7.8, yaw -14.2 vs -6.5; real spread 6 deg: friction or backlash the
  sim does not model). Adding the real-minus-sim offset to the policy's state input (`--state-offset`, offsets in
  `<eval data>/analysis/arm_offset.txt`) changed nothing in `off_*`. Hand joints match within 2.5 deg.
- **Policy latency.** Sim chunks take 0.16-0.18 s, robot sessions 0.20-0.22 s.
- **Execution.** Fed the same recorded tokens, the sim arm tracks the recorded real arm with no lag and ~2 deg rms
  (9 replays).
- **Sampling noise.** The robot sessions also used noise scale 0, and test 4b reproduced the robot's tokens from real
  inputs within 0.009.

## What is established

- **Bottle shape.** In every knock-over the hand part nearest the bottle when it starts moving is 12-19 cm above its
  base: the top of the body or the cap, mostly the wrist link. The real hand on s03 t0 stays ~6 cm from the bottle axis
  while high, then descends beside it; a full-radius 3.3 cm cylinder there is hit. With real frames, the tapered bottle
  takes GR00T from 0/3 to 2/3 (twice). Committed: `BOTTLE_PARTS` in `isaac_scene.py` (b1dfafe1). The neck radius
  (1.35 cm) and shoulder height (11-14 cm) are typical 500 ml values, not measured.
- **Command timing.** With real frames, GR00T's sim tokens equal the robot's tokens 0.30-0.38 s later (0.22 s with the
  arm offset). The export aligns frames to the episode clock (latest JPEG at `wall_start + k/30`), latency and
  execution are ruled out, so the delay comes from the state input; it remains in the successful runs.
- **Bottle material is not the gap.** Opaque grey (body brightness 190) and glass + water core (134; real 144) both
  give GR00T 0/10 at the same palm heights (37eae445). Partial opacity disappears in the real-time renderer; plain
  glass renders dark (it shows the black case).
- **The +5 cm case result** (Pi0.5 8/10, GR00T 4/10) only moved the bottle top relative to the hand path; it does not
  rank the policies.

## Composite-image test (2026-10-20)

`COMPOSITE_BG=<real frame .npy>` publishes the rendered robot and bottle (instance segmentation) over a real head frame;
`HEAD_CAMERA_FPS=15` keeps the extra render pass in real time (at 30 fps lag p99 reached 0.4-1.3 s). Background: s03
trial 0's first-tick frame with its bottle patched from trial 2 (ECC-registered, gain-matched, feathered;
`<eval data>/analysis/composite_bg.py`). Bottle pinned at s03 t0's spot. Results, GR00T 0/5 and Pi0.5 0/5, all
knock-overs, but:

- GR00T's tokens match the robot's (s03 t0) within 0.02-0.06 s (renders: 0.30-0.38 s late) and its palm follows the
  real path: at 2.5 s 0.12-0.16 m above the torso (real 0.145), at the end 0.115-0.125 m, 15.5 cm right (real 0.128 m,
  13.5 cm). The rendered scene's look caused the lag.
- The sim bottle is drawn within 1-14 px of the real bottle's place in the real frame (one Pi0.5 trial 35 px: the
  standing heading varies per trial). The real hand projected through the sim camera (logged arm and hand joints, the
  sim's SONIC stance) lands on the hand in the real frames within ~10-20 px (`hand_reproj.py`): no gross mount error.
- The judge is right: the bottle starts moving under the hand at 1.8-2.3 s, slides 8-9 cm and passes 60 deg tilt
  0.6-2.0 s later. The nearest hand parts at first contact are the wrist link and thumb or finger bases, 13-17 cm above
  the bottle's base (neck and cap).

## Contact experiments (2026-10-20)

Composite view, bottle pinned at s03 t0, GR00T, 3 trials per variant (`BOTTLE_FRICTION`, `CASE_FRICTION`,
`BOTTLE_COM_Z`, 1ba842fe; `<eval data>/analysis/run_contact.sh`):

| Variant | Success | First contact (all at 1.9-2.5 s) |
| --- | --- | --- |
| default (PhysX friction 0.5, uniform density: COM ~7 cm above the base) | 0/5 | tips over in 3/5 within 1 s |
| bottle + case friction 0.3 | 1/3 | the success slid 5.9 cm upright, then was grasped |
| bottle friction 0.15 | 1/3 | the success slid 5.0 cm upright |
| COM 4 cm above the base | 2/3 | **survived first contact 3/3** (tilt <= 12 deg, slide <= 2.6 cm); the failure tipped later, at the grasp |
| bottle + case friction 0.8 | 1/3 | the success barely moved (0.3 cm) |

The hand always pushes the bottle at ~2 s; the outcome turns on whether it slides or stays upright vs tips. Friction is
not monotonic (0.8 also 1/3), so with 3 trials it is noise-level; a low centre of mass is the one change that made the
first contact survivable every time. Caveat: 4 cm is lower than a full 500 ml bottle (~6.5 cm with water to the
shoulder); it would fit a bottle filled about halfway. In the real frames the hand also touches the bottle and it
barely moves, so the sim hand may push harder than the real one.

## Push force and bottle placement (2026-10-20)

`touch_force` (contact sensor on the bottle, per wrist/Dex3 link) and `cmd_kp`/`cmd_kd` are logged (b820272e);
`<eval data>/analysis/pushforce.py` summarizes a session.

- **The push.** In every composite trial the right wrist link (the Dex3 palm is merged into it) meets the bottle top at
  ~1.7-2.0 s and presses 5-9 N mean, 8-21 N peak, for >= 0.5 s; ~1.1 N at that height starts tipping the bottle. The
  palm moves at 0.09-0.17 m/s at the touch (real s03 t0 at the same time: 0.11-0.17 m/s); arm kp 14.3-16.8 Nm/rad,
  kd 0.91-1.07 (the deploy's); tracking error up to 15 deg: the arm's target lies past the bottle.
- **Collision approximation is not it.** `URDF_COLLIDER=convex_decomposition` (cce2ee82; the convex hull makes the
  wrist_yaw mesh 10.5x and the palm 1.5x their volume): 1/5, same touch time and force.
- **The real commands replayed** (s03 t0's logged tokens, open loop, `run_replay.sh` with `DS=<session>/dataset`, own
  network namespace: another process holds port 5556): the sim palm follows the real palm within ~1 cm to 2.0 s, then
  the sim wrist hits the bottle at 2.04 s and the bottle tips, where the real hand went down beside it.
- **Grid over the bottle's position** (`run_replay_grid.sh`, same real commands): lifted 12-13 cm at 7 of 9 positions
  on the case (x 0.312 and 0.352, y -0.09..-0.05); knocked over only at x 0.332, y -0.09/-0.07, the image estimate
  of s03 t0's bottle (x 0.292 hangs off the case edge). The failure is a ~2 cm band where the wrist path crosses the
  cap. Every composite run so far pinned the bottle inside it.
- **Closed loop at working spots** (composite, GR00T): (0.352, -0.070) **3/3**; (0.312, -0.090) 0/3 (the policy shifts
  its path with the bottle and meets the top again).

So the sim reproduces the real grasp; the outcome turns on centimetre-level placement relative to the hand path, as
it plausibly does on the robot (GR00T 6/11). Evaluation needs random placements over the real range, not one pinned
spot.

## Placement-randomized evaluation (in progress, started 2026-10-20 23:51)

Composite view (`COMPOSITE_BG`, `HEAD_CAMERA_FPS=15`), measured case, tapered glass bottle, default friction and mass,
server `--noise-scale 0`. The bottle is drawn per trial, uniform over the real trials' placement range clamped to stand
on the case: `BOTTLE=0.3675,-0.005 BOTTLE_JITTER=0.0525,0.115` (x 0.315-0.42, y -0.12..+0.11; 5271c24b). The draw is
seeded by the trial, so GR00T and Pi0.5 meet the same 20 placements (x 0.324-0.415, y -0.083..+0.107). Run:
`<eval data>/analysis/run_random.sh`; summary: `random_summary.py rnd_groot rnd_pi05` (per trial, Wilson intervals).

Interim, GR00T he after 9 of 20 trials: **4/9** (44%, 95% CI 19-73%); robot: 6/11 (55%).

| Trial | Bottle (x, y) m | GR00T |
| --- | --- | --- |
| 1 | 0.382, -0.058 | knocked over |
| 2 | 0.369, +0.099 | dropped after the lift |
| 3 | 0.342, -0.051 | knocked over |
| 4 | 0.324, -0.066 | success |
| 5 | 0.414, -0.002 | knocked over |
| 6 | 0.400, +0.066 | success |
| 7 | 0.372, -0.041 | knocked over |
| 8 | 0.381, +0.086 | success |
| 9 | 0.349, +0.107 | success |

So far the knock-overs cluster on the robot's right half (y -0.06..0.00), where the right hand's path crosses the
bottle top first (trial 4 at y -0.066 is the exception); bottles further left (y > +0.06) were grasped 3 of 4 times. Pi0.5 follows on the same placements; the
final rates, the paired comparison and the ranking against the robot go here when both sessions finish.

## Next experiments, in order

1. **Placement-randomized evaluation (composite view).** Sample bottle positions over the real range (x 0.30-0.42,
   y -0.12..+0.11 m, from the back-projected real placements; keep x >= 0.31 so the bottle stands on the case), 20+
   trials per policy, GR00T and Pi0.5, and compare with the robot. Also stochastic sampling (noise scale 1) for a
   distribution rather than one deterministic path per spot.
2. **Then the rendered scene.** Once composite runs grasp, go back to renders and close the look gap that delays the
   policy (case texture, floor and background, the person's hands), or keep the composite as the evaluation view.
3. **Measure the real bottle** (neck radius, shoulder start, cap size: the real cap looks larger in the frames) and
   update `BOTTLE_PARTS`.
4. **Ranking.** Once a policy succeeds in sim at the real geometry, rerun both policies (10+ trials) and compare with
   the robot (GR00T better there).

## How to run the variants

Session launch is as in the 2026-10-17 page (`sonic_official_sim_eval.sh <name> - - - <gpu> 20`). Extra arguments
after the six positionals go to the streamer; environment variables reach the sim host:

```bash
# arm-state offset (policy input only; the "=" matters, the list starts with a minus sign)
... sonic_official_sim_eval.sh off_groot - - - <gpu> 20 --state-offset=$(cat arm_offset.txt)
# real head frames of a recorded robot trial (closed loop on the sim state), bottle pinned at its real spot
BOTTLE=0.332,-0.090 ... sonic_official_sim_eval.sh rf_groot_s03e0 - - - <gpu> 20 \
  --images dataset --dataset-root <robot session>/dataset --episode 0
```

`--images dataset` with `--trials` is accepted only with a sim host's gate dir, so a robot never acts on recorded
frames. Metrics: `<eval data>/analysis/gen_metrics.py <session>...` (lerobot env with mujoco) writes `trials.json`
next to the script with per-trial palm/bottle heights, closest approach and lift. The wrapper scripts there
(`run_session.sh`, `run_taper.sh`, `run_realframe.sh`, `run_replays.sh`) wait for a quiet machine first.

## Pitfalls met in this round

- **Shared workstation.** Another session's MuJoCo demo or a dataset conversion pushes load past ~4; Isaac then lags
  and trials are redone (5 attempts, then "not saved"). The wrappers wait until load < 4 for 2 minutes.
- **VPN drops.** An ssh drop while the launcher waits for the policy server leaves the container running and the run
  "policy server failed"; remove the policy container on the GPU host (its name is the launcher's `P=`) and rerun.
- **GPU host GPUs.** A training job took most of GPU 3 on 10-19; sessions moved to GPU 2.
- **Not ours.** An ssh tunnel with `BatchMode=yes ... -L 127.0.0.1:34002` belongs to the dataset visualizer's
  annotation tunnel script; leave it.
- **pgrep self-match.** A wait loop written inline in `bash -c` matches its own command line; put it in a script and
  use a `[d]emo`-style pattern.
- `gen_metrics.py` overwrites `trials.json` beside it: copy it first if a page depends on it.
- The other session's processes now run from a `dimos_dev` checkout: the wrappers wait on any `[d]imos` process.
- In the scene, `UsdGeom.Imageable(...).MakeInvisible()` on the typeless `.../geometry` prim does nothing; target the
  `.../geometry/mesh` prim.
