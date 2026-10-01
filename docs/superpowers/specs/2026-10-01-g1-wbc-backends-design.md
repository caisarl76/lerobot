# G1 whole-body control backends for 28D / 31D joint policies (design)

Date: 2026-10-01. Status: approved in conversation, awaiting written-spec review.

## Goal

Run VLA policies that output joint targets — 28D (14 arm + 14 Dex3 hand, as in the original Humanoid Everyday
(HE) action) or 31D (28D + 3 waist) — on the Unitree G1 through two whole-body controllers, and compare them in the
same simulated scene:

- **A — SONIC** (existing): joints → SONIC tokens on the robot side (encoder mode 0) → NVIDIA C++ deploy.
- **C — GR00T decoupled WBC** (new): NVIDIA's official lower-body RL policy (legs + waist) from
  `NVlabs/GR00T-WholeBodyControl` `decoupled_wbc`, arm and hand targets straight to PD. This is the controller
  behind GR00T N1.5/N1.6 on G1.

Success: C reaches lower palm error than A at equal balance (no falls, tilt and foot contacts comparable), scored
by the unchanged `sonic_stream_eval.py` on the same held-out HE episodes and table scene.

## Background and decisions

- A's 28D path exists: `sonic_policy_streamer.py --action-space joint28` (`sonic_targets.joint_chunk_to_sonic()`),
  legs and waist fixed at `NOMINAL_BODY`.
- Palm error has two sources: SONIC's own tracking (round-trip audit: p50 1.4–1.9 cm, p95 3–5 cm; SONIC picks its
  own arm posture) and the policy's error (closed-loop median 6–9 cm). A controller change can only remove the first.
  SONIC's VR 3-point mode is worse (paper: 6 cm mean wrist error), so it is not considered.
- Official decoupled WBC accepts the waist through `instantiate_g1_robot_model(waist_location="lower_and_upper_body")`:
  forward kinematics of the commanded waist gives the torso roll/pitch/yaw passed to the lower-body policy
  (`G1DecoupledWholeBodyPolicy.get_action`). With `"lower_body"` the torso command is zero (28D case).
- HE's waist is static: over 60 sampled episodes the measured waist (`observation.leg_joints[12:15]`) has std
  0.004 / 0.003 / 0.017 rad (yaw/roll/pitch) and a per-episode range p90 ≤ 0.031 rad. So 31D is built and tested
  with a synthetic waist; no HE `joint31` dataset is built.
- Evaluation is sim-first in the existing scene; real-robot C comes later (out of scope).
- Order: (1) replay recorded actions through A and C (no model, isolates the controller); (2) train a 28D GR00T on
  HE and compare closed loop.
- Rejected alternatives: LeRobot `UnitreeG1` + `GrootLocomotionController` (reimplementation, no waist→torso step,
  different scene and gains); the official `run_g1_control_loop.py` as-is (own sim and DDS loop, hard to align with
  our table, startup and logs); Unitree built-in locomotion + `arm_sdk` and research WBCs (ULC, AMO, HOMIE, TWIST2)
  — possible later.

## Architecture

```
policy server (GPU, unchanged, returns 28D)  ── or ──  --replay (recorded action chunks)
                         │ chunk [N, 28|31]
                         ▼
sonic_policy_streamer.py (robot side, CPU)
  ├─ --backend sonic      joint → tokens (encoder; waist from action if 31D) → 50 Hz tokens → NVIDIA deploy → MuJoCo
  └─ --backend decoupled  same 30→50 Hz resampler → 29 body + 14 hand targets → decoupled_wbc_sim_host.py → MuJoCo
                                                                         │ same scene, table, sim_state.npz
                                                                         ▼
                                                           sonic_stream_eval.py (unchanged)
```

## Components

All under `examples/g1_dex3_training/`.

| File | Change |
| --- | --- |
| `sonic_targets.py` | `joint_chunk_to_sonic()` accepts `[N, 31]`; columns 28–30 replace `NOMINAL_BODY[12:15]` in the encoder's 29D body target. 28D unchanged. |
| `sonic_policy_streamer.py` | `--action-space joint31`; `--backend {sonic,decoupled}` (default `sonic`); `--replay`: serve the episode's recorded action chunks (same chunk length and replan times) instead of querying a policy; `--synthetic-waist`: append the generated waist track below to a 28D chunk. With `decoupled`, publish 50 Hz joint targets on the action port and skip the planner / POSE token handoff. |
| `decoupled_wbc_sim_host.py` (new, ~200 lines) | Same scene, table, gates, video and `sim_state.npz` as `sonic_official_sim_host.py`, reusing its table and recording helpers (imported or factored out, not copied). Physics 500 Hz; at 50 Hz runs the official `G1GearWbcPolicy` (Balance/Walk ONNX; walking command zero; height 0.74) for legs + waist; arms and hands get the streamer targets through PD with the official `g1_gear_wbc.yaml` gains; torso command from the robot model (`lower_and_upper_body` for 31D, zero for 28D). Publishes robot state in the NVIDIA deploy's ZMQ state format on the same port. |
| `decoupled_wbc_sim_eval.sh` (new) | Copy of `sonic_official_sim_eval.sh` that starts the new host. Official repo mounted at pinned commit `b042411fae` (2026-09-23); onnxruntime from a mounted `PYTHONPATH` folder (container Python 3.10 already has pinocchio 2.7, torch, mujoco 3.12). Nothing installed on the host. |
| `sonic_stream_eval.py` | For 31D runs, the palm FK target uses the commanded waist instead of `NOMINAL_BODY[12:15]`. |

31D action layout: `[arms 14 | hands 14 | waist 3]`, waist in motor order yaw, roll, pitch (indices 12–14). The
first 28 columns equal `joint28`, so 28D models, stats and tools are unaffected.

## Startup, handoff and safety (backend C)

1. Robot hangs on the elastic band at the official default pose; host releases it; Balance policy stands it up with
   arms at the gear-WBC default. After 5 s settled, host writes `GATE/deploy_ready`, then `GATE/settled`.
2. Table parked 10 m away, moved in after `settled` (as in A). HE episodes: `TABLE_GAP_CM=30`, matching A's HE
   runs (`--start planner`, table 25 cm further out).
3. Streamer blends arms (and waist for 31D) linearly in joint space from the measured pose to the episode's first
   action over 2 s, holds 1 s, then streams.
4. At episode end or `GATE/done`: blend back to the start pose over 2 s; host stops 3 s later.

Safety in sim: the streamer's state-age, max-chunk-age and `--chunk-blend-s` work unchanged. No joint-step cap in
sim (arm speed is logged); it is required before real-robot C. The host logs `fell` and stops if tilt > 20° or
both feet leave the ground, so a failed run is never scored as normal.

## 31D waist per backend

- A: the waist columns go into the encoder's body target; SONIC tracks them with its own posture choice.
- C: the commanded waist → torso orientation → `torso_orientation_rpy` of the lower-body policy, which moves waist
  and legs to follow it (not direct PD; NVIDIA's design).

Synthetic waist track (on a real HE episode's 28D actions), one axis at a time: yaw ±0.4 rad sine at 0.2 Hz, then
roll ±0.15 rad sine at 0.2 Hz, then pitch 0 → 0.3 rad → 0 (each ramp 2 s, hold 2 s). Joint limits: yaw ±2.6,
roll/pitch ±0.52 rad.

Reported per backend: torso orientation reached vs commanded (median / p95 per axis), palm error vs FK including the
commanded waist, tilt, foot contacts, fall. Pass: no fall and palm p95 at most 2 cm worse than the same episode
replayed as 28D. Torso error is reported without a threshold (nothing to base one on yet).

## Evaluation plan

Step 1 — replay (no model). Six held-out HE episodes (ho5 split), chosen with a fixed seed across tasks, including
1293 and 1300. Per episode:

| Run | Source | Backend |
| --- | --- | --- |
| A-stored | stored `sonic78_nolimit` tokens | sonic |
| A-28D | recorded 28D actions, `--replay` | sonic |
| C-28D | recorded 28D actions, `--replay` | decoupled |

Plus the synthetic-waist run (A and C) on two of the episodes. Gate: A-28D must match A-stored within run-to-run
noise before any A-vs-C conclusion. Metrics: palm p50/p95/max, wrist orientation, max tilt, min foot contacts,
table contacts, arm speed p95, palm jerk p95.

Step 2 — closed loop. Train GR00T (official recipe) on HE `joint28` (GR00T normalizes with mean/std, so the stale
`joint28` quantiles do not matter). Ask the user for the H100 queue slot (GPUs 0/6, shared with the retraining)
before queueing. Then A-28D vs C-28D closed loop on the same six episodes, with the existing HE 78D GR00T
`*_official_full` (A-native) as the baseline.

## Tests

1. `assert`: `joint_chunk_to_sonic` on 31D with waist = `NOMINAL_BODY[12:15]` gives tokens identical to the 28D input.
2. `assert`: `--replay` returns the recorded chunk for each replan time.
3. Host smoke: backend C stands 30 s with constant arm targets; no fall, tilt < 3°.
4. Existing `sonic_joint28_stream_check.py` still passes.

## Out of scope

Real-robot backend C (and its joint-step cap); walking and base-height commands; an HE `joint31` dataset; changes to
the policy server; Unitree built-in locomotion and research WBCs.

## Risks

- Official `decoupled_wbc` imports may pull heavy dependencies (robocasa); if so, import only the lower-body policy
  and robot model modules.
- NVIDIA's MuJoCo bridge swaps Dex3 right-hand index/middle; reuse `--dex3-right-order swap` in the new host.
- Lower-body policy ONNX files are git-lfs objects in the official repo
  (`decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-{Balance,Walk}.onnx`); fetch them
  explicitly at the pinned commit.

## Deliverables

Code above, results in `docs/research/<run-date>-g1-wbc-backends.md`, and a pointer in `CLAUDE.md`.
