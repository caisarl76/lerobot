# G1 whole-body control backends for 28D / 31D joint policies (design)

Date: 2026-10-01. Revision 3 (review rounds 1 and 2 addressed). Status: approved.

> Implemented in parts: this PR builds the 28D path (evaluation step 1 and the step 3 tooling). The 31D waist
> parts below (`--action-space joint31`, `--synthetic-waist`, waist location `lower_and_upper_body`, the `syn` gates)
> are implemented in the follow-up 31D PR.

## Goal

Run VLA policies that output joint targets — 28D (14 arm + 14 Dex3 hand, as in the original Humanoid Everyday
(HE) action) or 31D (28D + 3 waist) — on the Unitree G1 through two whole-body controllers, and compare them in the
same simulated scene:

- **A — SONIC** (existing): joints → SONIC tokens on the robot side (encoder mode 0) → NVIDIA C++ deploy.
- **C — GR00T decoupled WBC** (new): NVIDIA's lower-body RL policy (legs + waist) from `NVlabs/GR00T-WholeBodyControl`
  `decoupled_wbc` at pinned commit `b042411fae` (2026-09-23); arm and hand targets straight to PD. This is the
  controller behind GR00T N1.5/N1.6 on G1.

Success is decided by the numerical gates in "Comparison gates" below, on held-out HE episodes in the same table
scene, scored by one scorer that applies the same validity rules to A and C.

## Background and decisions

- A's 28D path exists: `sonic_policy_streamer.py --action-space joint28` (`sonic_targets.joint_chunk_to_sonic()`),
  legs and waist fixed at `NOMINAL_BODY`.
- Palm error has two sources: SONIC's own tracking (round-trip audit: p50 1.4–1.9 cm, p95 3–5 cm; SONIC picks its
  own arm posture) and the policy's error (closed-loop median 6–9 cm). A controller change can only remove the first.
  SONIC's VR 3-point mode is worse (paper: 6 cm mean wrist error) and is not considered.
- HE's waist is static: over 60 sampled episodes the measured waist (`observation.leg_joints[12:15]`) has std
  0.004 / 0.003 / 0.017 rad (yaw/roll/pitch), per-episode range p90 ≤ 0.031 rad. 31D is built and tested with a
  synthetic waist; no HE `joint31` dataset is built.
- Sim first in the existing scene; real-robot C later (out of scope).
- Order: (1) 28D replay through A and C with valid scoring; (2) synthetic-waist 31D replay; (3) train a 28D GR00T on
  HE and compare closed loop.
- Rejected: LeRobot `UnitreeG1` + `GrootLocomotionController` (reimplementation, no waist→torso step, different scene
  and gains); upstream `run_g1_control_loop.py` as-is (own sim and DDS loop); Unitree built-in locomotion + `arm_sdk`
  and research WBCs (ULC, AMO, HOMIE, TWIST2) — possible later.

## Architecture

```
policy server (GPU, unchanged, returns 28D)  ── or ──  --replay (recorded action chunks)
                         │ chunk [N, 28|31]
                         ▼
sonic_policy_streamer.py (robot side, CPU), existing 30→50 Hz resampler
  ├─ --backend sonic      joint → tokens (encoder; waist from action if 31D) → token msg → NVIDIA deploy ─┐
  └─ --backend decoupled  joint msg (below) ──────────────────────────────────→ decoupled_wbc_sim_host.py ─┤
                                                                                                         ▼
                                        sim_scene.py (shared MuJoCo scene, table, recording, termination)
                                                                                                         ▼
                                                     sonic_stream_eval.py (backend-aware, validity-gated)
launcher: sonic_official_sim_eval.sh with BACKEND=sonic|decoupled (one script)
```

## Components

All under `examples/g1_dex3_training/`.

| File | Change |
| --- | --- |
| `sim_scene.py` (new, extracted) | Scene setup, table park/move-in and clearance, per-tick state recording, video, gate helpers and the termination record, moved out of `sonic_official_sim_host.py` into functions with no import-time side effects. Today that module runs setup and launches the deploy at import, so its helpers cannot be imported; the extraction is a required first step. |
| `sonic_official_sim_host.py` | Uses `sim_scene.py`; behaviour unchanged (regression check in Tests). Gains the termination record and fall detection. |
| `decoupled_wbc_sim_host.py` (new) | Backend C host; integration contract below. |
| `sonic_targets.py` | `joint_chunk_to_sonic()` accepts `[N, 31]`; columns 28–30 replace `NOMINAL_BODY[12:15]` in the encoder's body target. 28D unchanged. |
| `sonic_policy_streamer.py` | `--action-space joint31`; `--backend {sonic,decoupled}` (default `sonic`); `--replay` serves the episode's recorded action chunks at the same chunk length and replan times; `--synthetic-waist` appends the waist track below to 28D chunks. With `decoupled`: sends the joint message, skips the planner / POSE token handoff and `TOKEN_BOUND` (which only applies to tokens), and uses the startup in "Startup, handoff and safety". |
| `sonic_official_sim_eval.sh` | Parameterized, not copied: `BACKEND=sonic` (default, current behaviour) or `decoupled` selects the host script, its mounts (pinned upstream checkout, onnxruntime `PYTHONPATH` folder) and the streamer flags. Containers, gates, cleanup, datasets and scoring stay shared. |
| `sonic_stream_eval.py` | Backend-aware and validity-gated; see "Scorer". |

31D action layout: `[arms 14 | hands 14 | waist 3]`, waist in motor order yaw, roll, pitch (motor indices 12–14).
The first 28 columns equal `joint28`.

## Backend C integration contract (upstream commit `b042411fae`)

| Item | Value |
| --- | --- |
| WBC top-level config | `decoupled_wbc/control/main/teleop/configs/g1_29dof_gear_wbc.yaml` |
| Lower-body policy config | its `GEAR_WBC_CONFIG` = `decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.yaml`: `num_obs` 516 (86 × 6 history), `num_actions` 15, `rpy_cmd`, `height_cmd` 0.74, `cmd_scale` [2, 2, 0.5], `freq_cmd` 0.75. The other `teleop/configs/g1_gear_wbc.yaml` (570 obs, 29 gains, no `rpy_cmd`) is not referenced by the pinned code and is not used. |
| Policy ONNX | `model_path` = `policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx` under `decoupled_wbc/sim2mujoco/resources/robots/g1/` (git-lfs; fetched explicitly, SHA-256 recorded in the run log). Walking command is always zero, so Balance runs throughout. |
| Body PD gains (29) | `MOTOR_KP` / `MOTOR_KD` from `g1_29dof_gear_wbc.yaml` (legs 150/150/150/200/40/40, waist 250 ×3, arms 100/100/40/40/20/20/20; KD 2/2/2/4/2/2, 5 ×3, 5/5/2/2/2/2/2). Sim, so the `env_type == "real"` waist-pitch KD change does not apply. |
| Hand PD gains (14) | Neither upstream yaml has them. A's applied gains come from the deploy's hand commands, not from `SimLoopConfig`: `Dex3Hands` initializes every motor to kp 1.5 / kd 0.1 (`dex3_hands.hpp:396-397`), `open()`/`close()` use the same defaults, and the per-tick `setAllJointsCommand` (`g1_deploy_onnx_ref.cpp:4108`) leaves them unchanged. The sim applies torque = kp·(q* − q) + kd·(0 − dq) from the received command (`base_sim.compute_hand_torques`). Checked in A's container (sim image `sonic-vla-sim:startupfix-20260907`) source. C applies the same 1.5 / 0.1 with the same formula, so the hands are identical in both backends; test 8 confirms the gains A applies at run time. |
| Robot model | `instantiate_g1_robot_model(waist_location=...)`: `"lower_and_upper_body"` for 31D (the commanded waist is in the upper-body target and its FK gives `torso_orientation_rpy`), `"lower_body"` for 28D (torso command 0). |
| Upper-body policy | `IdentityPolicy`: the streamer already resamples to 50 Hz, so both backends get the same targets. (Upstream's default `InterpolationPolicy` would add its own joint-speed cap.) |
| Activation sequence | `IdentityPolicy.get_action()` returns its last goal unchanged, and the whole-body wrapper's `set_goal()` replaces the upper-body goal with only the keys it is given, so every goal sent through the wrapper must be complete. (1) Before the first `get_action()`, send through the wrapper a complete goal: `target_upper_body_pose` = measured upper-body pose, `base_height_command=[0.74]`, `navigate_cmd=[0, 0, 0]`. This sets `use_teleop_policy_cmd=True` (default False = torso commands ignored). (2) After release from the band, activate the RL output by calling the lower-body policy directly, once: `wbc.lower_body_policy.set_goal({"toggle_policy_action": True})` (`use_policy_action` defaults to False = lower body holds measured q). Never send a toggle-only goal through the wrapper. (3) Every per-tick goal through the wrapper carries all three keys: `target_upper_body_pose` (from the joint message), height and walking command. (4) The host asserts both flags are True before writing `GATE/settled`. |
| Rates | Physics 500 Hz (`SIMULATE_DT` 0.002), control 50 Hz. |
| Joint mapping | Body targets by motor index 0–28 → MuJoCo joint names (the `BODY` list in the host); hands by explicit name table from the dataset hand order. The SONIC bridge's right-hand index↔middle swap is not assumed for C; the round-trip test decides it. |

## Joint message (streamer → C host)

One ZMQ PUB message per 50 Hz tick on the action port, msgpack:

| Field | Type | Meaning |
| --- | --- | --- |
| `t_wall` | f64 | streamer wall-clock send time |
| `seq` | u64 | strictly increasing |
| `phase` | str | `leadin`, `hold`, `episode`, `return` |
| `frame` | i64 | source dataset frame (−1 outside `episode`) |
| `q_body` | f32[29] | motor order 0–28 (legs ignored by C; waist used only for 31D) |
| `q_hand` | f32[14] | dataset hand order: left thumb0–2, middle0–1, index0–1; right thumb0–2, index0–1, middle0–1 |

Limits: no token-style magnitude bound. The host rejects a message with non-finite values (holds the previous
target and logs it); finite values are clipped to the MuJoCo joint ranges and every clip is counted per joint in the
log. A message older than 0.1 s on arrival is dropped and counted. State goes back in the NVIDIA deploy's ZMQ state
format on the state port (read from the deploy source before writing the host), so `--state-source robot` and the
state-age check are unchanged.

## Startup, handoff and safety (backend C)

1. Robot hangs on the elastic band at the official default pose; host releases it, activates the policy (contract
   above). After 5 s settled with both flags True, host writes `GATE/deploy_ready`, then `GATE/settled`.
2. Table parked 10 m away, moved in after `settled` (as in A). HE episodes: `TABLE_GAP_CM=30`, matching A's HE runs
   (`--start planner`, table 25 cm further out).
3. Streamer blends arms (and waist for 31D) linearly in joint space from the measured pose to the episode's first
   action over 2 s (`leadin`), holds 1 s (`hold`), then streams (`episode`).
4. End of episode or `GATE/done`: blend back over 2 s (`return`); host stops 3 s later.

Safety in sim: streamer state-age, max-chunk-age and `--chunk-blend-s` unchanged. No joint-step cap in sim (arm
speed is logged); required before real-robot C.

## Termination record and fall detection (both backends)

Both hosts write `termination.json`: `{"reason": "completed" | "fell" | "timeout" | "error", "t_wall", "phase",
"detail"}`. Fall detection is armed only after `GATE/settled` (the robot hangs before that): tilt > 20° or both feet
off the ground for > 0.2 s ⇒ `fell`, stop. `completed` requires the streamer's `GATE/done` after the `episode`
phase ended normally.

## Scorer (`sonic_stream_eval.py`)

1. **Validity first.** Output `{"valid": false, "reason": ...}` and exit non-zero unless `termination.json` says
   `completed` and the scored source frames cover ≥ 98% of the episode's frames, with no streamer timestamp past the
   last sim sample (today it clips to the last sample). Same rule for A and C. Invalid runs never enter a comparison.
2. **Backend-aware.** Token and stored-token metrics only when the log has `token` (backend sonic). Hands vs dataset
   for both.
3. **Palm error two ways.** Against FK of the **joint reference** (controller error; see "Joint reference log") and
   against FK of the **recorded action** (policy + controller). The two can differ even in replay: resampling,
   chunk blending, lead-in, clipping and dropped messages separate them, so both are always reported. For C, a third
   value against the **applied** target (after clipping and drop-hold) isolates the PD/RL tracking. For 31D, FK
   targets include the commanded waist. **A-native runs (stored tokens, 78D policies) have no joint reference and get
   recorded-action metrics only.**
4. **Added metrics:** torso orientation reached vs commanded (31D; per axis median/p95), table contacts during
   `episode` (reuse `sonic_stream_phases.py` clearance code), arm joint speed p95, arm jerk p95 and palm jerk p95
   (the windowed third difference from `robot_run_smoothness.py`, applied to arm joints and to the logged palm positions), clip and dropped-message counts (C).
5. **Waist reaches the policy (31D, C).** The C host logs the lower-body observation's `rpy_cmd` slice
   (`single_obs[4:7]`) every tick; the scorer reports its error vs FK of the commanded waist and marks the run
   invalid if `rpy_cmd` stays at 0 while the commanded waist moves.

## Joint reference log

- **Streamer, both backends** (`--action-space joint28|joint31`): each 50 Hz tick logs `joint_ref` f32[31]
  (arms 14 | hands 14 | waist 3; waist = `NOMINAL_BODY[12:15]` for 28D) with the tick's `wall` time and `phase`. It
  is the target for that tick after resampling, `--chunk-blend-s` and lead-in/return blending — the value the
  backend is asked to reach. For A it is taken before encoding (the encoder's look-ahead uses the following ticks;
  the reference for time t is still the pose for t). For C it is the joint message's content.
- **C host** logs, per 50 Hz control tick, `applied_ref` f32[31] (after clipping and drop-hold), a per-joint
  `clipped` mask, a `held` flag (non-finite or stale message), and the received `seq`.
- **Alignment:** the scorer matches `joint_ref` (and `applied_ref`) to sim state by wall time, as it already does for
  tokens; ticks outside `episode` are not scored.
- **A-stored / A-native:** no `joint_ref` exists (no joint chunk); only recorded-action metrics are computed.

## Comparison gates

Repeats: every configuration × episode runs 3 times (timing over ZMQ makes runs non-identical). Statistics are the
per-episode mean over valid repeats; "spread" is max − min over the repeats.

- **G0 — online A is faithful** (before any A-vs-C claim): for each episode, on the recorded-action palm error (the
  only metric A-stored has), |A-28D − A-stored| ≤ max(0.5 cm, spread of A-stored) on p50 and ≤ max(1.0 cm, spread)
  on p95.
- **G1 — equal balance:** all repeats valid (no `fell`); max tilt ≤ A's max tilt on that episode + 2°; table contacts
  during `episode` ≤ A's.
- **G2 — C better at tracking:** mean palm p95 over the 6 episodes (vs `joint_ref`) lower than A-28D by ≥ 1.0 cm,
  and lower on ≥ 4 of 6 episodes. Wrist orientation p95 reported alongside, without a threshold.
- **31D synthetic (per backend):** valid, no fall, palm p95 ≤ the same episode's 28D replay + 2 cm, and (C) the
  waist-reaches-policy check passes. Torso error reported, no threshold yet.
- **Closed loop (step 3):** G1 applies; palm error reported both ways; no pass threshold set until the replay
  numbers exist.

## Evaluation plan

Step 1 — 28D replay. Six held-out HE episodes (ho5 split), fixed seed across tasks, including 1293 and 1300.
Configurations A-stored (stored `sonic78_nolimit` tokens), A-28D (`--replay --backend sonic`), C-28D
(`--replay --backend decoupled`), 3 repeats each. Check G0, then G1/G2.

Step 2 — synthetic 31D. Waist track, one axis at a time: yaw ±0.4 rad sine at 0.2 Hz, roll ±0.15 rad sine at
0.2 Hz, pitch 0 → 0.3 → 0 rad (2 s ramps, 2 s holds); joint limits yaw ±2.6, roll/pitch ±0.52 rad. A and C, two of
the episodes, 3 repeats.

Step 3 — closed loop. Recompute exact quantiles on HE `joint28` (`augment_joint_quantiles.py --root <dataset>`, the H100 copy used on 2026-09-25 — it is not in this repo, so
add it under `examples/g1_dex3_training/` first;
GR00T N1.7 normalizes with q01/q99 — `tests/policies/groot/test_groot_new_embodiment_defaults.py` asserts
`use_percentiles`), then train GR00T with the official recipe and verify the saved processor's q01/q99 equal the
recomputed stats. Ask the user for the H100 queue slot (GPUs 0/6, shared) before queueing. Then A-28D vs C-28D
closed loop on the same six episodes, 3 repeats, with the HE 78D GR00T `*_official_full` (A-native) as baseline.

## Tests

1. `assert`: `joint_chunk_to_sonic` on 31D with waist = `NOMINAL_BODY[12:15]` gives tokens identical to the 28D input.
2. `assert`: `--replay` returns the recorded chunk for each replan time.
3. State mapping (C host, sim, no physics step): assign distinct `qpos` values to all 29 body and 14 hand joints by
   MuJoCo name; the published state message must return each value at its documented index.
4. Command mapping (C host): send a joint message with distinct values on every arm and hand slot; the host's
   `applied_ref` / PD targets must hold each value on the expected MuJoCo joint name (decides the right-hand order;
   no settling involved). Waist command mapping is checked at the policy input: with a 31D waist command, after
   `settled` both flags are True and the logged `rpy_cmd` (`single_obs[4:7]`) is ≠ 0 and within 0.01 rad of FK of the
   commanded waist. Physical waist/torso tracking is not tested here; it is the synthetic-waist evaluation.
5. Host smoke: C stands 30 s with constant arm targets; valid, tilt < 3°.
6. Scorer: a run truncated by a forced `fell` and a run with a missing tail are both reported invalid.
7. Regression: after the `sim_scene.py` extraction, one A episode (stored tokens) scores within G0 tolerance of a
   pre-extraction run; `sonic_joint28_stream_check.py` still passes.
8. A's applied hand gains: during one A run, the sim host logs the received hand command's kp/kd once after
   `Init Done` and once during `episode`; both must be 1.5 / 0.1 on all 14 motors (otherwise C's hand gains are
   changed to match before any comparison).
9. Activation goal: a unit check that the first wrapper goal and every per-tick goal contain
   `target_upper_body_pose`, `base_height_command` and `navigate_cmd`, and that `get_action()` runs after the
   lower-body toggle without `KeyError`.

## Out of scope

Real-robot backend C (and its joint-step cap); walking and base-height commands beyond the constants above; an HE
`joint31` dataset; changes to the policy server; Unitree built-in locomotion and research WBCs.

## Risks

- Upstream `decoupled_wbc` imports may pull heavy dependencies (robocasa); if so, import only the policy, robot-model
  and config modules.
- The sim container (Python 3.10) has pinocchio 2.7, torch, mujoco 3.12 but no onnxruntime; it comes from a mounted
  `PYTHONPATH` folder. Nothing is installed on the host.
- `sim_scene.py` extraction can change A's behaviour; test 7 guards it.

## Deliverables

Code above, results in `docs/research/<run-date>-g1-wbc-backends.md`, and a pointer in `CLAUDE.md`.
