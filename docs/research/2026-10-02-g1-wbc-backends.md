# G1 whole-body control backends for 28D joint policies: SONIC vs GR00T decoupled WBC (2026-10-02)

This page reports the simulation comparison of two whole-body controllers behind 28D joint policies. Design:
`docs/superpowers/specs/2026-10-01-g1-wbc-backends-design.md`; plan: `docs/superpowers/plans/2026-10-01-g1-wbc-backends.md`.
Background: [`2026-09-28-sonic-28d-and-combined-handover.md`](./2026-09-28-sonic-28d-and-combined-handover.md) (28D through
SONIC) and [`2026-09-23-sonic-roundtrip-audit.md`](./2026-09-23-sonic-roundtrip-audit.md).

This repository is public: addresses, user names and credentials are left out on purpose. `A` below is the audit/sim
directory used by the other docs; `/run-output` is the container path of the training outputs.

## Status in one paragraph

Two whole-body backends now drive 28D (14 arm + 14 Dex3 hand) joint policies in the same MuJoCo
scene, with one validity-gated scorer: **A** = SONIC (joints encoded to tokens on the robot side, NVIDIA C++ deploy) and
**C** = NVIDIA GR00T decoupled WBC (`NVlabs/GR00T-WholeBodyControl`, `decoupled_wbc` at commit `b042411fae`: lower-body RL
policy for legs and waist, arm and hand targets straight to PD). Replay results (6 held-out Humanoid Everyday (HE)
episodes x 3 repeats, all valid): online A reproduces the stored-token result (11 of 12 gate rows within 0.07 cm;
the one miss is a table-contact event, tokens identical), C tracks the palm about 2.7-4x worse (p95 9.6-11.0 cm vs 2.7-3.5 cm on
five episodes), touches the table more, and fails gates G1 and G2. With arm gravity compensation and SONIC's stance
(controlled C) the gap shrinks 2-2.5x but A still wins. Closed loop with a 28D GR00T (3 rounds x 54 runs): the policy's
chunk-to-chunk jumps trip the arm-speed watchdog on both backends; a 0.3 s chunk blend removes almost all trips, the
sampler noise scale 0 only part of them; A keeps the better palm tracking and C touches the table far more.

31D waist extension: see the follow-up PR.

## What was added

Repository (`examples/g1_dex3_training/`; `sim_scene.py` and the A host changes come from the preceding scene PR):

| File | What it does |
| --- | --- |
| `wbc_common.py` | Pure logic shared by streamer, hosts and scorer: joint message pack/unpack, replay chunks, fall detector, termination record, run validity, done/gate files. |
| `sim_scene.py` | Scene setup extracted from the A host: table park/move-in, per-tick recording, fall check, `termination.json`. No import-time side effects. |
| `sonic_official_sim_host.py` | Backend A host, now on `sim_scene.py`; behaviour unchanged (regression PASS), adds termination record and fall detection. |
| `decoupled_wbc_sim_host.py` | Backend C host: upstream `decoupled_wbc` lower-body policy at 50 Hz control, PD for arms and hands, state back in the deploy's ZMQ format; also the host self-checks. |
| `sonic_policy_streamer.py` | `--replay`, `--action-space joint28`, `--backend decoupled`, 50 Hz `joint_ref` log. |
| `stream_backends.py` | `SonicBackend` and `DecoupledBackend` with the same calls (`start`, `wait_tick`, `begin`, `set_chunk`, `episode_tick`, `finish`); the streamer's loop is backend-agnostic. |
| `sonic_stream_eval.py` | Backend-aware scorer; validity first (`termination.json` completed, >= 98% of frames covered); palm error vs the recorded action and vs `joint_ref` (and vs the applied target for C). `leg_dev_max_rad` now covers the 12 leg joints only (it used to include the waist), so it does not compare with older `stream_eval.json` files. |
| `wbc_compare.py` | Gates G0/G1/G2 and the closed-loop table over the run directories `WBC_<cfg>_ep<E>_r<R>`. |
| `sonic_official_sim_eval.sh` | One launcher for both backends (below). |

### How to run

Launcher: `sonic_official_sim_eval.sh OUT RUN_NAME DATASET EPISODE GPU TABLE_GAP_CM [streamer args...]`. Env:

| Variable | Meaning |
| --- | --- |
| `BACKEND` | `sonic` (default) or `decoupled`; selects the host script, its mounts and `--dex3-right-order` (`swap` for A, `dataset` for C); `BACKEND=decoupled` already adds `--backend decoupled --dex3-right-order dataset`, so do not pass them again. |
| `REPLAY` | `1`: stream the episode's recorded actions, no policy server (`RUN_NAME` is `-`). |
| `HOST_ARGS` | Extra host args, decoupled only, e.g. `--waist-location lower_body`. |
| `SIM_HOST` | the GPU host that runs the containers (the script's default H100 host). |
| `LEROBOT_DIR` | lerobot code mounted for the policy server and streamer; default the official-recipe copy `lerobot-g1-official-20260929-ceb40b77`. |
| `SONIC_DIR` | Stored-token dataset used for scoring. |
| `CODE_DIR` | Script copy on the GPU host (default `$A/code`). |
| `JOINT28_DIR` | `joint28` dataset used for scoring. |

Streamer flags: `--replay`; `--action-space sonic78|joint28`; `--backend decoupled`.
A joint runs also need `--encoder-model`, `--observation-config`, `--robot-xml` (`$ENC` below). One command per
configuration (episode `$E`, repeat `$R`, `HE=../humanoid_everyday_g1_20260923/datasets`,
`ENC="--encoder-model /sonic-model/model_encoder.onnx --observation-config /sonic-model/observation_config.yaml --robot-xml /run-output/environment/g1_29dof_with_hand.xml"`):

```bash
# Astored: stored tokens through SONIC
JOINT28_DIR=/run-output/humanoid_everyday_g1_20260923/datasets/joint28 REPLAY=1 ./sonic_official_sim_eval.sh WBC_Astored_ep${E}_r$R - $HE/sonic78_nolimit_g2 $E $GPU 30 --start planner --action-space sonic78
# A28: 28D joints encoded online, SONIC
JOINT28_DIR=/run-output/humanoid_everyday_g1_20260923/datasets/joint28 REPLAY=1 ./sonic_official_sim_eval.sh WBC_A28_ep${E}_r$R - $HE/joint28_g2 $E $GPU 30 --start planner --action-space joint28 $ENC
# C28: 28D joints, decoupled WBC
JOINT28_DIR=/run-output/humanoid_everyday_g1_20260923/datasets/joint28 BACKEND=decoupled REPLAY=1 HOST_ARGS="--waist-location lower_body" ./sonic_official_sim_eval.sh WBC_C28_ep${E}_r$R - $HE/joint28_g2 $E $GPU 30 \
  --action-space joint28
```

`joint28_g2` is `joint28` with re-encoded video; scoring reads `joint28` (`JOINT28_DIR`; the launcher defaults to the Unitree
`joint28` dataset, so the HE runs set it as above). Gates:
`wbc_compare.py $A --episodes 1293,1300,1455,2207,3555,3600 --out $A/wbc_compare_step1.json`.

### GPU-host prerequisites for backend C

- `$A/upstream_b042411fae/decoupled_wbc`: only `decoupled_wbc/` from the pinned commit; both ONNX files fetched from the LFS
  media URLs (SHA-256 Balance `f645da599d4ca3d29ed273c8f4712620bb680d34977469ca3aeabe5bb9631c18`, Walk
  `7c82255b6905ffcc4468fa7f8ddcf7b70db168cf1042107ccab887cb6a8e5407`); the 65 URDF meshes replaced by gear_sonic's identical G1
  meshes, with sha256 checked against the LFS oids.
- `$A/pylib_ort310`: onnxruntime 1.20.1 and gymnasium 1.0.0, installed with `pip --target` inside the sim image, with numpy
  removed so the image's 1.26.4 is used.
- For policy runs: the GR00T base `nvidia/GR00T-N1.7-3B` and `nvidia/Cosmos-Reason2-2B` in the HF cache.
- The official-recipe lerobot code (`LEROBOT_DIR`).

### Backend C contract

- Upstream: `decoupled_wbc` at `b042411fae`; top-level config `g1_29dof_gear_wbc.yaml`, lower-body config
  `g1_gear_wbc.yaml` under `decoupled_wbc/sim2mujoco/resources/robots/g1/` (`num_obs` 516, `num_actions` 15, `rpy_cmd`,
  `height_cmd` 0.74). Policy ONNX `GR00T-WholeBodyControl-Balance.onnx` and `-Walk.onnx` (git-lfs, fetched explicitly,
  SHA-256 logged per run); the walking command is always zero, so Balance runs throughout.
- Body PD gains from `g1_29dof_gear_wbc.yaml` (legs 150/150/150/200/40/40, waist 250 x3, arms 100/100/40/40/20/20/20).
  Hand PD kp 1.5 / kd 0.1 on all 14 motors, the same values A's deploy applies (confirmed at run time: "hand gains while
  streaming: kp [1.5] kd [0.1]").
- 28D: waist location `lower_body`, torso command 0.
- Upper-body policy `IdentityPolicy` (the streamer already resamples to 50 Hz, so both backends get the same targets).
- Every goal sent through the wrapper is complete (`target_upper_body_pose`, `base_height_command` [0.74],
  `navigate_cmd` [0, 0, 0]); RL output is switched on once through the lower-body policy only (key `]`).
- Joint message: one msgpack message per 50 Hz tick (`t_wall`, `seq`, `phase`, `frame`, `q_body` f32[29], `q_hand`
  f32[14], dataset hand order). Non-finite values: previous target held and counted; finite values clipped to the
  MuJoCo joint ranges and counted; a message older than 0.1 s on arrival is dropped and counted.
- Startup: band hang, RL on at 1 s, band released at 3 s, Backspace reset onto the ground at 4 s, 9 s settled with both flags True, then `GATE/deploy_ready` and
  `GATE/settled`. Table parked away, moved in after `settled`. Streamer lead-in 2 s, hold 1 s, episode, return 2 s.
- Safety in sim: no joint-step cap (arm speed is logged).

### Host self-checks (all ok on the sim container)

`decoupled_wbc_sim_host.py OUT GATE --selfcheck contract|state|command|stand`: `contract` (upstream imports,
waist location `lower_body`), `state` (each of the 29 body and 14 hand joints returns at its documented state index), `command`
(each arm and hand slot lands on the expected MuJoCo joint; the right-hand bridge slots are name-mapped), `stand` (30 s with constant arm targets, bridge objects survive the Backspace
reset). Stand max tilt: 0.57 deg (first host), 0.45 deg (measured start pose), 1.68 deg (final START_ARMS version).

## As-built differences from the spec

| Spec | As built | Reason |
| --- | --- | --- |
| Physics 500 Hz (`SIMULATE_DT` 0.002) | 200 Hz, control 50 Hz (`DECIMATION` 4; host asserts `scene.dt * DECIMATION == 0.02`) | A's scene runs at 200 Hz (`scene.dt` 0.005), and upstream's `sim_frequency` is 200 too; both backends share the scene. |
| RL activation through `lower_body_policy.set_goal({"toggle_policy_action": True})` | key `]` on the lower-body policy (`handle_keyboard_button`), RL activated at 1 s while hung, before the band release at 3 s | Upstream's own practice; releasing with RL off drops the robot with limp legs. Never a toggle-only wrapper goal. |
| C starts at the measured pose | C settles in A's planner-stance arm pose `START_ARMS` (mean of the pre-episode records of an A run; std <= 0.007 rad across A runs) | History: C first started at the gear-WBC default arm pose, then at the measured pose, then at `START_ARMS`. The gear-WBC default arm pose and the G1 zero arm pose both put the palms at table-top height (z 0.75-0.80 m, table top 0.80 m) when the table moves in; the first C runs touched the table from sim t = 9.00 s, 22 s before the episode. A's stance has the palms near 0.62 m, and both backends now start with identical arms. Deviates from spec activation step 1 (measured pose) by design (hands and waist do start at the measured pose). |
| Termination reasons `completed`, `fell`, `timeout`, `error` | adds `aborted` | Streamer reports an unfinished episode, no joint message for 5 s without `GATE/done`, or no first message within `FIRST_MSG_S`. |
| Abort without a first message after 60 s (first ruling) | 600 s | The policy server can take about 2 min to load (MolmoAct2) before the streamer starts. |
| Launcher `LEROBOT_DIR` mounted a stale lerobot copy | default = official-recipe code (`lerobot-g1-official-20260929-ceb40b77`) | Official checkpoints need it (older code fails with an unexpected `color_jitter_params` argument); only official-recipe models remain. The launcher also stops when the policy server dies instead of waiting forever. |
| Streamer default `--policy-timeout-s 1.5` | `--policy-timeout-s 5` for every policy run, both sides | The cold first GR00T call exceeds 1.5 s; steady chunks take about 0.3 s (max chunk age stays 2 s). |
| Upstream imports only | `gymnasium` and the LFS meshes are needed | Upstream `identity_policy.py` imports `gymnasium` (not in the sim image; installed into the mounted onnxruntime folder, container numpy kept). The codeload tarball's URDF meshes are LFS pointers; replaced by the 65 G1 meshes from `gear_sonic` after checking SHA-256 == LFS oid for all 65. |

## Results: 28D replay

Six held-out HE episodes (ho5 split file `heldout_sonic78_nolimit_5pct.json`, 226 held out; picked at random with a fixed seed,
tasks with "walk" in the name excluded because both backends keep the legs at the nominal stance at an 80 cm table). Table
80 cm, `TABLE_GAP_CM` 30, `--start planner`. 3 repeats x 3 configurations x 6 episodes = 54 runs (the 51-run batch plus the 3 smoke runs of episode 1293 r1), all valid.

| Episode | Task |
| --- | --- |
| 1293, 1300 | put dumpling into plate (g1) |
| 1455 | stack two cubes (g1) |
| 3555 | open a bottle (g1), 856 frames |
| 2207 | hold a lunch bag and hand it back, 468 frames |
| 3600 | unplug a charger, 437 frames |

Means over valid repeats; palm error against FK of the recorded action (cm); table = records with table contact.

| Config | Episode | valid | palm p50 | palm p95 | palm vs `joint_ref` p95 | wrist p95 (deg) | max tilt (deg) | min feet | table records | arm jerk p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Astored | 1293 | 3/3 | 1.57 | 2.77 | - | 6.60 | 2.44 | 8.00 | 0.00 | 384.93 |
| Astored | 1300 | 3/3 | 1.77 | 2.77 | - | 6.86 | 2.74 | 8.00 | 0.00 | 400.73 |
| Astored | 1455 | 3/3 | 1.80 | 3.90 | - | 15.77 | 2.33 | 8.00 | 157.33 | 360.20 |
| Astored | 2207 | 3/3 | 1.50 | 3.10 | - | 8.11 | 2.36 | 8.00 | 17.67 | 351.53 |
| Astored | 3555 | 3/3 | 1.70 | 3.33 | - | 12.33 | 3.58 | 7.67 | 0.00 | 448.23 |
| Astored | 3600 | 3/3 | 2.00 | 3.57 | - | 16.94 | 2.82 | 8.00 | 5.00 | 451.97 |
| A28 | 1293 | 3/3 | 1.60 | 2.73 | 2.77 | 6.79 | 2.50 | 8.00 | 0.00 | 359.63 |
| A28 | 1300 | 3/3 | 1.73 | 2.73 | 2.77 | 6.75 | 2.72 | 8.00 | 0.00 | 379.00 |
| A28 | 1455 | 3/3 | 1.83 | 7.77 | 7.77 | 21.37 | 3.36 | 8.00 | 198.67 | 403.23 |
| A28 | 2207 | 3/3 | 1.50 | 3.13 | 3.17 | 8.09 | 2.32 | 8.00 | 23.67 | 365.97 |
| A28 | 3555 | 3/3 | 1.70 | 3.37 | 3.40 | 12.48 | 3.52 | 8.00 | 0.00 | 464.03 |
| A28 | 3600 | 3/3 | 1.93 | 3.53 | 3.57 | 17.01 | 2.60 | 8.00 | 5.67 | 445.03 |
| C28 | 1293 | 3/3 | 5.80 | 10.40 | 10.47 | 15.81 | 2.37 | 8.00 | 68.00 | 344.13 |
| C28 | 1300 | 3/3 | 5.50 | 11.00 | 11.00 | 16.13 | 2.86 | 8.00 | 80.33 | 337.00 |
| C28 | 1455 | 3/3 | 6.00 | 10.60 | 10.67 | 18.79 | 5.60 | 4.00 | 335.00 | 249.83 |
| C28 | 2207 | 3/3 | 6.70 | 10.67 | 10.67 | 15.89 | 2.47 | 8.00 | 0.00 | 269.87 |
| C28 | 3555 | 3/3 | 6.70 | 10.20 | 10.20 | 18.65 | 2.33 | 8.00 | 7.67 | 310.80 |
| C28 | 3600 | 3/3 | 4.93 | 9.63 | 9.80 | 17.87 | 4.80 | 6.67 | 395.67 | 296.83 |

(Episode 2006, run as the 28D base of the 31D waist gate (follow-up PR): A28 2/3 valid, C28 3/3, C28 palm p95 vs
`joint_ref` 16.8 cm.)

Gates (`wbc_compare.py`, spec thresholds):

- **G0 FAIL (online A faithful).** 11 of 12 rows pass: |A28 - Astored| is 0.00-0.07 cm on p50 (tolerance 0.5) and
  0.03 cm on p95 (tolerance 1.0) on all rows but one. The failing row is episode 1455 p95: diff 3.87 cm (tolerance 1.0).
  Cause: one A28 repeat (r2) has palm p95 15.4 cm vs 3.9 and 4.0 in the other two; the right hand caught the table
  (right palm p95 16.1 cm, 285 table contact records vs about 155). Tokens are identical in all A28 repeats (p95 0,
  max one 1/16 grid step). So the online encoding is faithful and the miss is contact chaos in "stack two cubes", but per the spec rule this is reported
  as a FAIL.
- **G1 FAIL (equal balance), 5 of 6 episodes.** All C runs are valid (no fall), but table contact records (C vs A):
  68 vs 0 (1293), 80 vs 0 (1300), 335 vs 199 (1455), 8 vs 0 (3555), 396 vs 6 (3600); only 2207 passes (0 vs 24).
  Tilt (C vs A): 5.6 vs 3.4 (1455) and 4.8 vs 2.6 (3600) degrees, the others within +2 deg.
- **G2 FAIL (C better at tracking).** Palm p95 vs `joint_ref`: C 9.8-11.0 cm on all six, A 2.8-7.8 cm. Mean gain -6.56 cm;
  C better on 0 of 6. Wrist p95: C 15.8-18.8 deg vs A 6.8-21.4 deg.

Read: A's online encoding reproduces the stored result within 0.00-0.07 cm (means over 3 repeats) in 11 of 12 rows. C tracks the palm about 2.5-4.5x worse at p50
(p50 4.9-6.7 cm vs 1.5-2.0 cm for A28 and Astored) and has more table contact on five of six episodes.

## Facts worth knowing

- **C's arm error is PD tracking.** Palm error against the applied target is about the same as against the sent target
  (ep 1293 r1: 5.8/10.5 vs 5.8/10.4 cm). Upstream ships `enable_gravity_compensation=False` (BaseConfig default, inherited
  by `deploy_g1`) and the comparison uses upstream defaults, so the arms sag under PD alone. C therefore under-represents
  what the decoupled WBC can do with compensation.
- **C stands lower.** Height command 0.74 (upstream): pelvis 0.747 m vs A 0.764 m (ep 1293 r1, final start pose); before the start-pose fix 0.745 vs 0.779 m.
- **Replay is deterministic enough.** On the H100 host A28 matches Astored within 0.00-0.07 cm (means over 3 repeats) in 11 of 12 G0 rows.
- **C message counts:** stale 0 and non-finite 0 in every C run (27 runs, 6 of them the 31D waist runs of the follow-up PR); seq gaps 0-153 per run (most runs a few; 153 on C28 ep 3600 r2, 826 messages vs about 975, about 15% missing; 76 and 86 on two others); clipped values 87-1558 per run (1517-1558 on ep 3555).
- **Batch-runner pitfall.** Reading `$?` after `$(date)` in the same `echo` logged "exit 0" for a failed launch (A28 ep 2006 r3,
  NVIDIA deploy exited while hanging). Validity is always taken from `stream_eval.json`, never from the runner log.
- **Stopping a launcher locally does not stop its remote chain.** After `TaskStop` (no tty) the remote ssh chain survived and
  started a streamer when the next same-named policy server came up. Kill the remote chain and `docker rm` its containers on
  the host.
- Hand gains seen by the sim at run time (A): kp 1.5, kd 0.1 on all motors, after `Init Done` and while streaming.

## Closed loop: model

The 28D GR00T for the closed loop trained (smoke run passed, processor q01/q99 equal to the dataset stats): `groot_joint28_ho5_official_full`
(official recipe, 20K steps x batch 32, bf16, HE `joint28` with exact quantiles). Exit 0, 20000/20000 steps in 3:13:04
(1.73 step/s), "End of training" 2026-10-02 15:31 UTC; checkpoints 005000, 010000, 015000, 020000 and last under
`/run-output/humanoid_everyday_g1_20260923/runs/groot_joint28_ho5_official_full`. Results are in "Closed loop: results"
below (after the controlled-C section, because the closed loop runs controlled C).

## Controlled C (C28g)

C as shipped mixes two things into its palm error: PD sag of the arms under gravity (upstream ships
`enable_gravity_compensation` off) and a lower stance (pelvis about 0.747 m at height command 0.74, A about 0.764 m), so
the palms sit lower relative to the table. C28g removes both:

- `--gravity-comp arms`: each 50 Hz step writes upstream's arm gravity torques
  (`RobotModel.compute_gravity_compensation_torques(q, "arms")` at the measured pose, fixed base, as upstream's
  `sync_env`) into the bridge's feed-forward `tau`; legs, waist and hands get 0. The `command` self-check asserts this.
- `--height-cmd H`: the lower-body RL's base height command. H is picked with the `stand` self-check (it prints the
  mean pelvis height), so the pelvis matches A's (~0.764 m). Stand (30 s, gravity compensation on): command 0.74 ->
  pelvis 0.746 m, 0.76 -> 0.755, 0.78 -> 0.761, **0.79 -> 0.764 (max tilt 2.1 deg, used)**, 0.80 -> 0.766, 0.82 fails
  the 3 deg tilt check. Without compensation, 0.74 -> 0.747 m.

Run: `BACKEND=decoupled REPLAY=1 HOST_ARGS="--gravity-comp arms --height-cmd H"` with run names `WBC_C28g_ep<E>_r<R>`
over the same 6 episodes x 3 repeats, then
`wbc_compare.py AUDIT --episodes 1293,1300,1455,2207,3555,3600 --test C28g` (G1/G2 against the existing A28 runs).
`termination.json` records `gravity_comp` and `height_cmd`.

**Results (2026-10-05, 18 runs, all valid; A28 and C28 are the step-1 runs).** Palm error vs `joint_ref` p50 / p95 (cm),
mean over 3 repeats, and table hit records:

| Episode | A28 | C28 | C28g |
| --- | --- | --- | --- |
| 1293 | 1.60 / 2.77, 0 | 5.80 / 10.47, 68 | 2.10 / 4.50, 2 |
| 1300 | 1.73 / 2.77, 0 | 5.50 / 11.00, 80 | 2.00 / 4.70, 13 |
| 1455 | 1.83 / 7.77, 199 | 6.00 / 10.67, 335 | 2.47 / 5.60, 189 |
| 2207 | 1.50 / 3.17, 24 | 6.70 / 10.67, 0 | 3.00 / 5.70, 0 |
| 3555 | 1.70 / 3.40, 0 | 6.70 / 10.20, 8 | 2.73 / 7.33, 36 |
| 3600 | 2.00 / 3.57, 6 | 4.90 / 9.80, 396 | 2.20 / 5.37, 120 |

- Gravity compensation plus the matched height cut C's palm error by about 2-2.5x (p50 4.9-6.7 -> 2.0-3.0 cm, p95
  9.8-11.0 -> 4.5-7.3 cm). Most of C28's error was PD sag, as suspected.
- C28g is still behind A: p95 worse on 5 of 6 episodes (better only on 1455, where A hits the table). **G2 FAIL**
  (gain -1.63 cm, better on 1 of 6). **G1 FAIL** on table contact (C28g hits more than A on 1293, 1300, 3555, 3600;
  tilt is lower than A's on every episode).
- Wrist orientation p95 (deg) is within 1.3 deg of A's (slightly worse) on 4 episodes and better on 1455 and 3600 (11.1 vs 21.4, 11.0 vs 17.0).
- The remaining gap is at p50 too (~0.5-1.5 cm), so it is not only contact events; plausible causes are the arm PD
  (kp 20-100, no velocity feed-forward) lagging the 50 Hz targets, against SONIC's learned tracking. Not tested.
- Ran on the main H100 host (A28/C28 ran on the second one); the same host's SONIC regression below matched the
  pre-refactor numbers exactly, so the hosts are comparable.

Recorded stance for reference: the raw HE source (`USC-PSI-Lab/Humanoid-Everyday-G1`) stores
`observation.leg_joints` (12 legs + 3 waist; our `joint28` keeps only arms and hands). Forward kinematics of the sim's
G1 model with both feet flat gives a pelvis height of 0.773-0.776 m for the 6 test episodes (knees ~0.48 rad, pelvis
tilt 1.8-2.9 deg); the same method on sim runs is within 1 mm of the recorded sim pelvis (0.7637 vs 0.7628 m). So the
HE robot (Unitree's own lower-body controller) stood ~1 cm higher than SONIC and ~2.8 cm higher than C at command 0.74;
C cannot reach it within its stable range (command 0.80 -> 0.766 m, 0.82 fails the tilt check). Palm error is measured
in the pelvis frame and does not see this offset; table contact does.

SONIC regression on the restructured streamer (`stream_backends.py`), same host and batch: episode 1293, HE GR00T
official, seed 0: palm p50 / p95 3.9 / 10.0 cm (pre-refactor 3.9 / 10.0), `termination.json` completed.

## Closed loop: results (2026-10-06)

Configurations, each 6 episodes x 3 repeats, policy server per run on the same GPU, `--policy-timeout-s 5`:
**A28cl** = 28D GR00T, joints encoded online, SONIC; **C28cl** = the same policy through controlled C
(`HOST_ARGS="--gravity-comp arms --height-cmd 0.79"`); **Anative** = the 78D HE GR00T (`groot_sonic78sonicstate_ho5_official_full`)
through stored-token SONIC. Three rounds that differ only in how chunk switches are smoothed:

- round 1 (`*cl`): server `--noise-seed 0`, chunks switch at once (`--chunk-blend-s 0`, the evaluated setting so far);
- round 2 (`*clb`): as round 1 plus streamer `--chunk-blend-s 0.3`;
- round 3 (`*cln`): server `--noise-scale 0` (the GR00T setting accepted on the robot, runs 20-21; ported to this branch's
  server), no blend.

Valid runs (of 18) and, over valid runs, palm error p50 / p95 (cm) vs the streamer's `joint_ref` (the policy's own
targets; not defined for Anative) and table hit records:

| Config | Round 1 (no smoothing) | Round 2 (blend 0.3 s) | Round 3 (noise scale 0) |
| --- | --- | --- | --- |
| A28cl | 8 valid; 1.75 / 3.30; 19 | **16**; 1.57 / 3.17; 24 | 11; 1.63 / 3.09; 20 |
| C28cl | 12; 2.57 / 7.83; 130 | **18**; 2.32 / 7.03; 136 | 14; 2.29 / 8.70; 224 |
| Anative | 17; -; 21 | **18**; -; 34 | 18; -; 37 |

- Every invalid run is the streamer's arm-speed watchdog (> 6 rad/s, mostly the left elbow). The cause is the 28D
  GR00T's chunk-to-chunk jumps: in the first C smoke run the elbow target went 0.36 -> -0.29 rad at the second chunk
  (t_ep 0.4 s). SONIC does not absorb these jumps either (A trips more often than C in rounds 1 and 3).
- A 0.3 s chunk blend removes almost all trips (A 8 -> 16, C 12 -> 18 valid) without hurting tracking.
- Noise scale 0 only helps partly (A 11, C 14). Its trips are deterministic per episode (A: none valid on 1300 and
  2207; C: none on 1293), so the jumps are in the policy's mean prediction for a changed observation, not only sampling
  noise.
- When runs complete, A tracks the policy's targets better (p95 ~3.1-3.3 cm vs C 7.0-8.7 cm) and C touches the table
  5-10x more, as in replay. G1 (C vs A) fails in every round.
- Palm error vs the recorded demo (closed loop: how far the policy's motion drifts from the demo, not a tracking
  measure) is p50 4-7 cm, p95 17-31 cm for all configs.

## Next steps

1. Closed loop with noise scale 0 plus the 0.3 s blend (the two smoothers together) as the candidate default for 28D
   joint policies.
2. C did not pass G2 in replay or G1 in closed loop. Arm velocity feed-forward or higher arm gains would test the
   remaining PD-lag explanation.
3. Real-robot C needs a joint-step cap first (none in sim).
