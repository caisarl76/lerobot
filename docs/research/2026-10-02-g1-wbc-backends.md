# G1 whole-body control backends for 28D / 31D joint policies: SONIC vs GR00T decoupled WBC (2026-10-02)

This page reports the simulation comparison of two whole-body controllers behind 28D / 31D joint policies. Design:
`docs/superpowers/specs/2026-10-01-g1-wbc-backends-design.md`; plan: `docs/superpowers/plans/2026-10-01-g1-wbc-backends.md`.
Background: [`2026-09-28-sonic-28d-and-combined-handover.md`](./2026-09-28-sonic-28d-and-combined-handover.md) (28D through
SONIC) and [`2026-09-23-sonic-roundtrip-audit.md`](./2026-09-23-sonic-roundtrip-audit.md).

This repository is public: addresses, user names and credentials are left out on purpose. `A` below is the audit/sim
directory used by the other docs; `/run-output` is the container path of the training outputs.

## Status in one paragraph

Two whole-body backends now drive 28D (14 arm + 14 Dex3 hand) and 31D (28D + 3 waist) joint policies in the same MuJoCo
scene, with one validity-gated scorer: **A** = SONIC (joints encoded to tokens on the robot side, NVIDIA C++ deploy) and
**C** = NVIDIA GR00T decoupled WBC (`NVlabs/GR00T-WholeBodyControl`, `decoupled_wbc` at commit `b042411fae`: lower-body RL
policy for legs and waist, arm and hand targets straight to PD). Replay results (6 held-out Humanoid Everyday (HE)
episodes x 3 repeats, all valid): online A reproduces the stored-token result (11 of 12 gate rows within 0.07 cm;
the one miss is a table-contact event, tokens identical), C tracks the palm about 2.7-4x worse (p95 9.6-11.0 cm vs 2.7-3.5 cm on
five episodes), touches the table more, and fails gates G1 and G2. With a synthetic 31D waist, C follows the commanded
waist (gate passes), while A's waist path trips the arm-speed watchdog in 5 of 6 runs. The **closed-loop comparison
(28D GR00T through A and C) is deferred**: the batch is prepared but not run, because GPU 7 was given to another training queue.

## What was added

Repository (`examples/g1_dex3_training/`, branch `feat/g1-wbc-backends`):

| File | What it does |
| --- | --- |
| `wbc_common.py` | Pure logic shared by streamer, hosts and scorer: joint message pack/unpack, replay chunks, synthetic waist, fall detector, termination record, run validity, done/gate files. |
| `sim_scene.py` | Scene setup extracted from the A host: table park/move-in, per-tick recording, fall check, `termination.json`. No import-time side effects. |
| `sonic_official_sim_host.py` | Backend A host, now on `sim_scene.py`; behaviour unchanged (regression PASS), adds termination record and fall detection. |
| `decoupled_wbc_sim_host.py` | Backend C host: upstream `decoupled_wbc` lower-body policy at 50 Hz control, PD for arms and hands, state back in the deploy's ZMQ format; also the host self-checks. |
| `sonic_policy_streamer.py` | `--replay`, `--action-space joint28\|joint31`, `--synthetic-waist`, `--backend decoupled`, 50 Hz `joint_ref` log. |
| `sonic_targets.py` | `joint_chunk_to_sonic()` accepts `[N, 31]` (waist columns replace the nominal waist in the encoder's body target). |
| `sonic_stream_eval.py` | Backend-aware scorer; validity first (`termination.json` completed, >= 98% of frames covered); palm error vs the recorded action and vs `joint_ref` (and vs the applied target for C); torso and `rpy_cmd` error for 31D. |
| `wbc_compare.py` | Gates G0/G1/G2 and the synthetic-waist gates over the run directories `WBC_<cfg>_ep<E>_r<R>`. |
| `sonic_official_sim_eval.sh` | One launcher for both backends (below). |
| `augment_joint_quantiles.py` | Exact q01/q99 (copy of the 2026-09-25 H100 script). |

### How to run

Launcher: `sonic_official_sim_eval.sh OUT RUN_NAME DATASET EPISODE GPU TABLE_GAP_CM [streamer args...]`. Env:

| Variable | Meaning |
| --- | --- |
| `BACKEND` | `sonic` (default) or `decoupled`; selects the host script, its mounts and `--dex3-right-order` (`swap` for A, `dataset` for C); `BACKEND=decoupled` already adds `--backend decoupled --dex3-right-order dataset`, so do not pass them again. |
| `REPLAY` | `1`: stream the episode's recorded actions, no policy server (`RUN_NAME` is `-`). |
| `HOST_ARGS` | Extra host args, decoupled only, e.g. `--waist-location lower_body` (28D) or `lower_and_upper_body` (31D). |
| `SIM_HOST` | the GPU host that runs the containers (the script's default H100 host). |
| `LEROBOT_DIR` | lerobot code mounted for the policy server and streamer; default the official-recipe copy `lerobot-g1-official-20260929-ceb40b77`. |
| `SONIC_DIR` | Stored-token dataset used for scoring. |
| `CODE_DIR` | Script copy on the GPU host (default `$A/code`). |
| `JOINT28_DIR` | `joint28` dataset used for scoring. |

Streamer flags: `--replay`; `--action-space sonic78|joint28|joint31`; `--synthetic-waist` (31D only; yaw +-0.4 rad sine at 0.2 Hz, roll
+-0.15 rad sine at 0.2 Hz, pitch 0 -> 0.3 -> 0 rad with a 2 s ramp up, one 2 s hold and a 2 s ramp down, one axis at a time); `--backend decoupled`.
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
# A31syn: 31D with synthetic waist, SONIC
JOINT28_DIR=/run-output/humanoid_everyday_g1_20260923/datasets/joint28 REPLAY=1 ./sonic_official_sim_eval.sh WBC_A31syn_ep${E}_r$R - $HE/joint28_g2 $E $GPU 30 --start planner --action-space joint31 --synthetic-waist $ENC
# C31syn: 31D with synthetic waist, decoupled WBC
JOINT28_DIR=/run-output/humanoid_everyday_g1_20260923/datasets/joint28 BACKEND=decoupled REPLAY=1 HOST_ARGS="--waist-location lower_and_upper_body" ./sonic_official_sim_eval.sh WBC_C31syn_ep${E}_r$R - $HE/joint28_g2 $E $GPU 30 \
  --action-space joint31 --synthetic-waist
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
- 28D: waist location `lower_body`, torso command 0. 31D: `lower_and_upper_body`; the commanded waist is in the
  upper-body target and its FK gives `torso_orientation_rpy` (`rpy_cmd`).
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

`decoupled_wbc_sim_host.py OUT GATE --selfcheck contract|state|command|waist|stand`: `contract` (upstream imports,
both waist locations), `state` (each of the 29 body and 14 hand joints returns at its documented state index), `command`
(each arm and hand slot lands on the expected MuJoCo joint; the right-hand bridge slots are name-mapped), `waist`
(`rpy_cmd` follows the commanded waist), `stand` (30 s with constant arm targets, bridge objects survive the Backspace
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

(Episode 2006, a 28D base for the 31D gate, is in the next section: A28 2/3 valid, C28 3/3.)

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

## Results: synthetic 31D waist

Episodes 3555 and 2006 (2006: "fold and pass coat", 507 frames, not one of the six; a fallback pick because only 3555 of the six episodes is >= 16 s; the extra 2006 28D runs exist because the synthetic-waist gate needs a same-episode 28D base). Waist track as above. Means over valid repeats.

| Config | Episode | valid | palm p50 | palm p95 | wrist p95 (deg) | max tilt (deg) | min feet | table records | arm jerk p95 | torso yaw p95 (rad) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A31syn | 2006 | 0/3 | - | - | - | - | - | - | - | - |
| A31syn | 3555 | 1/3 | 1.70 | 4.00 | 13.32 | 3.48 | 8.00 | 0.00 | 456.20 | 0.28 |
| C31syn | 2006 | 3/3 | 8.07 | 17.57 | 26.08 | 11.04 | 2.67 | 199.00 | 479.77 | 0.16 |
| C31syn | 3555 | 3/3 | 7.03 | 10.57 | 19.98 | 4.08 | 8.00 | 11.33 | 330.47 | 0.16 |

- **syn_C PASS.** Palm p95 vs the same episode's 28D replay: 10.6 vs 10.2 cm (3555) and 17.6 vs 16.8 cm (2006), inside +2 cm.
  Torso error p95 (roll / pitch / yaw): 0.09 / 0.07 / 0.16 rad (3555) and 0.10 / 0.09 / 0.16 rad (2006); the waist
  reaches the policy (`rpy_cmd` error p95 <= 0.01 rad). Episode 2006 shows tilt about 11 deg and 174-214 table records, as in its 28D replay.
- **syn_A FAIL.** 5 of 6 runs were aborted by the streamer's arm-speed watchdog (right wrist pitch, arm joint 12, 6.2-7.6
  rad/s against the limit of 6): SONIC whips the right wrist under the synthetic waist. The one valid run (3555 r1) has palm p95 4.0 cm
  (28D base 3.4) and torso yaw p95 0.28 rad (about 16 deg); roll / pitch p95 0.03 / 0.04 rad.

## Facts worth knowing

- **C's arm error is PD tracking.** Palm error against the applied target is about the same as against the sent target
  (ep 1293 r1: 5.8/10.5 vs 5.8/10.4 cm). Upstream ships `enable_gravity_compensation=False` (BaseConfig default, inherited
  by `deploy_g1`) and the comparison uses upstream defaults, so the arms sag under PD alone. C therefore under-represents
  what the decoupled WBC can do with compensation.
- **C stands lower.** Height command 0.74 (upstream): pelvis 0.747 m vs A 0.764 m (ep 1293 r1, final start pose); before the start-pose fix 0.745 vs 0.779 m.
- **Replay is deterministic enough.** On the H100 host A28 matches Astored within 0.00-0.07 cm (means over 3 repeats) in 11 of 12 G0 rows.
- **C message counts:** stale 0 and non-finite 0 in every C run (27 runs); seq gaps 0-153 per run (most runs a few; 153 on C28 ep 3600 r2, 826 messages vs about 975, about 15% missing; 76 and 86 on two others); clipped values 87-1558 per run (1517-1558 on ep 3555).
- **Batch-runner pitfall.** Reading `$?` after `$(date)` in the same `echo` logged "exit 0" for a failed launch (A28 ep 2006 r3,
  NVIDIA deploy exited while hanging). Validity is always taken from `stream_eval.json`, never from the runner log.
- **Stopping a launcher locally does not stop its remote chain.** After `TaskStop` (no tty) the remote ssh chain survived and
  started a streamer when the next same-named policy server came up. Kill the remote chain and `docker rm` its containers on
  the host.
- Hand gains seen by the sim at run time (A): kp 1.5, kd 0.1 on all motors, after `Init Done` and while streaming.

## Closed loop (deferred)

Not run. The user gave the second H100 host's GPU 7 to another training queue after this session's 28D GR00T training.
That training finished (smoke run passed, processor q01/q99 equal to the dataset stats): `groot_joint28_ho5_official_full`
(official recipe, 20K steps x batch 32, bf16, HE `joint28` with exact quantiles). Exit 0, 20000/20000 steps in 3:13:04
(1.73 step/s), "End of training" 2026-10-02 15:31 UTC; checkpoints 005000, 010000, 015000, 020000 and last under
`/run-output/humanoid_everyday_g1_20260923/runs/groot_joint28_ho5_official_full`. The closed-loop batch is prepared but not run: policy server
`--noise-seed 0` for all configs (this branch's server has no `--noise-scale`, so this is not the robot-accepted
noise-scale-0 setting), configurations A28cl, C28cl and Anative (the 78D HE GR00T `*_official_full` through stored-token
SONIC) x 6 episodes x 3 repeats, plus a re-run of A28 ep 2006 r3.

**Results to be added.**

## Next steps

1. Closed-loop results (A28cl vs C28cl vs Anative); G1 applies, palm error reported both ways.
2. C with arm gravity compensation (upstream `enable_gravity_compensation`) to separate PD sag from the controller.
3. A31syn without the watchdog (`--max-arm-speed 0`) to see whether SONIC stays stable under the synthetic waist.
4. Real-robot C needs a joint-step cap first (none in sim).
5. Height command matched to A's stance for table tasks (C stands lower).
