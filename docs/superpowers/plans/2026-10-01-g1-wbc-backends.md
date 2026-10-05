# G1 WBC backends (SONIC vs GR00T decoupled WBC): plan as built and how to reproduce

**Spec:** `docs/superpowers/specs/2026-10-01-g1-wbc-backends-design.md` (revision 3, approved). **Results:**
`docs/research/2026-10-02-g1-wbc-backends.md`.

**Status.** Step 1 of the spec's evaluation plan (28D replay through A and C, 6 held-out HE episodes x 3 configurations
x 3 repeats) is built, run and gated. Step 3 (closed loop with a 28D GR00T) has its model trained; the batch is
prepared but not run. The 31D waist extension (spec step 2: `--action-space joint31`, synthetic waist, waist location
`lower_and_upper_body`, torso and `rpy_cmd` metrics, `syn` gates) and the exact-quantile tool
(`augment_joint_quantiles.py`) are follow-up PRs. This page condenses the original task-by-task plan into what was
built and how to rerun it.

**Backends.** **A** = SONIC: the streamer encodes each 30 Hz joint chunk to SONIC tokens
(`sonic_targets.joint_chunk_to_sonic`), which NVIDIA's C++ deploy tracks. **C** = GR00T decoupled WBC
(`NVlabs/GR00T-WholeBodyControl`, `decoupled_wbc/` at `b042411fae38ee4d1af9aac82a37a1f8d14d6dd0`): the lower-body RL
policy holds legs and waist, and arm and hand targets go straight to PD.

## Files and responsibilities

All under `examples/g1_dex3_training/` (the `sim_scene.py` extraction, the A host's termination record and fall
detection and the streamer's `GATE/done` episode end come from the preceding PR).

| File | Responsibility |
| --- | --- |
| `wbc_common.py` | numpy-only helpers: `joint_ref` layout (`REF_NAMES`, `WAIST`), `to_joint_ref`, `replay_chunk`, `check_replay_width`, joint message pack/unpack, `log_settings`, `run_validity`, done/termination files, `FallDetector`. |
| `sonic_policy_streamer.py` | `--replay` / `--replay-horizon`, `--backend sonic\|decoupled`, the 50 Hz `joint_ref` log for every joint run, the decoupled startup / episode / return, refusal of `--backend decoupled` with `--dex3-right-order swap` or token-only options. |
| `decoupled_wbc_sim_host.py` | Backend C host in the shared scene: upstream policy, PD targets by joint name, message hold/clip/drop counters, `termination.json`, self-checks. |
| `sonic_official_sim_eval.sh` | One launcher for both backends: `BACKEND`, `REPLAY`, `HOST_ARGS`, `SIM_HOST`, `LEROBOT_DIR`, `SONIC_DIR`. |
| `sonic_stream_eval.py` | Validity-gated, backend-aware scorer. |
| `robot_run_smoothness.py` | `windowed_diff` (third difference for arm and palm jerk), also a standalone robot-log smoothness report. |
| `wbc_compare.py` | Gates G0/G1/G2 and the closed-loop table over `WBC_<cfg>_ep<E>_r<R>` run dirs. |

Local tests (`tests/datasets/`): `test_g1_dex3_wbc_common.py`, `test_g1_dex3_streamer_cli.py`,
`test_g1_dex3_stream_eval.py`, `test_g1_dex3_wbc_compare.py`; `test_g1_dex3_sonic_targets.py` and
`test_g1_dex3_sonic_token_stream.py` guard A's unchanged encoder and resampler. MuJoCo/upstream behaviour is checked
by the host self-checks in the sim container.

## Message and log contracts

- **`joint_ref`, f32[31]:** arms 14 (motor 15..28) | Dex3 hands 14 (dataset order) | waist 3 (motor 12..14, yaw, roll,
  pitch). The 31-value layout is the spec's contract; with 28D actions the waist is always `NOMINAL_BODY[12:15]`.
  The streamer logs it every 50 Hz tick of a joint run (both backends): the target after resampling,
  `--chunk-blend-s` and lead-in/return blending. For A it is a second `ChunkResampler` over the same 30 Hz joint
  chunk; A's token path is unchanged. Token runs (`sonic78`) have no `joint_ref`.
- **Joint message** (streamer -> C host, ZMQ PUB topic `joints`, msgpack, one per tick): `t_wall`, `seq`, `phase`
  (`wait first chunk`, `leadin`, `hold`, `episode`, `return`), `frame` (-1 outside the episode), `q_body` f32[29]
  motor order (legs nominal and ignored), `q_hand` f32[14] dataset order (left thumb0-2, middle0-1, index0-1; right
  thumb0-2, index0-1, middle0-1). `pack_joint_message` refuses non-finite values.
- **State** (C host -> streamer, topic `g1_debug`): `body_q` motor order, `left_hand_q` / `right_hand_q` dataset
  order, the format the deploy publishes, so `--state-source robot` works for both backends.
- **Gates:** `GATE/deploy_ready`, `GATE/settled`, `GATE/done` (JSON `{"episode_end": "completed" | reason}`, written
  atomically). **`sim/termination.json`:** `reason` completed | fell | aborted | timeout | error, plus `phase`,
  `detail`, and for C `counts` (messages, stale, nonfinite, seq_gap, clipped_values) and `waist_location`.
- **C's `sim_state.npz` extras:** `applied_ref[T,31]` (after clip/hold), `clipped[T,31]`, `held[T]`, `seq[T]`.

## Backend C contract as built

- Lower body: `G1GearWbcPolicy` with the `GEAR_WBC_CONFIG` of `control/main/teleop/configs/g1_29dof_gear_wbc.yaml`
  (516 obs = 86 x 6, 15 actions; the host asserts both), `model_path`
  `policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx` (SHA-256 logged per run).
  Walking command 0 and height 0.74, so Balance runs throughout.
- Upper body: `IdentityPolicy`; every goal through the wrapper is complete (`target_upper_body_pose`,
  `base_height_command` [0.74], `navigate_cmd` [0, 0, 0]).
- Robot model `waist_location` `lower_body`: torso command 0 (the only value this PR's host accepts).
- Gains: body `MOTOR_KP` / `MOTOR_KD` from `g1_29dof_gear_wbc.yaml`; Dex3 kp 1.5 / kd 0.1 on all 14 motors (A's deploy
  defaults). Targets are written by joint name into gear_sonic's bridge command slots (no DDS, no slot swap), so C
  runs with `--dex3-right-order dataset`, A with `swap`.
- Messages older than 0.1 s are dropped (held, counted); non-finite messages are held and counted; finite values are
  clipped to the MuJoCo joint ranges (counted).
- Timeline: band hang, RL on at 1 s (key `]` on the lower-body policy), band released at 3 s, Backspace reset onto
  the ground at 4 s, 9 s: both policy flags asserted True, table moved in, `deploy_ready` + `settled`, fall detection
  armed. Streamer: holds the measured pose while the first chunk is computed, 2 s joint-space lead-in, 1 s hold,
  episode, 2 s return. Host stops 3 s after `GATE/done`; aborts after 5 s without a message (no `GATE/done`) or when
  no first message arrives within `FIRST_MSG_S` (600 s) of `settled`; times out at 1800 s sim time.
- No joint-step cap in sim (arm speed is logged); required before real-robot C.

As-built deviations from the spec (details and reasons in the results doc):

| Spec | As built |
| --- | --- |
| Physics 500 Hz | 200 Hz (A's scene, upstream `sim_frequency` 200), control 50 Hz (`DECIMATION` 4, asserted) |
| RL activation via `toggle_policy_action` after release | key `]` on the lower-body policy at 1 s while hung (releasing with RL off drops the robot) |
| C starts at the measured pose | arms settle in A's planner-stance pose `START_ARMS` (palms ~0.62 m, below the 80 cm table); hands and waist at the measured pose |
| Termination reasons without `aborted` | adds `aborted` (streamer ended early, silent streamer, no first message) |
| First-message abort 60 s | 600 s (`FIRST_MSG_S`; MolmoAct2 takes about 2 min to load) |
| Upstream imports only | also `gymnasium` (imported by `identity_policy.py`) and the URDF meshes (LFS pointers in the tarball) |
| Launcher mounted a stale lerobot copy | `LEROBOT_DIR` defaults to the official-recipe code; launcher stops when the policy server dies |
| `--policy-timeout-s 1.5` | 5 s for policy runs (cold first GR00T call) |

## Scorer and gates

`sonic_stream_eval.py RUN_DIR JOINT28_ROOT SONIC_ROOT EPISODE`:

1. Missing `streamer.npz` or `sim/sim_state.npz`: invalid before MuJoCo is imported.
2. `run_validity` (same rule for A and C): `termination.json` says `completed`, episode ticks cover >= 98% of the
   episode's frames, no streamer tick outside the sim recording, and (C) targets held on <= 2% of episode records.
   Invalid runs print `{"valid": false, "reason": ...}` and exit 2.
3. Metrics during `episode`: stability (max tilt, min floor contacts, leg deviation, min pelvis z); palm error in the
   pelvis frame against FK of the recorded action, of `joint_ref` and (C) of `applied_ref`; per-hand p95; wrist
   orientation; table hits / contact records / clearance; arm speed p95, arm and palm jerk p95; hands vs dataset;
   C's message counts; for SONIC runs tokens/hands vs the stored dataset. Old logs without `backend` /
   `action_space` score as SONIC `sonic78` (`log_settings`).

`wbc_compare.py AUDIT_DIR [--out F] [--episodes ...]`: per-episode means over repeats. Coverage rule for every gate:
each side needs >= 3 runs and all of them valid; a run dir without a readable `stream_eval.json` is an invalid run;
a requested episode without runs fails. G0: |A28 - Astored| <= max(0.5 cm, spread) on recorded-action palm p50 and
max(1.0 cm, spread) on p95. G1: max tilt <= A + 2 deg and table contact records <= A's. G2: C's `joint_ref` palm p95
>= 1.0 cm lower than A28 on the mean and lower on >= 4 of 6 episodes (n episodes: ceil(2n/3)). `G1_closed` and
`closed_loop` rows appear when `A28cl` / `C28cl` / `Anative` runs exist.

## Reproducing on the GPU host

Everything runs in containers on the GPU host; nothing is installed on the host itself. Placeholders: `$A` = audit /
sim directory on the host (mounted as `/audit` or `/a`), `$RO` = training outputs (mounted as `/run-output`),
`$SIM_IMAGE` = the gear_sonic sim image (Python 3.10, mujoco 3.12, pinocchio 2.7, zmq, msgpack; python
`/opt/sonic-sim/bin/python`), `$LR_IMAGE` = the lerobot image (venv `/run-output/environment/venv/bin/python`). The
launcher hard-codes these for the reference host; `SIM_HOST` selects the ssh alias.

**1. Code sync** (the launcher reads scripts from `CODE_DIR`, default `$A/code`):

```bash
cd examples/g1_dex3_training
tar c --exclude=__pycache__ . | ssh $SIM_HOST "docker run --rm -i -v $A:/a --entrypoint sh $LR_IMAGE -c \
  'rm -rf /a/code_wbc && mkdir -p /a/code_wbc && tar x -C /a/code_wbc'"
export CODE_DIR=$A/code_wbc
```

**2. Upstream checkout** (`$A/upstream_b042411fae`, only `decoupled_wbc/`, written through a container):

```bash
C=b042411fae38ee4d1af9aac82a37a1f8d14d6dd0
curl -sSL https://codeload.github.com/NVlabs/GR00T-WholeBodyControl/tar.gz/$C \
  | tar xz --strip-components=1 GR00T-WholeBodyControl-$C/decoupled_wbc
for f in Balance Walk; do p=decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-$f.onnx
  curl -sSL -o $p https://media.githubusercontent.com/media/NVlabs/GR00T-WholeBodyControl/$C/$p; done
sha256sum decoupled_wbc/sim2mujoco/resources/robots/g1/policy/*.onnx | tee ONNX_SHA256.txt
```

Expected SHA-256: Balance `f645da599d4ca3d29ed273c8f4712620bb680d34977469ca3aeabe5bb9631c18`, Walk
`7c82255b6905ffcc4468fa7f8ddcf7b70db168cf1042107ccab887cb6a8e5407` (a file starting with `version https://git-lfs`
means the media URL failed). The tarball's 65 URDF meshes are LFS pointers too: replace each with the identically
named G1 mesh from gear_sonic, after checking its SHA-256 equals the pointer's `oid`.

**3. onnxruntime and gymnasium** into a mounted folder, keeping the image's numpy (1.26.4):

```bash
docker run --rm -v $A:/a --entrypoint bash $SIM_IMAGE -c "set -e
  /opt/sonic-sim/bin/python -m pip install --no-cache-dir --target /a/pylib_ort310 onnxruntime==1.20.1 gymnasium==1.0.0
  rm -rf /a/pylib_ort310/numpy /a/pylib_ort310/numpy-* /a/pylib_ort310/bin
  PYTHONPATH=/a/pylib_ort310 /opt/sonic-sim/bin/python -c 'import onnxruntime, numpy; print(onnxruntime.__version__, numpy.__version__)'"
```

**4. Host self-checks** (each ends with `selfcheck <name>: ok`):

```bash
for c in contract state command stand; do
  docker run --rm --gpus device=$GPU -e NVIDIA_DRIVER_CAPABILITIES=all -e MUJOCO_GL=egl -e PYTHONPATH=/pylib_ort:/upstream \
    -w /workspace/GR00T-WholeBodyControl/gear_sonic_deploy -v $CODE_DIR:/code:ro \
    -v $A/upstream_b042411fae/decoupled_wbc:/upstream/decoupled_wbc:ro -v $A/pylib_ort310:/pylib_ort:ro \
    --entrypoint /opt/sonic-sim/bin/python $SIM_IMAGE -u /code/decoupled_wbc_sim_host.py /tmp/o /tmp/g --selfcheck $c
done
```

`contract`: upstream imports, ONNX hashes, joint groups (arms upper body, waist not), gravity orientation in MuJoCo's
(w, x, y, z) order. `state`: each of the 29 body and 14 hand joints at its documented index. `command`: each arm and
hand slot lands on the right MuJoCo joint (prints the right-hand slot names). `stand`: 30 s with constant targets
after the startup sequence, no fall, max tilt < 3 deg.

**5. Runs.** `sonic_official_sim_eval.sh OUT RUN_NAME DATASET EPISODE GPU TABLE_GAP_CM [streamer args]` from the
workstation. `BACKEND=decoupled` already adds `--backend decoupled --dex3-right-order dataset` (do not repeat them);
`REPLAY=1` skips the policy server (`RUN_NAME` is `-`). HE episodes use `--start planner` for A and
`TABLE_GAP_CM=30`. One repeat of step 1 for episode `$E`, repeat `$R`:

```bash
export JOINT28_DIR=/run-output/humanoid_everyday_g1_20260923/datasets/joint28
export SONIC_DIR=/run-output/humanoid_everyday_g1_20260923/datasets/sonic78_nolimit_g2
HE=../humanoid_everyday_g1_20260923/datasets
ENC="--encoder-model /sonic-model/model_encoder.onnx --observation-config /sonic-model/observation_config.yaml --robot-xml /run-output/environment/g1_29dof_with_hand.xml"
REPLAY=1 ./sonic_official_sim_eval.sh WBC_Astored_ep${E}_r$R - $HE/sonic78_nolimit_g2 $E $GPU 30 --start planner --action-space sonic78
REPLAY=1 ./sonic_official_sim_eval.sh WBC_A28_ep${E}_r$R - $HE/joint28_g2 $E $GPU 30 --start planner --action-space joint28 $ENC
BACKEND=decoupled REPLAY=1 HOST_ARGS="--waist-location lower_body" \
  ./sonic_official_sim_eval.sh WBC_C28_ep${E}_r$R - $HE/joint28_g2 $E $GPU 30 --action-space joint28
```

`joint28_g2` is `joint28` with re-encoded video (decodes much faster; data and meta shared); scoring reads `joint28`.
The batch is a loop over `E in 1293 1300 1455 2207 3555 3600` and `R in 1 2 3` (54 runs, about 6 min each on one
GPU). Closed loop (prepared, not run) uses the same commands without `REPLAY=1`, with the policy run as `RUN_NAME`,
`SERVER_EXTRA="--noise-seed 0"`, `--policy-timeout-s 5`, and configs `A28cl`, `C28cl` (28D GR00T
`groot_joint28_ho5_official_full`) and `Anative` (78D HE GR00T `*_official_full`, stored-token SONIC).

**6. Gates:**

```bash
docker run --rm -v $A:/audit -v $CODE_DIR:/code:ro -v $RO:/run-output:ro --entrypoint /run-output/environment/venv/bin/python \
  $LR_IMAGE /code/wbc_compare.py /audit --episodes 1293,1300,1455,2207,3555,3600 --out /audit/wbc_compare_step1.json
```

If G0 fails, compare A28's `policy_token_vs_stored_abs` before drawing any A-vs-C conclusion.

Batch pitfalls: take validity from `stream_eval.json`, never from a runner's exit code (reading `$?` after
`$(date)` logged "exit 0" for a failed launch). Stopping a launcher locally does not stop its remote chain; kill it
and `docker rm` its containers on the host.

## Verification evidence

- Local: the unit tests above pass; `py_compile`, `bash -n` on the launcher and ruff are clean.
- Self-checks contract, state, command and stand pass in the sim container; stand max tilt 0.57 deg (first host),
  0.45 deg (measured start pose), 1.68 deg (final `START_ARMS` version). A's hand gains seen by the sim at run time:
  kp 1.5, kd 0.1 on all motors, after `Init Done` and while streaming. The `sim_scene.py` extraction passed the A
  regression run (preceding PR).
- Step 1 (54 runs, all valid): online A reproduces the stored-token result within 0.00-0.07 cm in 11 of 12 G0 rows
  (G0 FAIL on episode 1455 p95, one repeat's table contact; tokens identical across repeats). C's palm p95 vs
  `joint_ref` is 9.8-11.0 cm against A's 2.8-7.8 cm; G1 FAIL (more table contact on 5 of 6 episodes), G2 FAIL (C
  better on 0 of 6). C's arm error is PD tracking (applied vs sent target almost equal; upstream ships without arm
  gravity compensation). Full tables in the results doc.
- C message counts: no stale and no non-finite messages in any C run.

## Next

Closed-loop batch (A28cl, C28cl, Anative); C with upstream arm gravity compensation; a joint-step cap before
real-robot C; a height command matched to A's stance for table tasks; the 31D waist extension (follow-up PR).
