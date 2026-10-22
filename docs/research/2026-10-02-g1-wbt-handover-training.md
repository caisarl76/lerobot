# G1 WBT handover-bottle: VLA training on our own teleop data (2026-10-02 ~)

Four VLAs (GR00T, Pi0.5, Psi0, XR-1) are fine-tuned on our own GR00T-WBC teleoperation recording of the task "pick
drink bottle from the table and handover". Each model is trained twice: from its public pretrained weights (**base**)
and from our Humanoid Everyday checkpoint in the same action format (**he**). Everything runs as one queue on
h100_174 GPU 7.

## User decisions (2026-10-02)

- Action format **sonic78 at the recording's native 50 Hz** (no resampling to 30 Hz). The streamer is still
  hard-coded to 30 Hz and needs a 50 Hz mode before sim or robot runs.
- **Both starting points** (base and he): 8 runs.
- **1/4 of the HE training budgets** (43 episodes; the HE recipes would mean ~34–56 passes over the data).
- h100_174 GPU 7 for the queue. When another session's job took GPU 7, the user chose to wait for it (see Incidents).

## Dataset

Source: `/mnt/data/jihun/datasets/G1_WBT_GR00T/handover_bottle_260930_subset_261002` (workstation; LeRobot v2.1,
50 Hz, 43 episodes, 18,970 frames, one D435 `ego_view` 640×480 camera; a curated subset of `handover_bottle_260930`).

The recording already holds the SONIC motion tokens that drove the real robot (`action.motion_token`, 64) and the
measured robot state, which is the pose SONIC reached. So it needs no SONIC encoding and no state relabeling
(`sonic_state_relabel.py`). `examples/g1_dex3_training/prepare_wbt_sonic78.py` matches joints by name to
`sonic_targets.ACTION_NAMES`:

| Feature                         | Content                                                             |
| ------------------------------- | ------------------------------------------------------------------- |
| `observation.state` (28)        | arms 14 + Dex3 hands 14 from `observation.state`                    |
| `action` (78)                   | `action.motion_token` 64 + hand targets 14 from `action.wbc`        |
| `observation.images.egocentric` | `ego_view`, renamed to the HE camera key so the HE checkpoints load |

- The recording's joint names are the same URDF names as ours, so the hand order (left thumb, middle, index; right
  thumb, index, middle) needs no hand-written mapping. `teleop.*_hand_joints` uses another order and is not used.
- Output: `/mnt/data/jihun/datasets/G1_WBT_GR00T/handover_bottle_sonic78_50hz_261002` (LeRobot v3, AV1 re-encoded,
  stats with quantiles).
- Held out: episodes **27 and 35** (5 %, seed 0), in `handover_bottle_sonic78_50hz_261002.heldout.json`.
- The right thumb_0 hand target is constant (std 0) in this recording.

## Layout on h100_174

h100_174 has its own disk; it does not share `/mnt/data01` with h100. All writes go through containers.

- Job root: `/mnt/data01/jhkim/model_weight/g1_dex3_20260922/g1_wbt_handover_20261002` (`/run-output/...` in
  containers), with `datasets/`, `configs/`, `init/` (HE checkpoints), `stats/` (XR-1), `runs/`, `logs/`,
  `queue_gpu7.txt`, `official_queue.sh`, `retarget_init_checkpoint.py`.
- Container `jihun-lerobot-wbt-174-gpu7-20261002`: GPU 7, `--memory 200g`, `--shm-size 16g`,
  `LEROBOT_VIDEO_DECODER_CACHE_SIZE=5000`, `HF_HUB_OFFLINE=1`.
  - Code: `/mnt/data01/jhkim/code/lerobot-g1-psi0-xr1-dev`, a superset of the official-recipe tree: identical GR00T
    and Pi0.5 code, plus Psi0, XR-1 and their optimizers.
- Base weights: `nvidia/GR00T-N1.7-3B` and `lerobot/pi05_base` were downloaded directly on h100_174 (~41 MB/s).
  `nvidia/Cosmos-Reason2-2B` (GR00T N1.7 backbone) and the `google/paligemma-3b-pt-224` tokenizer are gated, so they
  were copied from h100's cache. Psi0 and XR-1 weights were already on the host.

## Recipes (1/4 of the HE ones)

Steps count micro-batches (see the scheduler pitfall in `2026-09-30-official-retraining-handover.md`).

| Model | Micro × accum | Steps  | Samples | Warmup       | HE config it derives from                   |
| ----- | ------------- | ------ | ------- | ------------ | ------------------------------------------- |
| GR00T | 32 × 1        | 5,000  | 160K    | 5 % (preset) | `groot_sonic78sonicstate_ho5_official_full` |
| Pi0.5 | **4 × 8**     | 60,000 | 240K    | 2,000        | `pi05_sonic78sonicstate_ho5_official_full`  |
| Psi0  | 16 × 8        | 10,000 | 160K    | 250          | `psi0_sonic78sonicstate_ho5_matched640k`    |
| XR-1  | 16 × 3        | 7,500  | 120K    | 375          | `xr1_sonic78sonicstate_ho5_official_full`   |

- Every other setting is copied from the HE config, including chunk lengths in frames. So a chunk covers less time
  at 50 Hz: GR00T 40 frames = 0.8 s, Psi0/XR-1 30 = 0.6 s, Pi0.5 50 = 1.0 s.
- Pi0.5 was planned at 8 × 4 (30,000 steps) and ran out of memory on the shared GPU 7. It needs ~72.5 GB, and GPU 7
  also holds a 2.2 GB vLLM process from another user. 4 × 8 keeps the effective batch at 32 and the sample count,
  with steps and schedule doubled; memory is still 72.5 GB (weights, optimizer and EMA dominate). The original
  configs are kept as `*.bak-bs8`.
- XR-1 normalization: `stats/xr1_sonic78.json` from `xr1_action_stats.py --relative sonic78 --exclude 27 35`.
- Configs: `{groot,pi05,psi0,xr1}_wbt50_{base,he}_{smoke,full}.json`. Smoke tests are 20 steps (Pi0.5: 40).

## Starting from the HE checkpoints

`init/{groot,pi05,xr1,psi0}_he` are the HE final weights: GR00T and Pi0.5 (EMA) from the workstation, XR-1 and
Psi0 matched640k from h100.

When training starts from a checkpoint, `lerobot_train` replaces only the stats of the standard (un)normalizer steps.
Two policies would therefore keep the HE dataset's normalization:

- GR00T keeps the stats in its own processor steps (`groot_n1_7_pack_inputs`, `groot_action_unpack_unnormalize`).
- XR-1 keeps them as persistent model buffers (`action_mean/std`, `state_q01/q99`, `state_min/max`).

`examples/g1_dex3_training/retarget_init_checkpoint.py` fixes this in place before training. It rebuilds the
checkpoint's processors from the new dataset's stats, as a base run builds them. For XR-1 it also reloads the
model's stats and re-saves the model. After it ran, the GR00T and Pi0.5 processor stats were checked to equal the
dataset's (e.g. hand q99 0.4025, 0.7084, 1.7500).

## Results (finished 2026-10-05)

All 8 full runs exited 0. Final training losses (scales differ between models; compare base vs he within a model):

| Model | base: time, loss  | he: time, loss    |
| ----- | ----------------- | ----------------- |
| GR00T | 48 min, 0.041     | 56 min, 0.033     |
| Pi0.5 | 7 h 15 min, 0.021 | 7 h 05 min, 0.017 |
| Psi0  | 2 h 01 min, 1.088 | 1 h 55 min, 1.079 |
| XR-1  | 2 h 22 min, 3.699 | 2 h 21 min, 1.064 |

- Every he run ends lower than its base run. These are training losses only; the held-out comparison (episodes 27, 35) is the next step.
- Checkpoints: `runs/<name>/checkpoints/last/pretrained_model` (Pi0.5 also `pretrained_model_ema`). Each run also keeps
  `training_state/`; Pi0.5 runs take 71 GB each. h100_174's `/mnt/data01` was at 99 % (185 GB free) on 10-06.
- Psi0 he started from a much higher loss than the other he runs (smoke: 15.9 vs base 22.4, against 3–6× lower for
  the others). The weights load completely; the cause is normalization. Psi0 uses MIN_MAX, and the HE ranges are
  ~2.7× wider for tokens and ~4.1× for arm state, so the rewritten stats rescale its inputs and targets. The user
  chose to keep the rewritten stats (2026-10-05) rather than the HE stats.
- The Psi0 he config needed the checkpoint's own `config.json` as its policy block: Psi0 loads a checkpoint only with
  `vlm_config`, which exists only in the saved config (originals kept as `*.bak-novlmconfig`).

## Held-out open-loop comparison (2026-10-06)

`openloop_smooth.py` on episodes 27 and 35, replanning every 20 frames (0.4 s at 50 Hz), run in the training
container (outputs in `eval/` of the job root). Std units of the dataset's action; err = mean |executed − recorded|,
seam = jump at the chunk switch. Zero noise (`temp:0`) was best or equal for every model; shown here:

| Model      | Tokens err | Tokens seam | Hands err | Hands seam |
| ---------- | ---------- | ----------- | --------- | ---------- |
| GR00T base | 0.332      | 0.191       | 0.068     | 0.124      |
| GR00T he   | 0.315      | 0.184       | **0.053** | **0.066**  |
| Pi0.5 base | 0.300      | **0.146**   | 0.079     | 0.172      |
| Pi0.5 he   | **0.284**  | 0.151       | 0.106     | 0.115      |
| Psi0 base  | 0.303      | 0.162       | 0.077     | 0.092      |
| Psi0 he    | 0.303      | 0.170       | 0.089     | 0.075      |
| XR-1 base  | 0.316      | 0.171       | 0.173     | 0.147      |
| XR-1 he    | 0.287      | 0.158       | 0.168     | 0.132      |

- Step inside a chunk at zero noise is 0.014–0.019 for tokens (recording 0.019).
- HE init lowers the token error for GR00T, Pi0.5 and XR-1 (5–9 %), not for Psi0. Psi0 he was rescaled 3–4× by the
  new MIN_MAX stats, see Results.
- Best token accuracy: Pi0.5 he and XR-1 he. Best hands: GR00T he. XR-1 hands are 2–3× worse than the others; XR-1
  predicts hands relative to the measured state.
- XR-1 base with random noise is unusable: token err 0.645 and steps 35× the recording's. At zero noise it is normal.
- Token errors are about twice the HE models' on HE held-out episodes (0.13–0.15). With 41 training and 2 held-out
  episodes, the differences between models (a few hundredths) are within the noise. Robot or sim runs should
  decide.
- `openloop_smooth.py` now skips action dimensions with zero std in the dataset. Here the right thumb_0 is constant,
  and the clipped std turned XR-1's tiny state-relative offsets on it into ~130 std errors. The first pass (before
  the fix) is kept in `eval/v1/`.

## Closed-loop sim, 50 Hz end to end (2026-10-06)

Official SONIC sim on h100_174 (`SIM_HOST=h100_174 HF_CACHE=/mnt/data01/jhkim/huggingface sonic_official_sim_eval.sh`, code copy `sonic_roundtrip_20260923/code_wbt`,
run dirs `WBT*\*`). Policy server + streamer in dataset-image mode at the dataset's 50 Hz, held-out episode 27,
`--start planner --max-token-step 0.05`, noise scale 0.5, table 25 cm unless noted. All runs completed: 465 of the 475
episode frames were scored, with no rejected or stale chunks (chunk time median 0.19 s for GR00T, 0.27 s for Pi0.5).

| Run                | Palm err p50 / p95 | p95 left / right | Wrist err p50 | Tilt max | Token vs stored p50 |
| ------------------ | ------------------ | ---------------- | ------------- | -------- | ------------------- |
| GR00T he           | **3.7 / 9.7 cm**   | 9.3 / 12.6 cm    | **7°**        | 8.0°     | 0.020               |
| GR00T he, no table | 4.3 / 10.7 cm      | 9.4 / 14.7 cm    | 9°            | 8.7°     | 0.022               |
| Pi0.5 he           | 9.2 / 17.6 cm      | 10.8 / 20.0 cm   | 32°           | 8.6°     | 0.058               |
| Psi0 he            | 4.2 / 20.3 cm      | 13.1 / 23.8 cm   | 10°           | 9.1°     | 0.027               |
| XR-1 he            | 9.9 / 34.0 cm      | 12.8 / 41.7 cm   | 28°           | 8.0°     | 0.058               |

- **Reference = measured joints.** The scorer's reference (`wbt_joint28_reference.py`) is the arm pose the real
  robot measured (`observation.state`). Against the commanded joints (`action.wbc`), all runs scored ~14 / 34 cm,
  and so did a replay of the recording's own tokens (12.9 / 32.7 cm, `WBT_replay_ep27`). In that replay, the sim arm
  joints match the real robot's measured ones within 1.2° (right) and 2.7° (left), while the real right arm stayed ~9°
  from its commands (it holds the bottle). SONIC's decoder output is a PD setpoint, not the reached pose.
- So the sim reproduces this real recording. Ranking in closed loop: GR00T he (best median and tail), Psi0 he (median
  close to GR00T, tail 2× worse), Pi0.5 he (2.5× GR00T's median, 32° wrist error), XR-1 he (right hand drifts; the
  token slew limit was active on 30 ticks vs 9 for GR00T). Open-loop token error ranked Pi0.5 and XR-1 first, so
  open loop did not predict closed-loop quality here.
- Psi0 and XR-1 servers on h100_174 need `HF_CACHE=/mnt/data01/jhkim/huggingface` (CLIP / Qwen3-VL are not in the
  shared `/mnt/data01/huggingface` there); the first Psi0 run failed on the missing CLIP text encoder.
- The table made no difference (it was not the cause of the error).
- Results scored against the commanded joints are kept as `stream_eval.wbccmd.json` in each run dir.

### Sim with the robot settings (2026-10-12)

Same harness, noise 0 + `--chunk-blend-s 0.3` (the launcher defaults since 10-08), held-out episodes 27 and 35,
`code_wbt` updated to the merged branch. The merged scorer marks runs covering < 98 % of the episode as invalid; episode
27 covers 465 of 475 frames (97.9 %) as in the earlier runs, so it was rescored with the threshold at 97 %
(`stream_eval_cov97.json`).

| Run             | Palm err p50 / p95 | p95 left / right | Wrist err p50 | Tilt max |
| --------------- | ------------------ | ---------------- | ------------- | -------- |
| GR00T he, ep 27 | 4.2 / 9.9 cm       | 9.8 / 11.3 cm    | 8°            | 7.8°     |
| GR00T he, ep 35 | 7.8 / 10.0 cm      | 10.0 / 9.9 cm    | 25°           | 7.7°     |
| Psi0 he, ep 27  | 4.0 / 12.3 cm      | 10.4 / 17.3 cm   | 8°            | 9.0°     |
| Psi0 he, ep 35  | 3.9 / 13.9 cm      | 11.8 / 23.4 cm   | 9°            | 7.8°     |

- With these settings, Psi0 he improves (p95 20.3 → 12–14 cm on ep 27) and is level with GR00T he. GR00T he stays
  about the same on ep 27 but is worse on ep 35 (median 7.8 cm, wrist 25°). Psi0 he is worth a robot trial.

## Real robot, first runs (2026-10-08)

Workstation GPU 0, port 5561 (the XR-1 no-state server held GPU 1 / 5560), launcher defaults (noise 0, chunk blend
0.3 s), `POLICY_FPS=50`, `--max-token-step 0.05`, planner start/end, 30 s, task "pick drink bottle from the table and
handover". Logs: `pc2_222:~/g1_sonic_eval/runs/run3{3..6}_wbt_*`. Smoothness by `robot_run_smoothness.py` (dataset std):

| Run | Model    | Duration | Token seam / step | Arm speed p95 | Jerk p95 | Step-limited ticks | Outcome (user)                                  |
| --- | -------- | -------- | ----------------- | ------------- | -------- | ------------------ | ----------------------------------------------- |
| 33  | GR00T he | 30 s     | 0.015 / 0.014     | 1.62 rad/s    | 309      | 0 %                | reached and picked the bottle, handover posture |
| 34  | GR00T he | 30 s     | 0.012 / 0.013     | 1.17 rad/s    | 239      | 0 %                | (same session)                                  |
| 35  | GR00T he | 30 s     | 0.011 / 0.014     | 1.50 rad/s    | 250      | 0 %                | (same session)                                  |
| 36  | Pi0.5 he | 0.4 s    | – / 0.065         | 6.91 rad/s    | 1998     | 39 %               | watchdog stop: right arm joint at 6.2 rad/s     |
| 37  | Pi0.5 he | 30 s     | 0.011 / 0.011     | 0.99 rad/s    | 242      | 1 %                | succeeded: pick and handover                    |

- **GR00T he works on the robot:** it reaches the bottle, picks it up and goes to the handover posture. Its smoothness
  matches the HE GR00T with noise 0 + blend (runs 24–27, jerk p95 214–276).
- **Pi0.5 he: one watchdog stop, then a success.** In run36 the right hand started closed on the robot (it is
  closed in the planner stance, but open at the start of all 43 recordings). Pi0.5's second chunk (~0.3 s) snapped
  it open by 1.67 rad in one tick and moved the arm hard, so the watchdog ended the episode. The recordings contain
  such instant hand switches too (up to 1.78 rad per frame, teleop trigger). In run37, from the same closed-hand start,
  Pi0.5 he succeeded at pick and handover, as smoothly as GR00T he; it still switched the hand by up to 1.64 rad in one
  tick, but without a fast arm move. So the abrupt start is intermittent.
- After the watchdog stop, the streamer exited with `terminate called without an active exception` (harmless; the
  robot was already back in planner mode). It did not reproduce on the workstation in 12 stops; the likely cause is
  the camera receiver thread still inside OpenCV/libzmq at interpreter exit on PC2. Since 10-12 the streamer stops and
  joins that thread before exiting (73750ab2).
- **Closed hand at the start (fixed as an option, 10-12):** in planner mode the deploy keeps the Dex3 hands closed and
  ignores streamed hand targets, so the policy's first observation showed a fist. `--hold-rest-s 1.5` holds the rest
  pose with open hands in POSE mode before the first observation, like the recordings (343028d2). Robot-tested in
  runs 57–62 (below): it held the rest pose for 1.5 s before the first chunk, as intended.

## Real robot, 20-episode cells (2026-10-14)

Same settings as above plus `--hold-rest-s 1.5`. First cell: Pi0.5 base trained on 20 episodes
(`wbt_pi05_wbt20_base_full`, EMA, bf16).

| Run | Server GPU | Duration | Chunk latency | Arm speed p95 | Jerk p95 | Right arm max move | Outcome (user)                                                                                   |
| --- | ---------- | -------- | ------------- | ------------- | -------- | ------------------ | ------------------------------------------------------------------------------------------------ |
| 57  | 0          | 1 s      | > 0.6 s       | –             | –        | –                  | invalid: "no valid chunk for 1 s"                                                                |
| 58  | 0          | 1 s      | > 0.6 s       | –             | –        | –                  | invalid: same                                                                                    |
| 59  | 0          | 1 s      | > 0.6 s       | –             | –        | –                  | invalid: same                                                                                    |
| 60  | 1          | 30 s     | 0.30 s        | 0.52 rad/s    | 202      | 50°                | reached toward the bottle but not close enough, closed the grip on nothing                       |
| 61  | 1          | –        | –             | –             | –        | –                  | no `.npz` (log ends at POSE mode: streamer killed or crashed)                                    |
| 62  | 1          | 30 s     | 0.30 s        | 0.66 rad/s    | 214      | 62°                | reached the bottle (slightly far), grasped it in an unstable grip, rotated the wrist, no release |

Run 37 (Pi0.5 he, 41 episodes, success) for comparison: arm speed p95 0.99 rad/s, jerk 242, right arm 90°.

- **Runs 57–59 are not trials.** Workstation GPU 0 also drives the desktop (gnome-shell and Chrome at ~30 % each)
  and an unrelated `dimos_dev` process. Chunks took > 0.6 s against the 0.4 s replan, so the second chunk missed the
  1 s chunk-age limit. On the idle GPU 1 the same server answers in 0.28 s (workstation) / 0.30 s (PC2). Run robot
  servers on GPU 1, and check that the streamer's `policy ready: X s per chunk` is below 0.4.
  Second cell, Pi0.5 he trained on the same 20 episodes (`wbt_pi05_wbt20_he_full`), GPU 1, the bottle placed differently
  in each run. User score: **3 of 5** (per-run outcomes not itemized; run64 left no `.npz`, like run61).

| Run | Duration | Chunk latency | Arm speed p95 | Jerk p95 | Right arm max move | Left hand range |
| --- | -------- | ------------- | ------------- | -------- | ------------------ | --------------- |
| 63  | 30 s     | 0.30 s        | 0.93 rad/s    | 201      | 81°                | 1.43            |
| 64  | –        | –             | –             | –        | –                  | –               |
| 65  | 30 s     | 0.30 s        | 0.83 rad/s    | 239      | 96°                | 0.12            |
| 66  | 30 s     | 0.30 s        | 1.01 rad/s    | 206      | 82°                | 1.45            |
| 67  | 30 s     | 0.30 s        | 0.98 rad/s    | 224      | 79°                | 1.43            |

Third cell, Pi0.5 base trained on 41 episodes (`wbt_pi05_base_full`), GPU 1 (0.30 s per chunk), bottle position varied.
User score: **1 of 5** (runs 68–72; runs 68 and 69 left no `.npz`).

| Run | Duration | Arm speed p95 | Jerk p95 | Right arm max move | Left hand range |
| --- | -------- | ------------- | -------- | ------------------ | --------------- |
| 70  | 30 s     | 0.69 rad/s    | 228      | 64°                | 0.68            |
| 71  | 30 s     | 0.84 rad/s    | 237      | 61°                | 0.81            |
| 72  | 30 s     | 0.80 rad/s    | 235      | 74°                | 1.41            |

- **HE pretraining helps Pi0.5 at 20 episodes:** Pi0.5 he 20 scores 3/5 where Pi0.5 base 20 scores 0/3 (user counts run61
  as a trial). With HE init the arm reaches as far as the 41-episode HE model (79–96° vs 90°), and in 3 runs the left hand
  works through its full range for the handover (1.43–1.45 vs 1.65 in run 37); base 20 never did (0.14–0.16).
  Fourth cell, Pi0.5 he 41 re-tested with `--hold-rest-s 1.5` (runs 73–77, GPU 1, 0.28 s per chunk), bottle position
  varied: user score **3 of 5**. Only run77 saved an `.npz` (30 s, right arm 81°, left hand range 1.44); runs 73–76 each
  reached POSE mode ~30 s after the previous one, so they were stopped early by hand.

GR00T base trained on 20 episodes (`wbt_groot_wbt20_base_full`, runs 78–82, GPU 1, 0.19 s per chunk): user score
**3.5 of 5** (one partial success). Logs only for run78 (right arm 63°, left hand range 0.30) and run82 (70°, 0.38);
runs 79–81 were stopped early.

GR00T he trained on 20 episodes (`wbt_groot_wbt20_he_full`), first evaluation session `s01_wbt20_groot_he_handover`
(`g1_groot_real_run.sh session`, 5 trials, 30 s cap, every trial saved; copy in
`/mnt/data/jihun/g1_eval_sessions/s01_wbt20_groot_he_handover`): **2 of 5**. Failures: "dropped the object" (trials 2
and 3), "grasp failed / missed" (trial 5). Trial 3 was recorded as a success at the robot; the user corrected it to
"dropped the object" at review (failure around 16.5 s, in "hand over the bottle"), and the session files were fixed. The first attempt at trial 2 was a watchdog stop (arm joint 11 at 6.9 rad/s) and
was redone; it was probably a false stop: before 71413403 the watchdog timed the first episode ticks by `t_ep`, which
the 1 s blend-in shares, and read 3–4× too fast (run48 replayed: 6.6 → 2.9 rad/s). Sessions from s02 on use the fix.

GR00T base trained on 41 episodes (`wbt_groot_base_full`), session `s02_wbt_groot_base_handover` (5 trials, first
robot use of the wall-clock watchdog fix 71413403: no watchdog stops): **4 of 5**; the failure was "grasp failed /
missed" (trial 4).

GR00T he trained on 41 episodes (`wbt_groot_he_full`), session `s03_wbt_groot_he_handover`: **5 of 5**, no watchdog
stops.

Psi0 he trained on 41 episodes (`wbt_psi0_he_full`, downloaded from HF, bf16 VLM with `tune_vlm false`, server on
workstation GPU 1 at 0.29 s per chunk), session `s04_wbt_psi0_he_handover` (2026-10-15): **2 of 5**, no watchdog stops.
Failures: "dropped the object" (trial 2), "grasp failed / missed" (trials 4 and 5, stopped at 9.4 and 6.9 s). Psi0 base
was not run.

All 8 GR00T/Pi0.5 cells plus Psi0 he 41, user scores (bottle position varied per trial; noise 0, blend 0.3, slew 0.05, `--hold-rest-s 1.5`):

| Policy | Init | 20 episodes | 41 episodes |
| ------ | ---- | ----------- | ----------- |
| Pi0.5  | base | 0/3         | 1/5         |
| Pi0.5  | he   | 3/5         | 3/5         |
| GR00T  | base | 3.5/5       | 4/5         |
| GR00T  | he   | 2/5         | 5/5         |
| Psi0   | he   | not run     | 2/5         |

- **Pi0.5 (user scores):** base 20 0/3, he 20 3/5, base 41 1/5, he 41 3/5 (runs 73–77; 1/2 in runs 36–37 without
  `--hold-rest-s`).
- **GR00T (user scores):** base 20 3.5/5, he 20 2/5 (3/5 before the review correction), base 41 4/5, he 41 5/5.
- **Conclusion (user, 2026-10-14): HE pretraining helps Pi0.5 at both sizes and GR00T only at 41 episodes; GR00T at
  20 episodes is inconclusive.** Pi0.5: base → he 0/3 → 3/5 at 20 episodes and 1/5 → 3/5 at 41, and Pi0.5 he 20
  beats Pi0.5 base 41. GR00T: base → he 4/5 → 5/5 at 41 episodes (the best cell). At 20 episodes GR00T base is ahead
  (3.5/5 vs 2/5, after the s01 trial 3 correction), but with 5 trials per cell one trial flips this comparison, so
  more trials per cell are needed before reading it either way.
- **Psi0 (user score):** he 41 2/5 (s04, 10-15), below Pi0.5 he 41 (3/5) and GR00T he 41 (5/5); no base cell, so it
  says nothing about HE init for Psi0.
- **Runs 61, 64, 68, 69, 73–76 and 79–81 left no `.npz`** (use `g1_groot_real_run.sh session` from now on: Enter ends a trial and each trial is saved with its verdict): each log ends at "POSE mode", written ~20 s after the previous run ended,
  so the streamer was stopped early (likely Ctrl-C, which also kills `tee`) and the log was not saved.
- **Pi0.5 base 20 is smooth but falls short:** it reaches less far than the 41-episode HE model (right arm 50–62°
  vs 90°), grasps at best unstably, and never releases for the handover (left hand range 0.14–0.16 vs 1.65 in run 37).
  Score (user): 0 of 3 (runs 60–62).

Check progress:

```bash
ssh h100_174 'cd /mnt/data01/jhkim/model_weight/g1_dex3_20260922/g1_wbt_handover_20261002; \
  for f in logs/*.exit; do echo "$f=$(cat $f)"; done; grep -v "^#" queue_gpu7.txt'
```

## Real robot, 94-episode GR00T he (2026-10-15/16)

`wbt_groot_handover261015_he_full` is GR00T he trained on the 43 old episodes plus the 10-14 recording, without
its ep 0: 94 episodes ([2026-10-15 note](2026-10-15-g1-handover-261014-out-of-view.md)). It is compared on the robot
with `wbt_groot_he_full` (41 episodes), same task, settings and server (workstation GPU 1, noise 0, blend 0.3, slew
0.05, `--hold-rest-s 1.5`, 30 s cap). Verdicts are the operator's; VLM failure labels (Genon `gpt-6.1-sol`, since the
h100 Qwen was down) agreed on every episode and were reviewed in the visualizer, where the user moved both failure
frames earlier.

| Session | Model                              | Trials | Success | Failures                                                                |
| ------- | ---------------------------------- | ------ | ------- | ----------------------------------------------------------------------- |
| s06     | `wbt_groot_he_full` (41 ep)        | 5      | 2/5     | grasp failed / missed (trials 3–5)                                      |
| s07     | `wbt_groot_he_full` (41 ep)        | 5      | 4/5     | grasp failed / missed (trial 4, stopped at 8.4 s; "grasp" at frame 104) |
| s08     | `wbt_groot_handover261015_he_full` | 10     | 9/10    | did not reach the object (trial 7, ran to the cap; "reach" at frame 88) |

- **94 episodes: 9/10 against 6/10 for the 41-episode model** (s06 + s07) on the 10-15/16 setup. Successful
  s08 trials took 11–16 s.
- **s06 → s07 is the same model:** 2/5 then 4/5, so one 5-trial session swings by 2 trials. s03 (5/5, 10-14) used the
  same model too; with it the 41-episode model is at 11/15. Compare models at 10 trials or more.
- s05 (10-15, one trial, `wbt_groot_he_full`) is left out: a second bottle was in the scene; the user scored it a
  fail ("dropped the object").
- Sessions and exports: `/mnt/data/jihun/g1_eval_sessions/s0{6,7,8}_*`; reviewed exports in
  `_visualizer/workspace/drafts/98ad57bbb4114e41928ffc085c111ef9` (s07) and `…/5cb635dd2bc94a7db0be5957cc59302c` (s08),
  each under `retained-dataset`.

## Real robot, 75-episode merged GR00T he (2026-10-20)

`wbt_groot_handover261019_he_full` is GR00T he trained on `handover_bottle_merged_261019` (the user's merge of
reviewed runs, converted to sonic78 50 Hz): 73 training episodes, 60,526 frames, held out 27/35. It holds the old
41 episodes, 25 segments of the 10-14 recording (the review kept 21 of its 51 episodes) and 7 episodes of the
10-19 recording (6 kept of 8, one split). Same init, recipe and server as s08 (GPU 1, noise 0, blend 0.3, slew 0.05,
`--hold-rest-s 1.5`), but the session ran with the launcher's default 120 s cap instead of 30 s (`MAX_S` was not
set). Training notes in the [2026-10-15 note](2026-10-15-g1-handover-261014-out-of-view.md#3-gr00t-he-run).

| Session | Model                              | Trials | Success | Failures                                                                     |
| ------- | ---------------------------------- | ------ | ------- | ---------------------------------------------------------------------------- |
| s09     | `wbt_groot_handover261019_he_full` | 10     | 4/10    | did not reach the object ×2, stalled / froze ×2, dropped ×1, grasp missed ×1 |

- **4/10 against 9/10 for the 94-episode model (s08).** Trial 1 (index 0) holds two attempts: the first succeeded
  (grasp 8.3 s, release ~11.6 s); the robot then went back toward ready, stood still for ~60 s and ended with the hand
  opening and closing at ~3 Hz. The operator scored it a fail; the user corrected it to success at review (10-21),
  in `results.jsonl` and `trial_000.npz` on PC2 and the workstation and in the dataset and draft
  `eval_session.json` (originals kept as `results.jsonl.orig`). The other successes took 81, 72 and 21 s (s08:
  11–16 s); with s08's 30 s cap only trials 1 (its first attempt) and 10 pass, 2/10.
- **Why the second attempt stalled:** the policy has no memory (`n_obs_steps` 1), so the first attempt does not
  carry over; the input does. 157 of the 200 training frames nearest the stalled pose are from the 10-19 recordings,
  where the robot waited at ready while the operator reset the bottle, and in those frames the arm moves half as
  much over the next 2 s as on average (0.35 against 0.67 rad). The old 43 episodes never start from that pose.
- **Operator:** the reach stalls about 2 cm from the bottle, with the arm still or creeping; the return to the
  ready pose after the handover is good.
- **Logs agree:** right-arm joint speed median 0.047 rad/s (s08 0.335), still (< 0.05 rad/s) 53 % of the policy
  time (s08 18 %), longest still stretch 3.2 s per trial (s08 0.6 s).
- **Likely cause, the 10-19 episodes:** each of them is a continuous recording of several pick-and-handover
  cycles (median 6 right-hand closes per episode, 11 in 126 s for ep 68; every other episode has 1), and 5 of 7 start
  with the hand closed. They are only 7 episodes but 35 % of the training frames. One cycle takes about 10 s, so the
  operator was not moving slowly; the episodes are just not cut per attempt, so mid-task states such as "back at
  ready, bottle on the table again" follow a handover instead of an episode start.
  The old 43 episodes run init pose (standing) → reach → pick → handover posture and end there; only the 10-19
  recordings go back to a ready pose between attempts, so s09's good return to ready can only come from them.
- **Not isolated:** the model also has fewer single-attempt episodes than the 94-episode one (66 against 94), so
  two things changed at once.
- Session and export: `/mnt/data/jihun/g1_eval_sessions/s09_wbt_groot_handover261019_he_handover` (`raw/`,
  `dataset/`); VLM failure labels and the human review are not done yet.

## Real robot, 105-episode GR00T he with the 10-19 attempts cut (2026-10-20)

`wbt_groot_handover261020_he_full`: same recipe, on `handover_bottle_merged_261020`, where the user cut the 10-19
recordings into 39 single attempts (hand-marked exclusions; the old 43 and the 25 10-14 segments are unchanged):
105 training episodes, 58,054 frames, held out 27/35. Same server and settings as s08, 30 s cap.

| Session | Model                              | Trials | Success | Failures                                                                       |
| ------- | ---------------------------------- | ------ | ------- | ------------------------------------------------------------------------------ |
| s10     | `wbt_groot_handover261020_he_full` | 10     | 8/10    | stalled, hand did not open (trial 3); wavy arm after the lift (trial 9, other) |

- **8/10, against 4/10 before the cut (s09) and 9/10 for the 94-episode model (s08).** Cutting the attempts fixed
  the reach: the grasp comes at a median 5.9 s (s09 11.9 s, s08 4.8 s) and the arm is still 33 % of the time (s09
  52 %, s08 18 %).
- **Slower than s08:** successes took 13–28 s (median 17), against 11–16 s; trials 1 and 2 needed 27–28 s.
- **Trial 3 (index 2):** grasped, then stood still for 15 s with the bottle and did not open the hand.
- **Trial 9 (index 8), corrected success → fail at review:** after the grasp and lift the right arm swayed at
  1.2 Hz (elbow and wrist pitch ~18° peak to peak, shoulder roll 10°) until the 30 s cap. The policy's tokens carry the
  same 1.2 Hz motion (70 % of their first principal component in 0.7–3 Hz), so the sway comes from the policy, not the
  controller. No other trial of s08 or s10 has more than 10 % of the elbow motion in that band (this one 68 %). Fixed in
  `results.jsonl` and `trial_008.npz` on PC2 and the workstation (reason "other"; originals kept as
  `results.jsonl.orig`).
- Session and export: `/mnt/data/jihun/g1_eval_sessions/s10_wbt_groot_handover261020_he_handover` (`raw/`,
  `dataset/`); VLM failure labels and the human review are not done yet.

## Stall investigation and inference tests (2026-10-21)

Stalls (right arm still for 2 s or more during the policy phase) were analysed in s08, s09 and s10 from the session logs
and with an offline probe (the streamer's `ChunkPolicy` re-run on the logged state and camera frame).

- **Mechanism, the same in all three models:** chunks keep arriving (2.6–2.9 per s) and the token slew limiter is never
  active, but each new chunk restarts from the unchanged view and the 0.3 s blend pulls the command back. In stalls the
  sent tokens go nowhere (net / path 0.01–0.07, against 0.30–0.35 when moving), the next chunk reverses the previous
  one (cos −0.16 to −0.43, against +0.3 to +0.6), and the arm moves 0–0.2° per chunk (6–10° when moving).
- **Dead zone:** over 2,075 chunk segments the arm moves 6–10° per unit of token change below 0.06 token per chunk,
  but 35–52° above 0.1. The residual command in a stall (~0.035 per chunk) sits inside it.
- **What drives the slow prediction:** the camera image more than the state (swapping in a moving moment's image
  restores most chunk motion). Near the bottle and in the holding posture the model predicts slow motion.
- **Ruled out:** unmasked padding at episode ends (GR00T masks `action_is_pad` correctly), sampling noise (noise 1.0
  over 10 seeds gives the same small motion), and the chunk's motion sitting only in its second half.

Inference tests on the robot, all with `wbt_groot_handover261020_he_full` except s08 (30 s cap, noise 0, token step
limit 0.05):

| Session | Change                                  | Success | Stall per trial | Success time (median) | Grasp time (median) |
| ------- | --------------------------------------- | ------- | --------------- | --------------------- | ------------------- |
| s08     | 94-episode model, blend 0.3, replan 0.4 | 9/10    | 0.9 s           | 12.4 s                | 4.8 s               |
| s10     | blend 0.3, replan 0.4                   | 8/10    | 2.5 s           | 17.0 s                | 5.8 s               |
| s11     | `--replan-s 0.8 --max-chunk-age-s 1.5`  | 8/10    | 0.2 s           | 19.1 s                | 8.2 s               |
| s12     | `--replan-s 0.6 --max-chunk-age-s 1.5`  | 6/9     | 1.1 s           | 20.8 s                | 9.2 s               |
| s13     | `--chunk-anchor-s 1.0` (PR #46)         | 7/10    | 0.0 s           | 13.5 s                | 5.3 s               |
| s14     | `--chunk-anchor-s 0.5` (PR #46)         | 6/10    | 0.0 s           | 15.1 s                | 7.4 s               |

- **Longer replans** remove stalls but the chunk runs out before the next arrives (inference 0.21–0.28 s): the command
  is frozen 30 % (s11) or 7 % (s12) of the time and the arm pulses at the replan rate (stop-and-go), and the grasp gets
  slower. The first s11 attempts ended with "no valid chunk for 1 s" (0.8 s replan + inference > the 1 s default),
  and the return to rest hit the box; raise `--max-chunk-age-s` with `--replan-s`.
- **Chunk anchoring** (each new chunk continues from the current command; the offset decays with time constant τ)
  removes every stall and brings the success time close to s08, but the reach gets less accurate: in s13 all failures
  came from an inaccurate reach (weak grasp, then the bottle falls; twice the palm knocked it over), and τ 0.5 (s14)
  did not help (4 missed grasps). The observer rated s13 above s14.
- **Caveat (found in review, 2026-10-22):** s13 and s14 ran a first version of anchoring that also offset the 14 Dex3
  hand targets, which have no slew limit. The right-hand targets sent went to −1.48 / +3.18 rad (s13) and
  −1.24 / +2.67 rad (s14), against −0.72 / +1.53 rad in s10, so their weak grasps and drops may come partly from
  corrupted finger commands rather than from anchoring the token. Fixed in PR #46 (`ef31309e`: only the 64 token dims are
  anchored, from the token actually sent); a fair anchoring test needs a new session with the fixed code.

**Decision (user, 2026-10-22): the reference handover model is `wbt_groot_handover261015_he_full`** (s08): faster
and more accurate than the 261019 and 261020 models and their inference variants. Keep the s08 settings (blend 0.3,
replan 0.4); chunk anchoring stays unmerged (PR #46) and, if tried again, should be tested on this model first.

## Incidents

- **GPU 7 shared (10-02 12:17 UTC).** Session `lerobot-f0` started `groot_joint28_ho5_official_full` (40 GB) on
  GPU 7 on the user's instruction. Our XR-1 base full then ran out of memory. Our queue was paused, and the user chose
  to wait. `lerobot-f0` cancelled its chained 55-run sim batch and keeps off GPU 7 until this queue ends. The
  interrupted runs were reset (`*.oom-gpu7-shared`, `*.stopped`) and the queue restarted 10-04 14:45 UTC.
- **VPN outage (10-02 ~12:40 UTC to 10-04).** Both H100 hosts were unreachable. The Psi0 HE copy failed and was
  restarted.
- **Slow links.** Workstation → h100_174 reached ~8 MB/s in total; h100 → workstation → h100_174 dropped to
  ~1–5 MB/s per stream. Parallel streams did not raise the total. Downloading public weights directly on h100_174 was
  5× faster.
- XFS reports inflated `du` sizes for files being written (speculative preallocation), so compare sizes only after
  a copy ends.

## Next steps

1. Done 2026-10-06: held-out open-loop comparison (section above).
2. Done 2026-10-06: `sonic_policy_streamer.py --policy-fps 50` (live camera mode; dataset mode already uses the
   dataset's rate), `POLICY_FPS=50` in `g1_groot_real_run.sh`. The PC2 streamer copy must be updated before use.
3. Done 2026-10-06 for the first two: `/mnt/data/jihun/g1_models/wbt_groot_he_full/pretrained_model` and
   `wbt_pi05_he_full/pretrained_model_ema` (config `dtype` set to bfloat16, original `config.json.fp32`). On the
   workstation RTX 3060 they reproduce the h100_174 open-loop numbers (GR00T he temp:0 tokens err 0.315, Pi0.5 he
   0.284). Psi0 and XR-1 servers would need the `feat/g1-psi0-xiaomi-policies` code.
4. Sim done for GR00T he and Pi0.5 he (section above). Robot next, with the task text "pick drink bottle from the
   table and handover" and `POLICY_FPS=50`; GR00T he first.
5. Done 2026-10-06: `training_state/` of all 8 runs deleted (181 GB; 365 GB free afterwards). Final weights kept.
