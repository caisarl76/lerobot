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

- **GR00T he works on the robot:** it reaches the bottle, picks it up and goes to the handover posture. Its smoothness
  matches the HE GR00T with noise 0 + blend (runs 24–27, jerk p95 214–276).
- **Pi0.5 he made an abrupt first move.** The right hand starts closed on the robot, in the planner stance and
  therefore at episode start, but open in all 43 recordings. Pi0.5's second chunk (~0.3 s) snapped the right hand open
  by 1.67 rad in one tick and moved the arm hard, so the watchdog ended the episode. The recordings themselves contain
  instant hand switches (up to 1.78 rad per frame, teleop trigger). GR00T he kept the hand closed at first and then
  performed normally.
- After the watchdog stop, the streamer exited with `terminate called without an active exception`. The normal exits
  of runs 33–35 don't show it. Harmless here (the robot was already back in planner mode); still to fix.

Check progress:

```bash
ssh h100_174 'cd /mnt/data01/jhkim/model_weight/g1_dex3_20260922/g1_wbt_handover_20261002; \
  for f in logs/*.exit; do echo "$f=$(cat $f)"; done; grep -v "^#" queue_gpu7.txt'
```

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
