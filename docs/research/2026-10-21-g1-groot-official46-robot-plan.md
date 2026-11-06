# G1 handover: official GR00T 46-D vs ours, robot session plan (2026-10-21)

Goal: compare on the robot three GR00T N1.7 models trained on the same 105 handover episodes
(`handover_bottle_merged_261020`, episodes 27 and 35 held out), plus NVIDIA's own inference client as a cross-check.

| Model                     | Init                   | State                                             | Recipe                                            | Workstation path                                                                |
| ------------------------- | ---------------------- | ------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------- |
| s10 (ours, 8/10 on 10-20) | HE-pretrained          | 28-D arms + hands                                 | LeRobot, `new_embodiment`, 16k × 32               | `/mnt/data/jihun/g1_models/wbt_groot_handover261020_he_full/pretrained_model`   |
| no-HE control             | `nvidia/GR00T-N1.7-3B` | 28-D                                              | same as s10, only the init differs                | `/mnt/data/jihun/g1_models/wbt_groot_handover261020_base_full/pretrained_model` |
| official 46-D             | `nvidia/GR00T-N1.7-3B` | 46-D: legs, waist, arms, hands, projected gravity | Isaac-GR00T 2a08eec, `UNITREE_G1_SONIC`, 20k × 32 | `/mnt/data/jihun/g1_models/wbt_groot_official46_handover261020_base_20k`        |

The three robot comparisons:

- **s10 vs no-HE** measures what HE pretraining adds on this dataset.
- **no-HE vs official** measures what NVIDIA's setup adds with the init held constant: the 46-D state, Isaac-GR00T's image
  geometry (no letterbox), teleop hand targets instead of `action.wbc`, and 20k instead of 16k steps. The embodiment tag is
  not a difference: `UNITREE_G1_SONIC` (slot 11) and `new_embodiment` (slot 10) are both untrained in
  `nvidia/GR00T-N1.7-3B` (std 0.0200 = the init in all three per-embodiment layers; only slots 24-27 are pretrained).

## Robot results (2026-11-05)

One day, launcher defaults (chunk anchoring 1.0, end on handover, ready pose), 10 trials each, one attempt per trial.
A fourth model joined: `groot_handover261020_state46_he46_full` (HE 46-D pretrain, 46-D fine-tune, LeRobot recipe),
served by `sonic_policy_server.py`, which reports `state_dim` from the policy config.

| Session | Model                     | Score | Median success time | Failures                              |
| ------- | ------------------------- | ----- | ------------------- | ------------------------------------- |
| s25     | official 46-D (base init) | 10/10 | 7.8 s               | -                                     |
| s26     | no-HE 28-D (base init)    | 3/10  | 16.3 s              | 5 stalled, 1 no reach, 1 grasp missed |
| s27     | s10 model, HE 28-D        | 7/10  | 9.1 s               | 1 dropped, 1 no reach, 1 grasp missed |
| s28     | HE 46-D, LeRobot recipe   | 6/10  | 7.0 s               | 4 stalled                             |

Measured on the right arm, s25 stood still 19 % of the episode (longest pause 1.5 s); s28 29 % (5.3 s), s26 46 %.
The 46-D state alone does not explain s25 (s28 has it). Both recipes apply Isaac's train-time crop 0.95 and colour
jitter; the official run used `letter_box_transform: false` (640x480 -> 341x256), LeRobot's default is true
(padded square, the scene in about 256x192). A run with only that and 20k steps changed
(`groot_handover261020_state46_he46_nolb20k_full`, h100_174) is s31 below.

## Robot results (2026-11-06): HE at 50 Hz, both recipes

Same streamer settings as s25–s28 (checked in each trial's saved arguments). The HE 46-D GR00T pretrain was redone on
HE resampled to 50 Hz (`resample_50hz.py`, `groot_sonic78sonicstate46_50hz_ho5_official_full`). s29 starts the
official recipe from it: `lerobot_to_isaac.py` converts the LeRobot checkpoint to an Isaac base model and copies the
`new_embodiment` slot (10) into `UNITREE_G1_SONIC` (11). The Isaac 20-step smoke started at loss 0.18 from it against
1.25 from NVIDIA's base, so the HE head carried over.

| Session | Model                                       | Score | Median success | Failures               | Arm still | Longest pause, median / max |
| ------- | ------------------------------------------- | ----- | -------------- | ---------------------- | --------- | --------------------------- |
| s25     | official recipe, NVIDIA base                | 10/10 | 7.8 s          | -                      | 19 %      | 0.6 / 1.5 s                 |
| s29     | official recipe, 50 Hz HE pretrain          | 10/10 | 6.9 s          | -                      | 17 %      | 0.5 / 0.7 s                 |
| s30     | LeRobot recipe (s28's), 50 Hz HE pretrain   | 8/10  | 8.0 s          | dropped, did not reach | 20 %      | 0.6 / 1.5 s                 |
| s31     | s28 + letterbox off + 20k steps (30 fps HE) | 8/10  | 7.5 s          | dropped, grasp missed  | 21 %      | 0.6 / 1.0 s                 |
| s28     | LeRobot recipe, 30 fps HE pretrain          | 6/10  | 7.0 s          | 4 stalled              | 29 %      | 0.8 / 5.3 s                 |

- The 50 Hz pretrain (s30) and the letterbox-off 20k run (s31) each removed s28's stalls; both still fail on the grasp.
- s29 and s30 share the init, so the remaining gap is in the fine-tune recipe.
- s29 ties s25 and moves faster and more evenly; with s25 at 10/10, 10 trials cannot show a higher score.
- The end-on-handover detector closed 10/10 trials for s25, 9/10 for s31, 7/10 for s29, 5/10 for s30 and 3/10 for s28;
  the operator stopped the rest and judged them.

## Official vs LeRobot recipe (checked 2026-11-06)

The official recipe (s25, s29): Isaac-GR00T 2a08eec `gr00t.experiment.launch_finetune` on one H100, base
`nvidia/GR00T-N1.7-3B`, Cosmos-Reason2-2B pinned to 9ce19a19 (offline launcher `launch_finetune_pinned.py`):

```
--embodiment-tag UNITREE_G1_SONIC --num-gpus 1 --max-steps 20000 --save-steps 1000 --save-total-limit 5
--global-batch-size 32 --gradient-accumulation-steps 1 --learning-rate 0.0001 --weight-decay 0.00001
--warmup-ratio 0.05 --state-dropout-prob 0.2 --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
--dataloader-num-workers 4 --episode-sampling-rate 0.1 --shard-size 1024 --num-shards-per-epoch 100000
```

It trains on a filtered copy of the recorded exporter dataset without the held-out episodes 27 and 35 (105 episodes;
a symlinked copy would leak them into the statistics, which are computed over every parquet). Evidence and data:
`*/evidence/{run_training.sh,launch_finetune_pinned.py}` and `*/dataset` in the run directories under
`/mnt/data01/jhkim/gr00t_runs/` (h100, h100_174). The LeRobot recipe is s30's saved `train_config.json` and
processors. Compared field by field and on the data:

|                 | Official (s25, s29)                                                                                                                                 | LeRobot (s28, s30)                                                                                                                                         |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Image resize    | aspect kept, 640x480 -> 341x256                                                                                                                     | letterboxed (`letter_box_transform true`)                                                                                                                  |
| Hand targets    | `teleop.*_hand_joints` as recorded                                                                                                                  | `action.wbc` hands: about 2 frames later, and the WBC changes them (right thumb_2 and middle differ from teleop; 0.017–0.02 rad mean on the moving joints) |
| Left-hand slots | as the deploy expects (below)                                                                                                                       | index_0 / middle_0 swapped (below)                                                                                                                         |
| Training video  | original per-episode H.264 (~1.9 Mbit/s)                                                                                                            | AV1 re-encode in LeRobot's merged files (~7.3 Mbit/s): one more lossy generation                                                                           |
| Sampling        | Isaac shard loader (`episode_sampling_rate 0.1`, shards of 1024); `allow_padding false`, so no sample whose 40-step chunk runs past the episode end | uniform random frames; chunks near the end padded                                                                                                          |
| Steps / warmup  | 20k / 1000                                                                                                                                          | 16k / 800                                                                                                                                                  |
| Statistics      | the 105 training episodes                                                                                                                           | dataset `stats.json`                                                                                                                                       |
| Seed, attention | 42, flash attention                                                                                                                                 | 1000, SDPA (backbone frozen)                                                                                                                               |

The same in both: frozen LLM and vision tower; projector, diffusion head and VL layer norm trained; AdamW lr 1e-4,
weight decay 1e-5, cosine, grad clip 1.0, bf16, batch 32, 40-step chunks; state dropout 0.2, percentile normalization
with outlier clipping, crop 0.95, colour jitter, formalized language. The SONIC tokens and the 46-D state are
identical in both datasets. The embodiment slot (11 vs 10) has no effect of its own: both are untrained in the base,
and s29's init carries slot 10 into slot 11.

**Left-hand slot order.** The streamer sends hand targets in the deploy's pose-message fields `left_hand_joints` /
`right_hand_joints`, the fields the teleop fills, so the deploy expects teleop slot order. Matched against the
exporter's labelled `action.wbc` (lag 2 frames), teleop is thumb 0–2, index 0–1, middle 0–1 for both hands: left slot 3
matches `index_0` (mean error 0.017 vs 0.045 for `middle_0`), slot 5 `middle_0` (0.015 vs 0.045); the `_1` slots are
identical in teleop. The official model learns teleop slots and `gr00t_official_server.py` passes them through:
correct. Our sonic78 conversion orders the left hand as `sonic_targets.ACTION_NAMES` (thumb, middle, index), so every
LeRobot model trained on it sends left `index_0` to the `middle_0` slot and back. The handover uses the right hand,
whose order matches, so the effect on these results is small; fix the conversion before tasks that use the left hand.

## Candidates queued (h100_174 GPU 4, after VLA-JEPA 261015, 2026-11-06)

Both: s30's chain (50 Hz HE pretrain, LeRobot) with letterbox off, 20k steps and the original H.264 videos
(`official_match_dataset.py`, statistics over the 105 training episodes). Waiter `~/state46_ft/match/match.sh`.

- `groot_handover261020_state46_he46hz50_match`: teleop hand targets in recorded slot order. At 10/10 the gap to s29
  is in the data and preprocessing; at about 8/10 it is in Isaac's loader (sampling, no padded chunks).
- `groot_handover261020_state46_he46hz50_h264`: `action.wbc` hand targets as before; against `match` it isolates the
  hand targets.

## Before the session

1. **No-HE copy.** It is still running detached. Its log `/mnt/data/jihun/g1_models/wbt_groot_handover261020_base_full/copy_from_h100.log` must end in `SHA MATCH`. If it does not, rerun `copy_from_h100.sh` in the same directory; it streams through the h100 container `jihun-lerobot-wbt-gpu7-261021`.
2. **Code.** Everything is on branch `feat/g1-groot-official46`:
   - `gr00t_official_server.py`;
   - `sonic46_state` in `stream_backends.py`;
   - `state_dim` handling and the `state46` log key in the streamer;
   - the `server46` launcher mode;
   - `official46_offline_check.py`.

   Merge it, or run from it, then copy the streamer files to PC2 with `g1_groot_real_run.sh sync`. The streamer stays compatible with 28-D servers (`state_dim` defaults to 28).

3. **Workstation environments.**
   - Official model: `/mnt/data/jihun/venvs/isaac-gr00t-2a08eec` with source `/mnt/data/jihun/code/isaac-gr00t-2a08eec`. These are the `server46` defaults.
   - LeRobot models (`server`): `uv run --project ../..` from a checkout that has a synced env. Running `server` from a worktree without one makes uv build a partial `.venv` there (one was made and deleted on 2026-10-21).
4. **GPU, one model at a time.** GPU 1 (RTX 3060, 12 GB) holds one model:
   - LeRobot GR00T needs about 9.4 GB.
   - The official model needs about 7.4 GB at 0.13 s per chunk.
   - Another session's s10 server may still be running as `groot-server-5561`. Stop a server (`PORT=… g1_groot_real_run.sh stop`) before starting the next.
   - GPU 0 runs a sim deploy and cannot hold the LeRobot GR00T.

## Session protocol (same for every model)

- `POLICY_FPS=50 MAX_S=30 TRIALS=10`, one attempt per trial, task `pick drink bottle from the table and handover`.
- Defaults: noise 0, `--chunk-blend-s 0.3`, replan 0.4 s.
- Use the same table, bottle and placement for all models, and alternate the order if more than one day is used.
- Use the next free session numbers (s13–s15 have since been used for the chunk-anchoring sessions): official46, no-HE, then an s10 rerun the same day. Rerunning s10 controls for day effects; s11 and s12 showed run-to-run variance from the stall issue.

```bash
cd examples/g1_dex3_training
# official 46-D through our streamer
MODEL=/mnt/data/jihun/g1_models/wbt_groot_official46_handover261020_base_20k GPU=1 PORT=5562 NOISE_SCALE=0 \
  ./g1_groot_real_run.sh server46
PORT=5562 POLICY_FPS=50 MAX_S=30 TRIALS=10 ./g1_groot_real_run.sh session sNN_official46_handover \
  "pick drink bottle from the table and handover"
PORT=5562 ./g1_groot_real_run.sh stop
# no-HE control (then s10 the same way, with its MODEL)
MODEL=/mnt/data/jihun/g1_models/wbt_groot_handover261020_base_full GPU=1 PORT=5560 ./g1_groot_real_run.sh server
PORT=5560 POLICY_FPS=50 MAX_S=30 TRIALS=10 ./g1_groot_real_run.sh session sNN_nohe_handover \
  "pick drink bottle from the table and handover"
```

Run `deploy` and `camera` as usual, then `pull` each session and review it in the eval logger. The official run's trial
logs carry `state46`: one row per chunk request with the exact 46-D state sent to the model, plus `state46_wall` and `state46_t_ep` (when it was sent), for checks against the training distribution.

## NVIDIA's own client (cross-check, about 5 trials, after the three sessions)

Purpose: if the official model scores much worse through our streamer, this run tells whether our adapter or
blending is at fault or the model itself.

- **Server, workstation.** Run the stock Isaac-GR00T server in the same venv:
  `PYTHONPATH=/mnt/data/jihun/code/isaac-gr00t-2a08eec HF_HUB_OFFLINE=1 python gr00t/eval/run_gr00t_server.py --model-path <official46 dir> --embodiment-tag UNITREE_G1_SONIC --port 5550`.
  Offline it hits the same Cosmos-processor Hub lookup that `gr00t_official_server.py` patches (`build_local_processor`). Either apply that patch or start it once online.
- **Client, PC2.** Run `gear_sonic/scripts/launch_inference.py` from GR00T-WholeBodyControl branch `work/g1-handover-loop-20261020`. It contains fix 5e35112, which preserves the measured VLA hand observations; the local `vr3pt-cleanup-minimal-official` branch still overwrites the left-hand state. Options: `--policy-host 192.168.0.62 --policy-port 5550 --prompt "pick drink bottle from the table and handover"`, `action_horizon` 40 and publish rate 50 (defaults). It starts the deploy itself and uses NVIDIA's camera server on port 5555.
- **Unverified on PC2:** that the branch and `.venv_inference` exist (`install_scripts/install_inference.sh`), the camera server and device, the `initial_pose` choice (`planner_standing` is closest to our `--start planner`), and the deploy options it passes.
- **Conflict.** Stop our `g1_head_camera_server` and our deploy first; both clients want the same ports.
- **Controls** (keyboard pane): `k` starts the control loop, `i` returns to standing, `p` starts/pauses the policy. No eval logger, so note verdicts by hand. Its data exporter can record episodes (`c`/`s`/`f`).
- **Differences from ours:** no chunk blending (it jumps to the latency-compensated index of each new chunk), unscaled sampling noise, no streamer watchdogs.

## Offline results so far (workstation GPU 0, 2026-10-21)

- **State.** `sonic46_state` rebuilds the recorded 46-D state of all 107 episodes exactly (max error 1e-16). The hand order follows NVIDIA's robot model: Dex3 `hand_q[[3,4,5,6,0,1,2]]` for both hands.
- **Open loop.** Token MAE of 40-step chunks every 0.4 s through the adapter. s10 was measured through its server. "Hold" repeats the current token.

| Episode       | official 46-D | s10    | hold   |
| ------------- | ------------- | ------ | ------ |
| 27 (held out) | 0.0240        | 0.0237 | 0.0220 |
| 35 (held out) | 0.0317        | 0.0290 | 0.0302 |
| 0 (train)     | 0.0094        | 0.0083 | 0.0177 |
| 50 (train)    | 0.0096        | 0.0094 | 0.0218 |

Both models are close to "hold" on held-out episodes, so this metric cannot rank them; the robot has to.

## Related work and open items

- HE pretraining of every VLA with the 46-D state runs on h100_174 GPUs 4 and 5, managed by another session
  (memory note `g1-he-state46-pretrain.md`). Its 46-D handover fine-tunes are the next models for this comparison.
- The official and no-HE models are on the Hub (`mncai`, 2026-10-27), as are the 2026-11-05 runs (nolb20k, the 50 Hz
  HE pretrain and its LeRobot fine-tune, and s29's `mncai/g1-wbt-handover-groot-official46-handover261020-he50-20k`).
- Finger order: confirmed for the hand actions on 2026-11-06 (left index_0 / middle_0, see "Left-hand slot order").
  The WBT 28-D state names the left pairs the same way; the HE 28-D right hand has the mirror case. Check both
  before training models for tasks that use the left hand.
