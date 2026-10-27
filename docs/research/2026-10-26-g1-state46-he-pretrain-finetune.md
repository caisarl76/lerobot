# G1 46-D state: HE pretrain and handover fine-tune of every VLA (2026-10-21 → 10-26)

Every VLA was retrained with the official 46-D SONIC state instead of the 28-D joint state:

1. A Humanoid Everyday (HE) pretrain.
2. A fine-tune on the merged handover set, initialized from that pretrain.

All 23 runs finished with exit 0 and are uploaded to private `mncai` repos. The handover fine-tunes are evaluated offline (2026-10-27, [below](#offline-evaluation-2026-10-27)); none has run on the robot yet.

## State layout and datasets

The 46-D state equals `stream_backends.sonic46_state` on branch `feat/g1-groot-official46`. Only `observation.state` changes; the action stays the 78-D SONIC token action.

| Slice | Content                                                                                         |
| ----- | ----------------------------------------------------------------------------------------------- |
| 0:12  | legs                                                                                            |
| 12:15 | waist yaw, roll, pitch                                                                          |
| 15:29 | arms, motor order                                                                               |
| 29:43 | hands, model order index0, index1, middle0, middle1, thumb0..2 (Dex3 `hand_q[[3,4,5,6,0,1,2]]`) |
| 43:46 | projected gravity                                                                               |

The datasets were built by `examples/g1_dex3_training/prepare_state46.py` (PR #44). Provenance is in `meta/state46_provenance.json`.

- **HE:** `humanoid_everyday_g1_20260923/datasets/sonic78_nolimit_sonicstate46`, on h100 and h100_174, with the same 226 held-out episodes (`ho5`).
  - Arms and hands come from the relabelled `sonicstate` set; legs and waist from raw HE.
  - Gravity is estimated from leg forward kinematics with both soles flat. Checked against the real IMU on the handover data, the error is p50 0.78° / p95 1.24°, against 3.5° / 7.6° for a constant [0, 0, −1].
  - The HE hand order is an assumption. See [the caveat](./2026-09-23-sonic-roundtrip-audit.md#caveat-dex3-finger-order-in-humanoid-everyday-unverified-2026-10-21).
- **Handover:** `handover_bottle_merged_sonic78_50hz_261020_state46`.
  - Locations: h100 `g1_wbt_handover_20261002/datasets`, h100_174 `g1_wbt_handover_261020/datasets`, workstation `G1_WBT_GR00T`.
  - It uses the raw 43-D state plus the recorded gravity.
  - All fine-tunes exclude episodes 27 and 35.

## Runs

Every run trained on h100_174, GPUs 4 and 5. Samples = micro-batch × steps; LeRobot steps and schedules per micro-batch, and accumulation only groups micro-batches into updates.

Loss is the first and the last logged total loss of the run. Loss scales differ between policies, so compare a policy's own pretrain and fine-tune only.

### HE pretrains (`humanoid_everyday_g1_20260923`, repos `mncai/g1-he-<run>`)

| Run                                               | Micro-batch × steps (accum) | Samples | Wall time | Loss first → last |
| ------------------------------------------------- | --------------------------- | ------- | --------- | ----------------- |
| `groot_sonic78sonicstate46_ho5_official_full`     | 32 × 20,000 (1)             | 640K    | 3 h 27    | 1.205 → 0.052     |
| `pi05_sonic78sonicstate46_ho5_official_full`      | 8 × 120,000 (4)             | 960K    | 14 h 51   | 1.254 → 0.036     |
| `vla_jepa_sonic78sonicstate46_ho5_official_full`  | 8 × 120,000 (4)             | 960K    | 19 h 37   | 1.285 → 0.146     |
| `molmoact2_sonic78sonicstate46_ho5_official_full` | 8 × 100,000 (2)             | 800K    | 26 h 29   | 1.177 → 0.035     |
| `fastwam_sonic78sonicstate46_ho5_official_full`   | 8 × 60,000 (2)              | 480K    | 17 h 14   | 2.421 → 0.282     |
| `xvla_sonic78sonicstate46_ho5_official_full`      | 64 × 200,000 (4)            | 12.8M   | 17 h 36   | 0.759 → 0.016     |
| `dm05_sonic78sonicstate46_ho5_official_full`      | 32 × 100,000 (2)            | 3.2M    | 64 h 09   | 1.393 → 0.081     |
| `xr1_sonic78sonicstate46_ho5_official_full`       | 16 × 30,000 (3)             | 480K    | 9 h 37    | 8.219 → 1.530     |
| `psi0_sonic78sonicstate46_ho5_matched640k`        | 16 × 40,000 (8)             | 640K    | 8 h 13    | 7.787 → 1.170     |
| `diffusion_sonic78sonicstate46_ho5_official_full` | 64 × 200,000 (1)            | 12.8M   | 4 h 47    | 1.004 → 0.016     |
| `act_sonic78sonicstate46_ho5_official_full`       | 8 × 200,000 (1)             | 1.6M    | 2 h 52    | 11.466 → 0.283    |

LaWAM has no 46-D pretrain because it doesn't read the state.

### Handover fine-tunes (`g1_wbt_handover_261020`, repos `mncai/g1-wbt-handover-<run>`)

Each fine-tune starts from its policy's HE 46-D pretrain (`he46`). LaWAM is the exception: it starts from the earlier HE LaWAM checkpoint (step 199992, `he`).

The policies that already had a 28-D handover fine-tune keep that recipe: GR00T, Pi0.5, X-VLA, Psi0, XR-1 and LaWAM. The others get 240K samples, which is 32 × 7,500 updates, with their HE optimizer and scheduler type.

| Run                                          | Micro-batch × steps (accum) | Samples | Wall time | Loss first → last |
| -------------------------------------------- | --------------------------- | ------- | --------- | ----------------- |
| `groot_handover261020_state46_he46_full`     | 32 × 16,000 (1)             | 512K    | 2 h 18    | 0.146 → 0.023     |
| `pi05_handover261020_state46_he46_full`      | 4 × 60,000 (8)              | 240K    | 6 h 54    | 0.243 → 0.018     |
| `vla_jepa_handover261020_state46_he46_full`  | 8 × 30,000 (4)              | 240K    | 5 h 10    | 0.850 → 0.134     |
| `molmoact2_handover261020_state46_he46_full` | 8 × 30,000 (4)              | 240K    | 7 h 43    | 0.359 → 0.013     |
| `fastwam_handover261020_state46_he46_full`   | 8 × 30,000 (4)              | 240K    | 8 h 14    | 1.000 → 0.212     |
| `xvla_handover261020_state46_he46_full`      | 32 × 7,500 (1)              | 240K    | 0 h 28    | 0.289 → 0.020     |
| `dm05_handover261020_state46_he46_full`      | 32 × 7,500 (1)              | 240K    | 4 h 46    | 0.242 → 0.070     |
| `xr1_handover261020_state46_he46_full`       | 16 × 7,500 (3)              | 120K    | 2 h 26    | 2.790 → 1.134     |
| `psi0_handover261020_state46_he46_full`      | 16 × 10,000 (8)             | 160K    | 1 h 49    | 6.299 → 1.144     |
| `lawam_handover261020_state46_he_full`       | 32 × 25,000 (1)             | 800K    | 4 h 22    | 0.409 → 0.025     |
| `diffusion_handover261020_state46_he46_full` | 64 × 3,750 (1)              | 240K    | 0 h 09    | 0.224 → 0.025     |
| `act_handover261020_state46_he46_full`       | 8 × 30,000 (1)              | 240K    | 0 h 27    | 0.605 → 0.224     |

### Schedules for the new fine-tunes

| Policy    | Schedule                                                                               |
| --------- | -------------------------------------------------------------------------------------- |
| VLA-JEPA  | warmup 2,000, decay 30,000 (HE optimizer: lr 3e-5; Qwen 1e-5, action 1e-4, video 5e-4) |
| MolmoAct2 | warmup 2,000, decay 30,000                                                             |
| FastWAM   | no scheduler (constant lr 1e-4)                                                        |
| DM05      | warmup 500, decay 7,500                                                                |
| Diffusion | default cosine, 500 warmup                                                             |
| ACT       | constant lr                                                                            |

## Offline evaluation (2026-10-27)

`openloop_smooth.py` on held-out episodes 27 and 35, replanning every 20 frames (0.4 s at 50 Hz), with zero sampling noise (`temp:0`), as in the [10-06 comparison](./2026-10-02-g1-wbt-handover-training.md#held-out-open-loop-comparison-2026-10-06). Run on h100_174 GPU 4, each model downloaded from the Hub into its training code tree; Pi0.5 and Diffusion use their EMA weights. Std units of the dataset's action; err = mean |executed − recorded|, seam = jump at the chunk switch. The recording's step is 0.019 (tokens) and 0.009 (hands).

The 28-D comparators are the merged_261020 models on `handover_bottle_merged_sonic78_50hz_261020`, the same episodes without the 46-D state.

| Model                    | Tokens err | Tokens seam | Hands err | Hands seam |
| ------------------------ | ---------- | ----------- | --------- | ---------- |
| **state46 fine-tunes**   |            |             |           |            |
| FastWAM                  | **0.261**  | 0.164       | 0.088     | 0.147      |
| VLA-JEPA (replan 7)      | 0.265      | **0.106**   | 0.069     | 0.095      |
| Psi0                     | 0.278      | 0.163       | 0.097     | 0.065      |
| Pi0.5                    | 0.279      | 0.170       | 0.069     | 0.130      |
| XR-1                     | 0.279      | 0.164       | 0.195     | 0.144      |
| MolmoAct2 (random noise) | 0.280      | 0.192       | 0.085     | **0.059**  |
| Diffusion                | 0.285      | 0.204       | 0.126     | 0.117      |
| ACT                      | 0.290      | 0.170       | 0.120     | 0.066      |
| DM05                     | 0.293      | 0.198       | 0.235     | 0.212      |
| GR00T                    | 0.297      | 0.167       | **0.032** | 0.083      |
| X-VLA                    | 0.303      | 0.196       | 0.096     | 0.141      |
| LaWAM                    | 0.375      | 0.286       | 0.112     | 0.111      |
| **28-D comparators**     |            |             |           |            |
| GR00T he (robot s10)     | 0.298      | 0.172       | 0.070     | 0.146      |
| GR00T base               | 0.320      | 0.185       | 0.075     | 0.148      |
| LaWAM he                 | 0.369      | 0.271       | 0.115     | 0.136      |

- The 46-D state doesn't change the token error: GR00T 0.297 vs 0.298 (28-D he), LaWAM 0.375 vs 0.369.
- GR00T's hands improve with it: err 0.032 vs 0.070 and seam 0.083 vs 0.146. The 28-D GR00T he is the s10 model (8/10 on the robot).
- All models but LaWAM fall within 0.261–0.303 on tokens. With 2 held-out episodes, differences of a few hundredths are within the noise; robot or sim runs should decide.
- LaWAM is the worst on tokens with either state. DM05 and XR-1 have the largest hand errors (XR-1 predicts hands relative to the measured state, as in 10-06).
- VLA-JEPA replans every 7 frames, its chunk length, so its seam isn't comparable and it has no overlap.
- MolmoAct2 at zero noise predicts the same action for every frame of a chunk (step and curvature 0.000, token err 0.307), so it is shown with random noise.
- GR00T he vs base keeps the 10-06 gap (0.298 vs 0.320 here, 0.315 vs 0.332 then), on a different dataset.

Eval pitfalls:

- **Stale eval script.** The `lerobot-g1-psi0-xr1-dev` tree on h100_174 has an old `openloop_smooth.py`: it ignores `REPLAN` and `temp:`, has no err, and doesn't skip zero-std dimensions (XR-1 hands came out at ~104 std). The eval used the `lerobot-g1-newvla-261014` copies of `openloop_smooth.py` (identical to main) and `sonic_policy_streamer.py` with each tree's policy code.
- **FastWAM on CPU.** `ChunkPolicy` loads every policy on the CPU first and then moves it. FastWAM fixes its device when it is built, so the UMT5 token ids stay on the CPU and the eval fails with a device mismatch. Load FastWAM directly on the GPU.
- **VLA-JEPA.** The eval copy's `config.json` sets `reinit_modules` to null, which avoids the strict-load failure above.
- **Hub access.** The repos are private under `mncai`; a fine-grained token scoped to the user only gets 404 on them.

Outputs: h100_174 `~/state46_ft/eval/results/` (runner `eval_state46.sh`, `models.tsv`).

## Per-policy 46-D changes

Without these settings the state is silently cut off or misread:

- **Pi0.5:** `tokenizer_max_length` 256, because the state goes into the prompt.
- **X-VLA:** `max_state_dim` 46.
- **FastWAM:** `proprio_dim` 46.
- **VLA-JEPA:** `state_dim` 46.
- **DM05:** `max_state_dim` 46.
- **LaWAM:** `flow_state_dim` 46.
- **Psi0:** `model_state_dim` 46 with `state_slots` 0..45. Its observation projection is re-initialized.
- **XR-1:** new action stats and relative hand indices by joint name.
  - HE uses `/run-output/psi0_xr1/stats/he_sonic78state46.json`.
  - Handover uses `g1_wbt_handover_261020/stats/xr1_state46.json`.
  - The handover relative hand indices are `[33,34,35,31,32,29,30, 40,41,42,36,37,38,39]`. HE's list doesn't fit the handover data: the handover 28-D hand was selected by name, so its index order differs from HE's, and HE's indices are off by up to 1.42 rad.
- **GR00T, ACT, Diffusion, MolmoAct2:** only the state shape changes.

## Pitfalls hit

- **VLA-JEPA strict load.** Qwen's `embed_tokens` and `lm_head` are tied, and the saved checkpoint keeps only `lm_head.weight`. The strict check in `modeling_vla_jepa._load_as_safetensor` then reports `embed_tokens.weight` as missing.
  - The fine-tune init was fixed by adding `embed_tokens.weight` as a clone of `lm_head.weight`.
  - Any VLA-JEPA checkpoint saved by this code, including the Hub HE46 and handover ones, needs the same fix before a strict load.
- **XR-1 init from another dataset.** Patch the init copy's `config.json` (`relative_action_state_indices`, `stats_path`) before retargeting. Otherwise `load_stats` rejects the new stats.
- **Offline HF cache.** A pinned-revision download writes neither `refs/main` nor `trees/<sha>.json`. huggingface_hub 1.30 needs both offline: `refs/main` for unpinned repo ids, and the trees listing for `snapshot_download` with `ignore_patterns`.
  - Download without a revision, or copy `refs/` and `trees/` from a working cache.
- **GPU sharing.** Pi0.5 at micro-batch 8 needs about 75 GB, so it ran out of memory next to a 4.5 GB service on the same GPU. It ran alone on GPU 5 instead (80.9 of 81.5 GB).
- **Queue runner.** `official_queue.sh` exits when no job is pending, and GPU 5 sat idle for 5 h because of it. Add the next job before a queue empties, or restart the runner.
- **Housekeeping race.** The init folder exists before its copy finishes. The uploader now deletes an HE pretrain's weights only when the init `model.safetensors` exists and has the same size as the source.

## Storage

Each run went through the same steps:

1. Upload the final `pretrained_model` (and `ema/`) with `model_registry.py upload`.
2. Verify every file on the Hub by size and sha256.
3. Delete the host copy of the final weights, `training_state` and the init copy.

On 10-26 the intermediate checkpoints and the HE smoke runs were also deleted. Only the final weights exist, on the Hub. The run folders on h100_174 keep only configs, logs and small processor files. The [model registry](./g1-model-registry.md) lists every run with its repo.

## Next

- Robot evaluation against the 28-D models, starting with GR00T 46-D (the hand gain above). Use the same handover protocol as the reference model in [`2026-10-15-g1-handover-261014-out-of-view.md`](./2026-10-15-g1-handover-261014-out-of-view.md).
- A 46-D deploy path for each policy. Only GR00T has one so far (`server46` on `feat/g1-groot-official46`). Loading the VLA-JEPA models needs the tied-key fix above.
- Settle the HE right-hand finger order (procedure in [the caveat](./2026-09-23-sonic-roundtrip-audit.md#caveat-dex3-finger-order-in-humanoid-everyday-unverified-2026-10-21)). Exactly one of the two HE sets is wrong, so fix only that one:
  - **Raw HE `hand_joints` are Dex3 `hand_q` slots** (the 46-D assumption): the 46-D mapping is correct. The 28-D converter's right-hand swap (`raw[[0,1,2,5,6,3,4]]`) is wrong, so the 28-D/78-D HE models have the right index and middle pairs swapped.
  - **Raw HE stores the right hand in URDF order** (thumb, index, middle): the 28-D swap is correct. The 46-D right hand (46-D[36:43]) has its index and middle pairs swapped.
  - Thumbs are unaffected either way.
