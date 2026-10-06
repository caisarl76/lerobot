# Psi0 and Xiaomi-Robotics-1 as native LeRobot policies for the G1 Dex3 pipeline (2026-09-30)

Design note and running status for adding **Psi0** (USC PSI Lab) and **Xiaomi-Robotics-1** (XR-1) to the G1
training pipeline next to GR00T, Pi0.5, ACT and the others retrained in
[`2026-09-30-official-retraining-handover.md`](./2026-09-30-official-retraining-handover.md). Both are trained on
both datasets (Humanoid Everyday "HE", Unitree) and both action spaces:

- **28D**: 14 arm joints + 7 left + 7 right Dex3 joints, the `joint28` datasets.
- **78D**: 64 SONIC v1.1 tokens + 14 Dex3 joints, the `sonic78_nolimit_sonicstate` datasets (relabelled state).

This repository is public: addresses, user names and credentials are left out on purpose.

## Status (keep up to date)

| Item                                                            | State (2026-09-30 10:05 UTC)                                                                             |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Native policies `psi0`, `xiaomi_robotics`; optimizer `adamw_sr` | Implemented; 21 new tests pass, pre-commit clean                                                         |
| Configs for the 8 combinations (`*_ho5_official_{smoke,full}`)  | On the H100 (`write_psi0_xr1_configs.py`); Psi0 28D now on the SONIC v1.1 checkpoint (user decision)     |
| Smoke runs + load-and-predict + open-loop checks                | All 8 passed (06:50); Psi0 28D re-smoked on the new checkpoint: HE passed, Unitree after the data fix    |
| Unitree corrupt-state fix (`fix_unitree_corrupt_state.py`)      | Dry run and apply test on a private copy done. **Not applied: blocked by the permission system** (below) |
| HE full runs                                                    | Running on GPU 2 (queue `/run-output/psi0_xr1/queue_gpu2.txt`), see [Full runs](#full-runs)              |
| Unitree full runs                                               | Held until the data fix is applied                                                                       |

## Decisions (user, 2026-09-30)

1. Full runs approved. Of the idle GPU 6 containers only `jihun_psi0_bridge_probe_20260911_gpu6_r2` was removed;
   GPU 0 is not to be used.
2. XR-1 keeps the released 48 × 10K. 3. Psi0 78D keeps the official 128 × 40K. 6. `adamw_sr` with bf16 moments is
   accepted for XR-1.
3. **Psi0 28D uses the SONIC v1.1 checkpoint** (`postpre.sonic1.1.unifolm.2609181726.40k`) instead of the AMO-era
   checkpoint of the official real-G1 recipe: **a departure**. Same architecture and recipe as 78D (tuned VLM, 12
   layer-wise blocks, CLIP task embedding, state token, 270×480, 128 × 40K). The header's action in/out layers are in
   SONIC token space, so at width 28 they are **re-initialised**: `action_proj_in.ac_proj.{0,2}` and
   `action_proj_out.linear` (5 tensors, `action_header_load: matching`). Everything else loads: VLM, the 12 blocks,
   time + CLIP embedding, the state token projection (same 45-D G1 state layout), `state_pos`, `state_null`, `dec_pos`
   and the output adaLN. No padding (28 = model width), so the loss scale matches 78D (78 of 80 dims valid).
4. Fix the corrupt Unitree right-hand state frames in the datasets and recompute the statistics for all policies,
   only in a safe window agreed with the session that owns the official queues.

## The two models

|                        | Psi0                                                                                                                                                   | Xiaomi-Robotics-1 (XR-1)                                                                                                                                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Source                 | github.com/physical-superintelligence-lab/Psi0 (arXiv 2603.12263), checkpoints `USC-PSI-Lab/psi-model`                                                 | github.com/XiaomiRobotics/Xiaomi-Robotics-1 (arXiv 2607.15330), checkpoint `XiaomiRobotics/Xiaomi-Robotics-1-5B`                                                                                            |
| License                | Code Apache 2.0. The weights repo has no license tag (the project README badge says Apache 2.0). CLIP-L (used by the SONIC recipe): MIT                | Code and weights Apache 2.0; Qwen3-VL-4B Apache 2.0                                                                                                                                                         |
| Backbone               | Qwen3-VL-2B (pre-trained on EgoDex + HE by PSI Lab)                                                                                                    | Qwen3-VL-4B (36 layers, hidden 2560)                                                                                                                                                                        |
| Action expert          | "Action header": SD3-style MM-DiT, hidden 1536, 24 heads. Action tokens and VLM-context tokens attend jointly; flow matching (velocity noise − action) | 36-layer DiT, hidden 1024, one DiT layer per VLM layer: each DiT layer attends to that VLM layer's **key/value cache** (Mixture-of-Transformers). Flow matching (velocity action − noise), 5 Euler steps    |
| Extra heads / losses   | none                                                                                                                                                   | VLM "choice policy": 5 action hypotheses (winner-take-all L1) + predicted errors, on action query tokens in the VLM. Frequency-domain loss on the chunk. Loss = 0.5 MSE + 1.0 freq + 0.5 choice + 0.5 score |
| Params                 | 2.1B VLM + 0.5B (6-block) or 0.67B (12-block) header                                                                                                   | 4.4B VLM + 0.6B DiT = 5.1B                                                                                                                                                                                  |
| Chunk / horizon        | 30 steps, all executed                                                                                                                                 | 30 steps (60 action query tokens available)                                                                                                                                                                 |
| Released action layout | AMO era: 36-D (14 hands, 14 arms, torso, base). SONIC: 80-D = 64 SONIC v1.1 tokens + 14 Dex3 + 2 neck                                                  | 60-D: relative end-effector pose + gripper per arm, waist, base velocity; per-step mean/std                                                                                                                 |

**What differs from Xiaomi-Robotics-0** (assessed in the 2026-09-28 handover): XR-1 is a new model, not a re-release.
The action shape grew from `[30, 32]` to `[30, 60]`; the DiT reads the VLM's per-layer KV cache; the VLM gets the
choice-policy head, special `<state>`, `<a_i>` and `<score>` tokens, and a frequency loss; training is asynchronous
(a random clean action prefix of 1–6 steps half of the time, with part of the prefix masked). Its loader is still a
JSON + video format for dual-arm end-effector data, and it pins transformers 4.57.1 and flash-attn.

## Why a native port instead of the upstream trainers

Both upstream trainers need their own environment (Psi0: Python 3.11, torch 2.7, transformers 4.57.1, flash-attn;
XR-1: torch 2.8, transformers 4.57.1, DeepSpeed, Lightning, decord) and their own data format. The ports run on the
existing LeRobot image (`4cbe2a3f7fc6`: Python 3.12, torch 2.11, transformers 5.5.4, diffusers 0.39, no flash-attn),
read the LeRobot datasets directly (held-out exclusions, `LEROBOT_VIDEO_DECODER_CACHE_SIZE`, the queue runner,
`finalize_baseline.py` and the open-loop scripts all apply unchanged), and save standard LeRobot checkpoints. **No new
Docker image was needed.** Upstream module and parameter names are kept, so the released weights load as they are.

Port differences that cannot change the math: attention uses PyTorch SDPA instead of flash-attn; XR-1 batches are
right-padded instead of flash-attn "packed"; XR-1's special token embeddings are injected with a forward hook on the
token embedding; XR-1's per-layer KV cache comes from a transformers `DynamicCache`; images stay tensors instead of
PIL (resize kernels differ slightly).

## Action and state mapping

| Model | Space | Action head                                                                                                                                                                                                                      | Trained from scratch                                                                                         |
| ----- | ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Psi0  | 28D   | **Resize** (user decision): SONIC v1.1 header at width 28; state into the 45-D layout as for 78D. (Before 2026-09-30 10:00: official AMO recipe, pad to 36, only the 6 transformer blocks loaded; smoke kept as `*_smoke_amo36`) | The action in/out layers (`ac_proj`, `action_proj_out.linear`); everything else loads                        |
| Psi0  | 78D   | **Pad** to 80: tokens 0–63, hands 64–77 (same order as UnifoLM's `action[:14]`, checked: identical per-dim min/max), neck 78–79 masked. State into the 45-D layout: arms 15–28, hands 29–42, legs/waist/neck 0                   | Nothing: the whole SONIC v1.1 header and VLM load (same chunk and width)                                     |
| XR-1  | 28D   | **Pad** to 60; unused dims masked (loss) and zeroed (input), as XR-1 does for its unused slots                                                                                                                                   | Nothing; the 60-D layers keep their weights (their meaning changes from end-effector deltas to joint deltas) |
| XR-1  | 78D   | **Resize** to 78: action in/out layers and the choice head keep their 60 pretrained rows/columns per hypothesis; the 18 new ones start from XR-1's init (N(0, 0.02))                                                             | The 18 new dims of those three layers                                                                        |

- XR-1 state (60-D): left arm 0–6, right arm 8–14, both hands 16–29 (grippers 7/15 unused); XR-1's arm slots are
  7-DoF joint positions, like the G1's.
- XR-1 actions are **relative** as in the release: every action dim with a state counterpart is `a[t+k] − s[t]`
  (all 28 joints; the 14 hands in 78D); SONIC tokens stay absolute. Per-step mean/std over complete 30-step windows,
  state q01/q99, both from `xr1_action_stats.py`, stored as model buffers (saved in `model.safetensors`). The policy
  returns absolute actions; the streamer is unaffected.
- Psi0 normalization is its "bounds" (min/max → [−1, 1], clipped), as released, for 28D and 78D. **Departure:** the
  Unitree **state** uses q01/q99 (clipped), because the stored min/max are corrupt (next section).

## Data finding: corrupt Unitree state frames

The Unitree `observation.state` (in `joint28` and every `sonic78*` variant) has **266 frames in 39 episodes** with
right-hand values up to ±3363 (dims 24–27, right index/middle; `find_bad_state.py`-style scan, threshold 3.2 rad).
Episode 2202 (already `always_excluded`) is not among them; two held-out episodes (1245, 2451) are. The stored
min/max (and the mean/std) include them.

- Psi0 on Unitree: state normalized with q01/q99 and clipped, so these frames saturate instead of squashing the
  whole right-hand range.
- XR-1: `xr1_action_stats.py` skips windows whose anchor state exceeds 3.2 rad (262 of 2.38M), and the model clamps
  the state to the valid range before it anchors a relative action.
- **Other policies are affected too**: ACT/Diffusion (MEAN_STD state) see inputs of ~10⁴ std on those frames, and
  every policy's state normalization includes them. Not fixed here; see open questions.

### Fix (`examples/g1_dex3_training/fix_unitree_corrupt_state.py`)

- **Which frames.** Per state dimension a value is corrupt if it is non-finite, |value| > 3.2 rad, or — for the
  spiking right index/middle dims 24–27 — outside that joint's commanded range (action min/max of `joint28`)
  ± 0.5 rad. Result: **267 frames in 40 episodes, dims 24–27 only**, identical in `joint28`, `sonic78_nolimit`
  and `sonic78_nolimit_sonicstate` (the 3.2 rule alone finds 266 in 39; the range rule adds one −2.93 rad spike).
  The thumbs (dims 14, 21) also leave the commanded range, but in long constant stretches (e.g. episode 2449, dim 21
  stuck at 1.42 rad): a hand offset, not a glitch; left unchanged. HE: no frame beyond 3.2 rad; not touched.
- **Repair.** Linear interpolation in `frame_index` between the nearest valid values of the same dimension and
  episode (at an episode edge the nearest valid value). Nothing else changes: other dims, actions, other columns,
  videos (symlinks into the read-only `/source-datasets`).
- **Statistics.** `observation.state` in `meta/stats.json` (all present keys: min, max, mean, std, count,
  q01/q10/q50/q90/q99) recomputed over all frames; per-episode stats in `meta/episodes/chunk-000/file-000.parquet`
  recomputed for the 40 episodes. The method reproduces the stored values exactly on unchanged dims and episodes
  (max difference 0 for min/max/mean/std, ≤ 6e-6 for the `sonic78*` quantiles; `joint28`'s stored quantiles are the
  old approximate ones, off by up to 0.55, and become exact).
- **Files per dataset**: 9 data parquet files, 1 episode-meta file, `stats.json`; each backed up as
  `<file>.bak-corruptstate-20260930` and replaced atomically; a re-read verifies schema, untouched columns and
  values, and a second pass finds nothing left.
- **Effect** (right index/middle state, dims 24–27): min −1846/−225/−3363/−1920 → −0.30/−0.03/−0.40/−0.03,
  max 1683/1903/1922/771 → 1.65/2.15/1.55/2.24, std 4.05/1.86/5.56/3.27 → 0.41/0.72/0.32/0.70.
- **Dry run** on the three shared datasets (read-only) and **apply test** on a private copy of `joint28`
  (`/run-output/psi0_xr1/datafix/copy/`, 11 files backed up and replaced, all verifications pass, reloads with
  `LeRobotDatasetMetadata`): reports in `/run-output/psi0_xr1/datafix/`.
- **Not applied.** The permission system refused the step that writes to the shared datasets. The prepared steps
  are (written, not staged on the H100) an apply wrapper (apply, second pass, backups list, XR-1 Unitree stats) after the
  host-side reader check `check_no_unitree_readers.sh`, then appending the two blocks of
  `queue_official/pending_unitree_after_datafix.txt` (GPU 0 block → `gpu0.txt`, GPU 6 block → `gpu6.txt`). They
  need the user's explicit permission. `combined_sonicstate_1cam` contains the same Unitree frames and is not in scope.
- Unitree GR00T and ACT official runs were trained on the uncorrected statistics; retraining them is the user's call.

## Cameras

- HE: one egocentric view. Unitree: `cam_left_high` and `cam_right_high`, both fed (like the other policies).
- Psi0: all views go into the Qwen3-VL prompt as images followed by the instruction; resize to 240×320 (28D) or
  270×480 (78D) with nearest neighbour, as released. Unitree doubles the VLM tokens (2 × 80 or 2 × 120).
- XR-1: prompt "The following observations are captured from multiple views.\n# <view>\n<image>…Generate robot
  actions for the task:\n<task> /no_cot", assistant "<cot></cot>". XR-1 was trained with "Ego View", "Left-Wrist
  View", "Right-Wrist View"; we use "Ego View" (HE) and "Ego View (left camera)" / "Ego View (right camera)"
  (Unitree). Images resized to multiples of 32 with at most 160,000 pixels (640×480 → 448×320, 140 tokens each).

## Official fine-tuning recipes and ours

"Update" = one optimizer step. LeRobot steps the scheduler per micro-batch, so the configs multiply `steps`, warmup
and decay by the accumulation factor (pitfall in the retraining handover).

### Psi0

| Setting            | Official, 28D (`finetune-real-psi0.sh`)                                                                                                                           | Official, 78D (`finetune-real-sonic-psi0-2.8B-sonic1.1-robust.sh`)                                                                              | Ours                                                                                      |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Init               | VLM `pre.fast.1by1.2601091803.ckpt.ego200k.he30k`, header `postpre.1by1.pad36.2601131206.ckpt.he30k`. **Ours: not used (decision 4); 28D follows the 78D column** | VLM + header `postpre.sonic1.1.unifolm.2609181726.40k` (post-trained on 50 h UnifoLM G1 in SONIC v1.1 tokens)                                   | same                                                                                      |
| Batch × updates    | 16/GPU × 8 GPUs = **128 × 40K** (5.1M samples)                                                                                                                    | **128 × 40K**                                                                                                                                   | 28D: 32 × 4 accumulation; 78D: 16 × 8. Same 128 × 40K                                     |
| Optimizer          | AdamW β (0.95, 0.999), wd 1e-6, clip 1.0, lr 1e-4                                                                                                                 | same; VLM tuned: language 1e-6, vision 1e-5, merger 1e-4                                                                                        | same (78D: `foreach=False`)                                                               |
| Schedule           | cosine to 0, warmup 1K                                                                                                                                            | same                                                                                                                                            | same (LeRobot `cosine_annealing_with_warmup` = transformers "cosine")                     |
| Precision          | bf16 autocast; header fp32; frozen VLM bf16                                                                                                                       | bf16 autocast; tuned VLM fp32 weights, gradient checkpointing                                                                                   | same                                                                                      |
| VLM                | frozen                                                                                                                                                            | tuned (final norm frozen, as upstream)                                                                                                          | same                                                                                      |
| Header             | 6 blocks, last-layer VLM context, state as context token, dropout 0.1, state-feature dropout 0.2                                                                  | 12 blocks, one VLM layer per block (3…28), qk RMSNorm, CLIP-L pooled task embedding, state as action token 0 with learned null token (drop 0.1) | same                                                                                      |
| RTC                | training-time RTC, delay 0–7                                                                                                                                      | off (test-time RTC)                                                                                                                             | same                                                                                      |
| Images             | 240×320, ColorJitter(0.2, 0.8–1.2, 0.8–1.2, 0.05)                                                                                                                 | 270×480, same jitter + view crop 85–100 %                                                                                                       | same                                                                                      |
| State augmentation | none                                                                                                                                                              | noise N(0, 0.05) on the normalized state; ±10-frame temporal jitter (p 0.5)                                                                     | noise yes; **temporal jitter not implemented** (needs a 21-frame state window per sample) |
| Normalization      | bounds (min/max), state normalized                                                                                                                                | bounds                                                                                                                                          | bounds; **Unitree state q01/q99** (corrupt frames)                                        |
| Chunk padding      | not masked                                                                                                                                                        | not masked                                                                                                                                      | same (`mask_padded_actions=false`)                                                        |

### XR-1

| Setting         | Official (`xr1/configs`, 1-GPU default)                                                                                             | Ours                                                                                                                                                                                                                                                                                                                 |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Init            | `Xiaomi-Robotics-1-5B`                                                                                                              | same                                                                                                                                                                                                                                                                                                                 |
| Batch × updates | **48 × 10K** (480K samples; the config is written for a 5-episode demo set)                                                         | 16 × 3 accumulation = 48 × 10K; see the plan for a longer option                                                                                                                                                                                                                                                     |
| Trainable       | everything but the token embedding (4.72B)                                                                                          | same (4.72B)                                                                                                                                                                                                                                                                                                         |
| Optimizer       | DeepSpeed FusedAdam, β (0.9, 0.95), wd 0.1 (none on bias/norm/rotary/adaLN), clip 1.0                                               | AdamW with the same groups. **Departure:** bf16 weights and **bf16 moments with stochastic rounding** (`adamw_sr`); DeepSpeed keeps fp32 master weights and moments, which need ~75 GB for 4.7B parameters and do not fit one 80 GB GPU. Plain bf16 AdamW would drop most updates at lr 2e-5 (below half a bf16 ULP) |
| Schedule        | warmup 500 from 5e-7 to 2e-5, cosine to 5e-6 at 10K                                                                                 | LeRobot `cosine_decay_with_warmup` (warmup from ~0; cosine measured from step 0)                                                                                                                                                                                                                                     |
| Precision       | bf16 weights (`model.to(bf16)`), bf16-mixed                                                                                         | same                                                                                                                                                                                                                                                                                                                 |
| Memory          | MLP gradient checkpointing in the VLM, vision checkpointing                                                                         | same                                                                                                                                                                                                                                                                                                                 |
| Losses          | 0.5 MSE (weighted by prefix error) + freq + 0.5 choice + 0.5 score; 4 noise draws per sample; freq loss excludes base-velocity dims | same; no excluded dims                                                                                                                                                                                                                                                                                               |
| Async training  | prefix 1–6 with p 0.5; half of the prefix (but the last 2) hidden from the suffix                                                   | same                                                                                                                                                                                                                                                                                                                 |
| Images          | multiple of 32, ≤ 160K pixels; brightness ±32/255, contrast and saturation 0.5–1.5, each p 0.5, shared by views                     | same                                                                                                                                                                                                                                                                                                                 |
| Actions         | relative end-effector/gripper deltas, per-step mean/std; state q01/q99                                                              | relative joints (tokens absolute), per-step mean/std; state q01/q99                                                                                                                                                                                                                                                  |

## Dependencies and environment

- Image `4cbe2a3f7fc6` and venv `/run-output/environment/venv` as for every other policy; code mounted at
  `/workspace/lerobot`. Extras: Psi0 needs `transformers` + `diffusers` (for SD3 attention blocks); XR-1 needs
  `transformers`.
- Extra mounts: `-v /mnt/data01/jhkim/model_weight/Psi0:/psi-weights:ro` and
  `-v /mnt/data01/jhkim/model_weight/XiaomiRobotics:/xr1-weights:ro`, plus the usual `/source-datasets` mount,
  `LEROBOT_VIDEO_DECODER_CACHE_SIZE=5000` and `--memory`.
- Downloaded 2026-09-30 (in a container): `psi0/postpre.sonic1.1.unifolm.2609181726.40k` (11 GB) into the Psi0
  directory, `model_states.pt` (10 GB) into `XiaomiRobotics/Xiaomi-Robotics-1-5B/`; Qwen3-VL-4B processor/config files
  and CLIP-L into the shared HF cache.
- Checkpoints carry the Qwen processor (`pretrained_model/vlm_processor/`) and the VLM config, so they load without
  the base-weight directories. The Psi0 78D policy still needs `openai/clip-vit-large-patch14` in the HF cache.

## Risks

- **XR-1 on one GPU** runs with bf16 weights and bf16 moments (stochastic rounding), not DeepSpeed's fp32 master
  copy. Unbiased, but noisier than the official setup. With a whole free 80 GB GPU, fp32 moments at micro-batch 8
  might fit (not measured).
- **XR-1 sample count**: the released 48 × 10K is a demo setting and is 2–4× below the other policies' official
  sample counts on our data (0.3 pass over HE).
- **Psi0 28D starts from the AMO-era checkpoint** (only its transformer blocks load), as the official real-G1 recipe
  does. The SONIC v1.1 checkpoint has a better G1 VLM but a token-space action head; using it for joints would be a
  departure.
- **Min/max action normalization of SONIC tokens** (Psi0, official): the VLA-JEPA note found MIN_MAX makes token
  errors 2–3× worse for that model. Kept because it is Psi0's own recipe (its SONIC checkpoint was trained that way).
- **XR-1 prompt views**: XR-1 never saw two head cameras; the view names are our choice.
- **Relative XR-1 actions and the SONIC state gap**: relative targets are anchored on the relabelled (sonicstate) or
  measured (joint28) state. In closed loop the anchor is the robot's measured state.

## Implementation

| File                                                                                                                      | What                                                                               |
| ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `src/lerobot/policies/psi0/{configuration,modeling,processor}_psi0.py`, `action_header.py`                                | Psi0 policy; the header port keeps upstream names (released weights load strictly) |
| `src/lerobot/policies/xiaomi_robotics/{configuration,modeling,processor}_xiaomi_robotics.py`                              | XR-1 policy                                                                        |
| `src/lerobot/optim/adamw_sr.py`, `AdamWSRConfig` (`adamw_sr`)                                                             | AdamW with stochastic rounding for bf16 weights and moments                        |
| `examples/g1_dex3_training/write_psi0_xr1_configs.py`                                                                     | The 16 configs (8 combinations × smoke/full)                                       |
| `examples/g1_dex3_training/xr1_action_stats.py`                                                                           | XR-1 per-step statistics                                                           |
| `finalize_baseline.py`, `ho5_openloop_eval.py`, `openloop_smooth.py`                                                      | Accept the two policy types and 28D (arms/hands split)                             |
| `tests/policies/psi0`, `tests/policies/xiaomi_robotics`, `tests/optim/test_adamw_sr.py`, `tests/policies/tiny_qwen3vl.py` | 20 tests                                                                           |

Tests (CPU, tiny random Qwen3-VL with the real processor files):

```bash
HF_HUB_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src uv run --no-project \
  --python /path/to/lerobot/.venv/bin/python --with pytest python -m pytest \
  tests/optim/test_adamw_sr.py tests/policies/psi0 tests/policies/xiaomi_robotics
```

They cover: the flow sampler against diffusers' `FlowMatchEulerDiscreteScheduler`; both header variants with
per-sample and per-token (RTC) timesteps; padded-context masking; full vs blocks-only header loading; XR-1 checkpoint
key conversion and 60 → 78 resizing per choice block; the XR-1 losses (including the upstream edge-case tests);
relative normalization round trip; training with and without an action prefix; save → `from_pretrained(strict=True)`
→ identical actions; stochastic rounding (unbiased, equals `torch.optim.AdamW` for fp32, keeps sub-ULP updates).

## Smoke runs

GPU 2 on the H100 (container `jihun-lerobot-psi0xr1-gpu2-20260930`, code at
`/mnt/data01/jhkim/code/lerobot-g1-psi0-xr1-dev`, `--memory 200g`, `LEROBOT_VIDEO_DECODER_CACHE_SIZE=5000`, the
`/source-datasets` mount; 72–76 GB free next to another user's 3–7 GB). Psi0: 200 micro-batches; XR-1: 90. Queue
runner: `official_queue.sh` with `/run-output/psi0_xr1/queue_gpu2_smoke.txt`; logs and `.exit` files in each root's
`logs/`. Speeds are after step 30 (data loading warmed up).

| Dataset | Job                                   | Micro × accum | Trainable | Samples/s | Peak GPU memory | Loss first → last (smoke) |
| ------- | ------------------------------------- | ------------- | --------- | --------- | --------------- | ------------------------- |
| HE      | psi0_joint28 (AMO ckpt, superseded)   | 32 × 4        | 0.49B     | 138       | 14 GB           | 32.4 → 23.5               |
| HE      | psi0_joint28 (SONIC v1.1 ckpt, 10:00) | 16 × 8        | 2.80B     | 23        | 50 GB           | 39.1 → 35.6               |
| HE      | psi0_sonic78sonicstate                | 16 × 8        | 2.80B     | 25        | 50 GB           | 12.9 → 8.1                |
| HE      | xr1_joint28                           | 16 × 3        | 4.72B     | 16        | 56 GB           | 6.8 → 5.2                 |
| HE      | xr1_sonic78sonicstate                 | 16 × 3        | 4.72B     | 15        | 56 GB           | 8.8 → 7.4                 |
| Unitree | psi0_joint28 (AMO ckpt, superseded)   | 32 × 4        | 0.49B     | 90        | 15 GB           | 33.8 → 25.0               |
| Unitree | psi0_sonic78sonicstate                | 16 × 8        | 2.80B     | 20        | 53 GB           | 9.6 → 5.6                 |
| Unitree | xr1_joint28                           | 8 × 6         | 4.72B     | 8.3       | 52 GB           | 6.4 → 4.9                 |
| Unitree | xr1_sonic78sonicstate                 | 8 × 6         | 4.72B     | 9.6       | 52 GB           | 9.1 → 7.4                 |

- Weights: Psi0 SONIC v1.1 header and VLM load completely (0 missing, 0 unexpected); the AMO header loads its 6
  transformer blocks (19 keys re-initialised, as upstream); XR-1 loads strictly, resizing 3 layers for 78D.
- **First attempt of the Unitree XR-1 smokes failed**: `xr1_joint28` ran out of GPU memory at 16 × 3 (two cameras
  double the VLM tokens to ~490; 66 GB at step 40 plus the other user's 7.5 GB), and `xr1_sonic78sonicstate` was
  stopped by SIGTERM (exit 143) during the same window; the container was not OOM-killed (host memory cap 200 GB).
  Rerun at 8 × 6 (same 48 per update): both pass. Failed logs kept as `logs/xr1_*_smoke.{log,exit}.failed-0533`.
- Checks on every smoke checkpoint (`check_smokes.sh`): `finalize_baseline.py` (strict reload, finite forward loss,
  chunk shape (1, 30, 28/78)), `ho5_openloop_eval.py` (all held-out episodes, stride 60) and `openloop_smooth.py`
  (one held-out episode). Held-out error at step 0, in std units (hold-previous in brackets) — **pipeline checks
  only; the models saw 15–50 updates**:

  | Checkpoint             | HE                                   | Unitree                              |
  | ---------------------- | ------------------------------------ | ------------------------------------ |
  | psi0_joint28           | arms 3.77, hands 2.59 (0.02, 0.07)   | arms 3.45, hands 1.93 (0.01, 0.01)   |
  | psi0_sonic78sonicstate | tokens 0.64, hands 1.16 (0.02, 0.07) | tokens 0.48, hands 0.36 (0.01, 0.01) |
  | xr1_joint28            | arms 0.22, hands 0.50                | arms 0.07, hands 0.47                |
  | xr1_sonic78sonicstate  | tokens 1.02, hands 0.59              | tokens 1.05, hands 0.54              |

  As expected at this stage: Psi0 28D starts from re-initialised projections (still in warmup); the SONIC-pretrained
  Psi0 header already places tokens well after ~25 updates; XR-1's relative arm actions start near "hold previous".

- `ho5_openloop_eval.py` had a bug on `feat/g1-combined-eval` (`zip(offsets, lengths, strict=True)` with one more
  offset than lengths) that made it exit 1 for every run; fixed on this branch.
- Smoke optimizer states were deleted (≈ 230 GB); the smoke model weights are kept for re-checks.

## Full runs

Approved 2026-09-30. Official batch × updates; LeRobot `steps` = updates × accumulation. GPU 2, container
`jihun-lerobot-psi0xr1-gpu2-20260930` (code `/mnt/data01/jhkim/code/lerobot-g1-psi0-xr1-dev`, exported from this
branch), queue `/run-output/psi0_xr1/queue_gpu2.txt` run by `official_queue.sh`. ETAs from the smoke speeds.

| Run                            | Batch × updates | GPU | Start (UTC)        | ETA (UTC)    |
| ------------------------------ | --------------- | --- | ------------------ | ------------ |
| HE xr1_joint28                 | 48 × 10K        | 2   | 09-30 09:56        | 09-30 ~17:40 |
| HE xr1_sonic78sonicstate       | 48 × 10K        | 2   | after it           | 10-01 ~02:30 |
| HE psi0_sonic78sonicstate      | 128 × 40K       | 2   | after it           | 10-03 ~11:00 |
| HE psi0_joint28                | 128 × 40K       | 2   | after it           | 10-05 ~23:00 |
| Unitree psi0_joint28 (+ smoke) | 128 × 40K       | –   | after the data fix | ~71 h        |
| Unitree psi0_sonic78sonicstate | 128 × 40K       | –   | after the data fix | ~71 h        |
| Unitree xr1_joint28            | 48 × 10K        | –   | after the data fix | ~16 h        |
| Unitree xr1_sonic78sonicstate  | 48 × 10K        | –   | after the data fix | ~14 h        |

- Only GPU 2 has room for these jobs now (50–56 GB each). GPU 6 runs the official queue (HE Diffusion next, Unitree
  Diffusion after the fix) next to ~20 GB of three idle containers; a 50 GB job there could starve it. When the
  official GPU 6 queue is done, move the HE Psi0 28D line (or the Unitree runs) to a second container there.
- The Unitree runs will be regenerated with `write_psi0_xr1_configs.py` after the fix (Psi0 state back to the
  official min/max; XR-1 stats recomputed) and appended to a queue.

**Placement as of 2026-10-02 03:00 UTC** (the GPU 2 container was killed on 10-01, exit 137):

| Run                            | Where             | State                                                     |
| ------------------------------ | ----------------- | --------------------------------------------------------- |
| HE xr1_joint28, xr1_sonic78    | H100 GPU 2        | done (step 30000); mid checkpoint and optimizer pruned    |
| HE psi0_sonic78sonicstate      | H100 GPU 0        | `_matched640k` after HE MolmoAct2 (GPU 3 cleared 10-02)    |
| HE psi0_joint28                | H100 GPU 0        | `_matched640k` after the 78D run; FastWAM after it         |
| Unitree xr1_joint28            | h100_174 GPU 4    | done (step 60000); pruned                                 |
| Unitree xr1_sonic78sonicstate  | h100_174 GPU 4    | running, ~44% at 10-02 02:40                              |
| Unitree psi0_sonic78sonicstate | h100_174 GPU 5    | `_matched640k` running, started 10-02 04:30 (~8.5 h)      |
| Unitree psi0_joint28           | h100_174 GPU 6    | `_matched640k` running, started 10-02 04:30 (~8.5 h)      |

**Psi0 at GR00T's sample budget (user decision 2026-10-02).** The VLAs are compared at equal training samples:
GR00T official is batch 32 × 20K updates = 640K samples (0.38 epoch of HE). The Psi0 `*_matched640k` runs keep batch
16 × accumulation 8 and every other setting, with `steps=40000` (steps count micro-batches: 40000 × 16 = 640K samples
= 5K updates), `policy.scheduler_warmup_steps=1000` (same warmup:total ratio as 8000:320000; the cosine preset decays
over `steps`) and `save_freq=40000`. The 320K-step `*_official_full` Psi0 runs were stopped before their first
checkpoint; their logs are kept as `*.log.stopped-for-matched640k`.

## Open-loop results (2026-10-02)

`openloop_smooth.py` on the six held-out HE episodes (91, 102, 1208, 1219, 1293, 1300), replan 12 frames, H100 GPU 7,
code `/mnt/data01/jhkim/code/lerobot-g1-psi0-xr1-eval` (this branch merged with `feat/g1-combined-eval`). Std units;
the baselines are from `2026-09-30-official-retraining-handover.md`. `temp:0` = zero initial flow noise.

| Model / mode         | Tokens: seam | Tokens: disagree | Tokens: err | Hands: seam | Hands: err | Hands: step (rec. 0.063) |
| -------------------- | ------------ | ---------------- | ----------- | ----------- | ---------- | ------------------------ |
| ACT                  | 0.087        | 0.085            | 0.132       | 0.120       | 0.236      | 0.015                    |
| GR00T temp:0         | 0.091        | 0.097            | 0.131       | 0.128       | 0.233      | 0.014                    |
| Pi0.5 temp:0         | 0.077        | 0.078            | 0.119       | 0.105       | 0.213      | 0.013                    |
| XR-1 78D random      | 0.115        | 0.123            | 0.135       | 0.194       | 0.248      | 0.058                    |
| **XR-1 78D temp:0**  | 0.081        | 0.086            | 0.122       | 0.133       | 0.219      | 0.011                    |

XR-1 28D (HE `joint28_g2`, arms 0:14 instead of tokens): random arms seam 0.081, disagree 0.087, err 0.082, hands
seam 0.198, err 0.239; **temp:0** arms seam 0.060, disagree 0.066, err 0.073, hands seam 0.124, err 0.209. Not
comparable with the token columns (different action space and std).

- XR-1 78D at temp:0 sits between GR00T and Pi0.5 on tokens and is the second most accurate on hands. Like the others,
  temp:0 trades the hands' motion inside a chunk (step 0.011 vs the recording's 0.063) for smoothness and accuracy.
- With random noise XR-1's seams (0.115) are smaller than GR00T random's (0.140) but still above ACT.
### All runs (2026-10-04)

Same script and replan; HE on the six HE held-out episodes (H100 GPU 7), Unitree on six held-out Unitree episodes
(216, 1302, 1994, 2155, 2526, 2612: `random.Random(0).sample` of `heldout_sonic78_nolimit_5pct.json`, all in the
runs' `exclude_episodes`; h100_174 GPU 5 and H100 GPU 7). Psi0 = `*_matched640k`. Unitree GR00T and ACT
(`*_official_full`) are the Unitree baselines on the same episodes; the std differs per dataset, so compare within a
block only.

**78D, HE**

| Model / mode         | Tokens: seam | Tokens: disagree | Tokens: err | Hands: seam | Hands: err |
| -------------------- | ------------ | ---------------- | ----------- | ----------- | ---------- |
| GR00T temp:0         | 0.091        | 0.097            | 0.131       | 0.128       | 0.233      |
| Pi0.5 temp:0         | **0.077**    | **0.078**        | **0.119**   | 0.105       | **0.213**  |
| XR-1 temp:0          | 0.081        | 0.086            | 0.122       | 0.133       | 0.219      |
| Psi0 temp:0          | 0.082        | 0.093            | 0.142       | **0.103**   | 0.234      |
| Psi0 random          | 0.172        | 0.197            | 0.171       | 0.277       | 0.276      |

**78D, Unitree**

| Model / mode         | Tokens: seam | Tokens: disagree | Tokens: err | Hands: seam | Hands: err |
| -------------------- | ------------ | ---------------- | ----------- | ----------- | ---------- |
| ACT                  | 0.106        | 0.116            | 0.180       | 0.083       | 0.166      |
| GR00T temp:0         | 0.104        | 0.105            | 0.190       | 0.083       | 0.163      |
| XR-1 temp:0          | 0.099        | 0.097            | **0.172**   | 0.085       | 0.158      |
| XR-1 random          | 0.139        | 0.138            | 0.184       | 0.126       | 0.170      |
| Psi0 temp:0          | **0.091**    | **0.094**        | 0.181       | **0.068**   | **0.122**  |
| Psi0 random          | 0.183        | 0.200            | 0.209       | 0.134       | 0.145      |

**28D (arms 0:14, hands 14:28)**

| Data / model / mode  | Arms: seam | Arms: disagree | Arms: err | Hands: seam | Hands: err |
| -------------------- | ---------- | -------------- | --------- | ----------- | ---------- |
| HE XR-1 temp:0       | **0.060**  | **0.066**      | **0.073** | 0.124       | **0.209**  |
| HE XR-1 random       | 0.081      | 0.087          | 0.082     | 0.198       | 0.239      |
| HE Psi0 temp:0       | 0.073      | 0.076          | 0.101     | **0.094**   | 0.228      |
| HE Psi0 random       | 0.172      | 0.196          | 0.147     | 0.262       | 0.285      |
| Unitree XR-1 temp:0  | **0.082**  | 0.090          | **0.076** | 0.078       | 0.143      |
| Unitree XR-1 random  | 0.095      | 0.108          | 0.082     | 0.116       | 0.154      |
| Unitree Psi0 temp:0  | 0.085      | **0.087**      | 0.109     | **0.070**   | **0.123**  |
| Unitree Psi0 random  | 0.136      | 0.162          | 0.132     | 0.129       | 0.145      |

- **temp:0 is required for Psi0.** With random noise Psi0 has the largest seams of any model (0.17–0.18 on tokens and
  arms, about 2× its temp:0); at temp:0 it is as smooth as the best.
- **Psi0 is best on hands, XR-1 on arms and tokens.** At temp:0 Psi0 has the smallest hand seams in every block and
  the most accurate Unitree hands (0.122 vs 0.158–0.166); XR-1 is the most accurate on arms (HE 0.073, Unitree 0.076,
  vs Psi0 0.101/0.109) and on Unitree tokens.
- On Unitree both new VLAs at temp:0 match or beat the GR00T and ACT baselines; on HE, Pi0.5 temp:0 is still the most
  accurate on tokens, with XR-1 close.
- The usual temp:0 cost holds: almost no hand motion inside a chunk (step ~0.006–0.012).

### Noise scale sweep, HE 78D (2026-10-06)

GR00T runs on the robot at noise 0.5 to escape stalls, so the two new VLAs were checked between 0 and 1.

| Model / noise | Tokens: seam | Tokens: err | Hands: seam | Hands: err | Hands: step (rec. 0.063) |
| ------------- | ------------ | ----------- | ----------- | ---------- | ------------------------ |
| XR-1 0        | 0.081        | 0.122       | 0.133       | 0.219      | 0.011                    |
| XR-1 0.25     | 0.084        | 0.122       | 0.135       | 0.220      | 0.012                    |
| XR-1 0.5      | 0.088        | 0.124       | 0.150       | 0.223      | 0.017                    |
| XR-1 1        | 0.115        | 0.135       | 0.194       | 0.248      | 0.058                    |
| Psi0 0        | 0.082        | 0.142       | 0.103       | 0.234      | 0.012                    |
| Psi0 0.25     | 0.095        | 0.145       | 0.128       | 0.237      | 0.020                    |
| Psi0 0.5      | 0.118        | 0.151       | 0.160       | 0.242      | 0.031                    |
| Psi0 1        | 0.172        | 0.171       | 0.277       | 0.276      | 0.048                    |

- XR-1 is nearly flat up to 0.5 (token seam +0.007, err +0.002), so 0.5 costs little if stalls need noise.
- Psi0 degrades steadily with noise; start it at 0 and raise only to 0.25 if it stalls.

## Open questions for the user

1. **Permission to apply the Unitree data fix** (writes 11 files in each of the three shared datasets, backups
   kept) and to append the pending official Unitree blocks to `gpu0.txt` / `gpu6.txt` afterwards. The permission
   system blocked it; it needs your explicit go-ahead (or a permission rule). Safe window: after
   `/run-output/logs/act_sonic78sonicstate_ho5_official_full.exit` exists (~10:40 UTC) and before the next Unitree job.
2. A second GPU for the Unitree runs (~170 GPU-hours): GPU 6 once its official queue ends, and/or the three idle
   GPU 6 containers (~20 GB).
3. Also repair `combined_sonicstate_1cam` (same Unitree frames)?

## Closed-loop sim (2026-10-06)

`sonic_official_sim_eval.sh` (NVIDIA deploy + MuJoCo, table 25 cm, planner start and end, slew `--max-token-step
0.05`, `--policy-timeout-s 5`, server `--noise-scale 0`) on the six HE held-out episodes, H100 GPUs 0 and 6, code
`lerobot-g1-psi0-xr1-eval`. HE GR00T `official_full` at noise 0 is the baseline (1293 / 1300 from the earlier run with
the same settings). Palm error is against the recording; jerk is the fastest arm joint's p95 (rad/s³, from
`robot_run_smoothness.py` on the streamer log).

| Model | Ep   | Palm p50 / p95 (cm) | Wrist p95 (°) | Tilt max (°) | Arm jerk p95 | Arm speed p95 (rad/s) |
| ----- | ---- | ------------------- | ------------- | ------------ | ------------ | --------------------- |
| GR00T | 91   | 11.6 / 34.5         | 102.8         | 3.4          | 582          | 1.12                  |
| GR00T | 102  | 9.2 / 40.9          | 91.8          | 2.6          | 707          | 1.25                  |
| GR00T | 1208 | 4.3 / 25.5          | 36.4          | 1.6          | 343          | 0.78                  |
| GR00T | 1219 | 3.7 / 23.9          | 42.1          | 2.4          | 840          | 1.29                  |
| GR00T | 1293 | 3.9 / 10.4          | 19.2          | 2.2          | 413          | 1.42                  |
| GR00T | 1300 | 3.6 / 10.8          | 23.9          | 2.2          | 394          | 1.57                  |
| XR-1  | 91   | 20.6 / 46.4         | 109.2         | 2.5          | 321          | 0.52                  |
| XR-1  | 102  | 22.1 / 48.5         | 113.0         | 2.3          | 298          | 0.46                  |
| XR-1  | 1208 | 8.5 / 53.4          | 57.8          | 2.4          | 245          | 0.73                  |
| XR-1  | 1219 | 9.0 / 53.9          | 62.3          | 3.1          | 308          | 0.45                  |
| XR-1  | 1293 | 4.8 / 29.3          | 57.8          | 2.3          | 300          | 1.01                  |
| XR-1  | 1300 | 6.1 / 29.7          | 56.7          | 2.9          | 518          | 1.44                  |
| Psi0  | 91   | 10.5 / 27.2         | 66.9          | 2.8          | 813          | 1.53                  |
| Psi0  | 102  | 6.6 / 27.7          | 81.1          | 3.3          | 531          | 1.29                  |
| Psi0  | 1208 | 3.7 / 22.6          | 34.7          | 3.2          | 484          | 1.34                  |
| Psi0  | 1219 | 3.2 / 19.0          | 34.7          | 2.7          | 788          | 1.39                  |
| Psi0  | 1293 | 2.4 / 6.3           | 17.2          | 1.9          | 969          | 1.96                  |
| Psi0  | 1300 | 3.3 / 10.2 (86 of 337 frames) | 19.5 | **9.8** | **3122**     | 4.24                  |

Means over the six episodes: palm p50 / p95 GR00T 6.1 / 24.3 cm, XR-1 11.9 / 43.5 cm, Psi0 5.0 / 18.8 cm (1300
partial); wrist p95 GR00T 53°, XR-1 76°, Psi0 42°.

- **Psi0 is the most accurate in closed loop**, ahead of GR00T on palm and wrist in five of six episodes. Its arm
  jerk is higher (484–969 vs GR00T's 343–840, so about 1.3× on average), and on episode 1300 the joint-speed watchdog
  ended the run after 2.9 s (arm joint 3 at 6.7 rad/s > 6), with 9.8° tilt and one floor contact by then.
- **XR-1 is the smoothest and never unstable, but it drifts** in every episode (palm p95 29–54 cm, its hand outputs
  1.3–1.4 rad from the recording at p95) although its open-loop chunks were accurate. It is not a robot candidate
  until the closed-loop drift is understood (check its state input on the robot-state path first).
- Episodes 91 and 102 are short and hard for every model, GR00T included.

## Workstation fit (RTX 3060 12 GB, GPU 1; 2026-10-06)

| Model            | Local copy                                                        | Peak GPU memory | Chunk latency (warm) |
| ---------------- | ----------------------------------------------------------------- | --------------- | -------------------- |
| HE XR-1 78D      | `/mnt/data/jihun/g1_models/he_xr1_sonic78sonicstate_ho5_official_full` | 9.8 GB      | 0.27 s               |
| HE Psi0 78D      | `/mnt/data/jihun/g1_models/he_psi0_sonic78sonicstate_ho5_matched640k` | 6.8 GB       | 0.28 s               |

- The Psi0 checkpoint is float32 (11.2 GB). Its local `config.json` sets `tune_vlm: false` (original in
  `config.json.orig`), which builds the VLM in bf16. The bf16 copy reproduces the H100 float32 open-loop numbers
  exactly at temp:0 (tokens err 0.142, hands err 0.234).
- Psi0 needs `openai/clip-vit-large-patch14` weights in the local HF cache (`combined_temb`); the workstation cache
  had only the tokenizer files, so `model.safetensors` was copied from the H100 cache (sha256 checked).
- Neither model loads with the main checkout's `lerobot` until this branch is merged; run the server with the main
  venv and `PYTHONPATH=<this worktree>/src:<this worktree>/examples/g1_dex3_training`.

### Follow-up: XR-1 state feedback, Psi0 chunk blend (2026-10-06)

Same settings. **XR-1 with the recorded state** (`--state-source dataset`: the policy sees the dataset's state, the
robot still executes) against the closed-loop runs above:

| XR-1, ep | Palm p50 / p95 (cm), robot state | Palm p50 / p95 (cm), recorded state | Hands p95 vs stored (rad) | Arm jerk p95 |
| -------- | -------------------------------- | ----------------------------------- | ------------------------- | ------------ |
| 1208     | 8.5 / 53.4                       | **1.7 / 7.3**                       | 1.31 → 0.32               | 245 → 572    |
| 1219     | 9.0 / 53.9                       | **1.9 / 6.0** (watchdog at 7.7 s)   | 1.42 → 0.20               | 308 → 330    |
| 1293     | 4.8 / 29.3                       | **2.1 / 4.9**                       | 1.28 → 0.22               | 300 → 792    |
| 1300     | 6.1 / 29.7                       | **2.5 / 4.5**                       | 1.40 → 0.21               | 518 → 445    |

- **XR-1's drift comes from its state input.** Fed the recorded state it is the most accurate model in sim (palm p95
  4.5–7.3 cm, better than GR00T's 10–26 and Psi0's 6–23 on the same episodes); fed the robot's measured state it
  drifts by 30–54 cm. Its hand outputs are deltas on the hand state (`relative_action_state_indices` 64:78 → state
  14:28), but the SONIC tokens are absolute and drift too, so the whole policy leans on the state: small tracking
  differences between the sim robot and the recording are fed back and compound.
- The right-hand order is converted for both the command and the state (`RIGHT_ORDER`), so it is not an ordering bug.
- The recorded state cannot be used on the robot. Options: train XR-1 with state noise or dropout (Psi0 trains
  with `state_noise_std`), drop the state from the token half, or make only the hands relative to the *commanded*
  hand target instead of the measured one.

**Psi0 with `--chunk-blend-s 0.3`** (cross-fade between chunks) and two repeats of episode 1300 without it:

| Psi0, ep        | Palm p50 / p95 (cm) | Token seam / step | Arm jerk p95 | Notes                                   |
| --------------- | ------------------- | ----------------- | ------------ | --------------------------------------- |
| 1300 (first)    | 3.3 / 10.2          | 0.121 / 0.011     | 3122         | watchdog at 2.9 s, 9.8° tilt            |
| 1300 r2         | 2.1 / 6.9           | 0.065 / 0.009     | 308          | full run                                |
| 1300 r3         | 2.1 / 6.7           | 0.064 / 0.009     | 748          | full run                                |
| 1300 blend 0.3  | 2.2 / 8.9           | 0.008 / 0.009     | 747          | full run                                |
| 91 blend 0.3    | 9.2 / 32.6          | 0.008 / 0.009     | 231 (813)    |                                         |
| 1293 blend 0.3  | 2.4 / 6.4           | 0.008 / 0.009     | 894 (969)    |                                         |
| 1219 blend 0.3  | 3.0 / 19.4          | 0.005 / 0.007     | 195 (788)    |                                         |

- The episode-1300 watchdog stop happened in 1 of 3 runs without blending: intermittent, not systematic.
- Blending removes the chunk seams (token seam 0.06–0.12 → 0.005–0.008), cuts jerk on 91 and 1219 by ~3–4×, and
  keeps accuracy (palm p95 within ±5 cm). First Psi0 robot runs: noise 0, `--chunk-blend-s 0.3`, slew 0.05, the
  default joint-speed watchdog.

## Real robot (2026-10-06)

Workstation GPU 1 server (`g1_groot_real_run.sh server` with `MODEL=.../he_psi0_sonic78sonicstate_ho5_matched640k
BACKBONE_DTYPE= NOISE_SCALE=0`), PC2 streamer `--max-token-step 0.05 --chunk-blend-s 0.3`, planner start/end, 30 s,
head camera 640×480 `egocentric`, "close a laptop g1". GR00T runs 20–21 (noise 0, slew 0.1, no blend) for reference,
from `robot_run_smoothness.py`:

| Run   | Model / settings                    | Token seam / step | Hands seam / step | Arm speed p95 | Arm jerk p95 | 1–2 Hz | Result (user)                                                    |
| ----- | ----------------------------------- | ----------------- | ----------------- | ------------- | ------------ | ------ | ---------------------------------------------------------------- |
| run20 | GR00T, noise 0, slew 0.1            | 0.107 / 0.005     | 0.045 / 0.002     | 0.99          | 514          | 7.4 %  | acceptable zero-shot (touched the laptop)                        |
| run21 | GR00T, noise 0, slew 0.1            | 0.074 / 0.005     | 0.037 / 0.002     | 0.70          | 315          | 7.1 %  | acceptable zero-shot                                             |
| run22 | Psi0, noise 0, blend 0.3, slew 0.05 | 0.003 / 0.003     | 0.004 / 0.005     | 0.50          | 194          | 9.6 %  | reached, touched the lower part of the laptop, not the lid; did not close it; smooth |

- Full 30 s, no watchdog stop, slew limit never active, 73 chunks at 0.29 s.
- The smoothest robot run so far: no seams, jerk 194 vs GR00T's 315–514. The arm also moved slower (speed p95 0.50
  vs 0.70–0.99), so part of the smoothness may be less motion.
- PC2 deploy: `g1_groot_real_run.sh deploy` crashed at "Creating G1Deploy object" (`corrupted size vs. prev_size`);
  it ran from a PC2 shell after `conda deactivate`, with the SDK's own DDS libraries preloaded:
  ```bash
  cd ~/GR00T-WholeBodyControl/gear_sonic_deploy && conda deactivate && source scripts/setup_env.sh
  export LD_PRELOAD="$(pwd)/thirdparty/unitree_sdk2/thirdparty/lib/aarch64/libddsc.so:$(pwd)/thirdparty/unitree_sdk2/thirdparty/lib/aarch64/libddscxx.so"
  bash deploy.sh --cp policy/sonic_v1_1/model --obs-config policy/sonic_v1_1/observation_config.yaml \
    --input-type zmq_manager --zmq-host localhost real
  ```

## Handover: real-robot continuation (open, 2026-10-06)

To be continued by another session. State at handover:

- **Psi0 server is running** on the workstation as user unit `groot-server-5560` (GPU 1, port 5560, noise scale 0,
  model `/mnt/data/jihun/g1_models/he_psi0_sonic78sonicstate_ho5_matched640k`, VLM in bf16 via the local
  `config.json`; log `server_gpu1_5560_t0.log` in the model dir). Stop it with `./g1_groot_real_run.sh stop`.
- **Robot side:** start the deploy with the PC2 command in the run22 note above (`conda deactivate` + `LD_PRELOAD`;
  the launcher's `deploy` step crashes). Camera: `./g1_groot_real_run.sh camera`.
- **Last run:** run22 (see the table above). Logs on PC2 in `~/g1_sonic_eval/runs/`, copies in `~/g1_runs/`.

Next trials, in this order, one change at a time (same scene, "close a laptop g1"):

1. run23 — same settings as run22, for repeatability:
   `./g1_groot_real_run.sh stream run23_he_psi0_laptop_t0 "close a laptop g1" --max-token-step 0.05 --chunk-blend-s 0.3`
2. run24 — slew 0.1 (GR00T's setting; run22 never hit 0.05):
   `... stream run24_he_psi0_laptop_t0 "close a laptop g1" --max-token-step 0.1 --chunk-blend-s 0.3`
3. run25 — noise 0.25 if it still stops short of the lid (restart the server with `NOISE_SCALE=0.25`):
   `... stream run25_he_psi0_laptop_t025 "close a laptop g1" --max-token-step 0.1 --chunk-blend-s 0.3`

After each run: copy the `.npz`/`.log` back, run `robot_run_smoothness.py
/mnt/data/jihun/datasets/he_sonic78_nolimit_sonicstate_heldout6 <runs>`, and add a row with the user's observation
(reach, which part of the laptop, closed or not, smoothness) to the table above. XR-1 is not a robot candidate (its
closed-loop drift, see the sim follow-up); do not run it on the robot until it is retrained without the state
dependence.
