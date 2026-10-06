# Open issue: GR00T predicts jerky SONIC token chunks although the dataset replays smoothly (2026-09-29)

**Status: open. To be handled in a new session.** Found during the first real-robot runs
([`2026-09-29-sonic-real-robot-first-runs.md`](./2026-09-29-sonic-real-robot-first-runs.md)).

This repository is public: addresses, user names and credentials are left out on purpose.

## Summary

The trained HE GR00T (`humanoid_everyday_g1_20260923/runs/groot_sonic78sonicstate_ho5_full`) produces jerky arm
motion on the real G1 and in sim, but the SONIC dataset it was trained on (`sonic78_nolimit`) replays smoothly
through NVIDIA's deploy. The jerk is already present in the model's open-loop predictions on recorded held-out
observations (images from HE's own camera, so no visual gap): the model's output itself is not smooth. The dataset,
the streamer, the robot and the D435i visual gap are therefore not the main cause. The training procedure **may**
have a problem (leading suspect: batch size / amount of training); the action normalization or the inference
settings (flow-matching sampling) are the other candidates. The closed loop on the real robot can amplify the
motion (a larger ~1 Hz back-and-forth than in sim), but does not create it.

## Evidence

**1. The dataset is fine.** Official-deploy MuJoCo replay of three no-limit training episodes (stored tokens, no
policy), reviewed visually: no jerky motion.

| Episode                                      | Palm error p50 / p95           | Tilt during the episode |
| -------------------------------------------- | ------------------------------ | ----------------------- |
| `unitree:1841` Put the apple into the plate. | 1.7 / 2.9 cm                   | 2.5°, all feet down     |
| `humanoid_everyday:99` close a laptop g1     | 3.3 / 5.1 cm (wrist p95 19.7°) | 3.6°, all feet down     |
| `humanoid_everyday:1232` push duck g1        | 1.7 / 3.5 cm                   | 3.6°, all feet down     |

Output on H100: `sonic_roundtrip_20260923/review_nolimit_20260929/` (`replay_raw.mp4`, `compare.json`); made with
`sonic_replay_extract.py` (`SONIC_SUFFIX=_nolimit`), `sonic_official_replay.py`, `sonic_replay_compare.py`.

**2. The predictions are not.** Open loop on 6 HE held-out episodes (91, 102, 1208, 1219, 1293, 1300), a chunk every
0.4 s, in action-std units, tokens (hands):

| Model                   | Consecutive chunks disagree | Seam jump     | Step inside chunks | Recording step | Curvature (recording 0.038) |
| ----------------------- | --------------------------- | ------------- | ------------------ | -------------- | --------------------------- |
| GR00T                   | 0.238 (0.339)               | 0.211 (0.312) | 0.033              | 0.019          | 0.048                       |
| GR00T, fixed noise seed | 0.152 (0.175)               | 0.125 (0.136) | 0.032              | 0.019          | 0.047                       |
| ACT (same data)         | 0.096 (0.108)               | 0.092 (0.129) | 0.016              | 0.019          | 0.014                       |

- Consecutive chunks disagree 2.5× more than ACT's; seams jump 6× the normal step (11× the recording's step).
- Inside a chunk GR00T moves 1.7× more per frame than the demonstrations.
- Two samples for the same observation differ by ~0.23 std, as much as the model's error against the recording.

Script: `openloop_smooth.py` (`examples/g1_dex3_training/openloop_smooth.py`).

**3. Effect on the robot** ("close a laptop g1", runs 15–17): arm speed p95 1.65–1.97 rad/s against 1.12 in the
human demonstrations; an extra ~1 Hz back-and-forth (11–34 % of the motion power at 1–2 Hz against 4 % in the
demonstrations); arm jerk p95 400–1018 rad/s³ against 318 for ACT. GR00T does reach the right target (right palm
41–43 cm forward, as in training); ACT does not.

## Leads (training side)

1. **Severe undertraining / small batch.** GR00T was trained with **batch 4** for 40K steps = 160K samples ≈ **9 %
   of one pass** over HE (1.78 M frames); ACT used batch 32. VLA-JEPA failed with batch 1 on the same tokens and
   recovered with batch 8 ([`2026-09-29-vla-jepa-sonic-tokens.md`](./2026-09-29-vla-jepa-sonic-tokens.md)). A
   flow-matching head this undertrained could give high-variance, inconsistent samples.
2. **Action normalization of the 1/16-grid tokens.** LeRobot normalization is `IDENTITY` for GR00T; its own
   processor rescales with q01/q99 (min-max style). The VLA-JEPA note found MIN_MAX distorts token errors 2–3×;
   check GR00T's pack/unpack statistics for the SONIC tokens.
3. **Inference settings.** `num_inference_timesteps` is unset (model default); more flow steps or averaging several
   samples may reduce the sample spread.
4. The same recipe applies to the Unitree and combined GR00T runs; check them too (the combined GR00T hit the slew
   limit on 106–130 ticks per real run).

## Comparison with NVIDIA Isaac-GR00T (2026-09-29)

Isaac-GR00T clone at `626af89` (includes `3df8b38`, "Add SONIC embodiment"), read against the LeRobot port and the
run's `train_config.json`. The effective Isaac-GR00T values come from `gr00t/experiment/launch_finetune.py` plus
the base checkpoint's `config.json` / `processor_config.json`, which override the code defaults.

**Same in both:**

- Architecture: 32-layer DiT, 4-layer VL self-attention, LLM cut at layer 16.
- Flow matching: Beta(1.5, 1) time sampling, `noise_s` 0.999, 1000 buckets, masked velocity MSE, 4 Euler steps.
- Dropout: DiT dropout 0.2.
- Actions: absolute. NVIDIA's own `unitree_g1_sonic` config is also absolute, 40 × 78 (64 tokens + 7 + 7).
- Embodiment slot: randomly initialised. Slots 10 (`new_embodiment`) and 11 (`unitree_g1_sonic`) are untouched in
  the checkpoint (std 0.02, bias 0).
- Frozen LLM and vision tower.
- No EMA, action noise or ensembling. NVIDIA's SONIC client switches chunks hard at 2.5 Hz, so it relies on
  consecutive chunks agreeing.

**Different, most likely cause first:**

| #   | Item                 | Isaac-GR00T                                                                                                       | Our run                                                                                                                                           |
| --- | -------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Samples / batch      | Batch 32–64. NVIDIA's SONIC guide (`GR00T-WholeBodyControl` `vla_workflow.md`): 32 × 20K steps = **640K samples** | **Batch 4** × 40K = **160K** (4× fewer samples, batch 8× smaller)                                                                                 |
| 2   | Images               | Square letterbox, shortest edge 256, random crop 0.95, ColorJitter (0.3/0.4/0.5/0.08); eval centre crop 0.95      | 640×480 squashed to 256×256, fixed centre crop 0.898, **no augmentation** (the random crop exists only in the albumentations path, which was off) |
| 3   | Action normalization | q01/q99 → [-1, 1], clipped                                                                                        | min/max → [-1, 1], clipped. The range is 6.9 std instead of 4.4 std for tokens, so the same normalized error is ~1.6× larger in raw units         |
| 4   | State dropout        | 0.2 on the raw state (processor) and 0.2 on the state features (model)                                            | Model only                                                                                                                                        |
| 5   | Warmup               | 5 % of the steps                                                                                                  | 500 steps (1.25 %): `get_scheduler_preset` uses the unused `max_steps=10000`, not `--steps`                                                       |
| 6   | Data rate            | SONIC data at 50 Hz: 40 steps = 0.8 s                                                                             | 30 Hz: 40 steps = 1.33 s (a deliberate choice, see the audit note)                                                                                |
| 7   | Batch contents       | One shard per batch (episode-correlated)                                                                          | Fully shuffled; not a cause                                                                                                                       |

Ruled out:

- The eval path: inference mode, centre crop, no dropout.
- The normalization stats themselves: they are sane.

**Why items 2–4 happened.** `_load_n1_7_checkpoint_processor_assets` reads the base checkpoint's
`processor_config.json` only when `base_model_path` is a local directory. Our run passed the Hub id, so every
processor setting fell back to LeRobot's defaults. The base checkpoint's `embodiment_id.json` also has no
`new_embodiment` entry: pointing at a local copy would have mapped it to slot 0, a trained slot. Isaac-GR00T fills
missing tags from its default map (`new_embodiment` = 10).

**Fix (this branch, `processor_groot.py`).** Without sidecars, the processor now uses the base checkpoint's values:

- q01/q99 normalization for state and action, and its inverse (`GrootActionUnpackUnnormalizeStep.use_percentiles`).
- Raw-state dropout 0.2.
- Albumentations geometry with letterbox, shortest edge 256 and crop 0.95.
- Train-time ColorJitter, one draw per sample replayed across views (new `color_jitter_params`).
- Missing tags in the embodiment map are filled from the defaults.

Saved pipelines, i.e. existing checkpoints, are unchanged: the new fields default to off. Test:
`tests/policies/groot/test_groot_new_embodiment_defaults.py`. Cost: +~90 ms of CPU preprocessing per batch of 32.
Warmup is not changed in code; set `scheduler.num_warmup_steps` (e.g. 5 % of `--steps`) in the retrain config.

## Other policies against their official recipes (2026-09-29)

The same question for the other HE policies. Sources:

- ACT: tonyzhaozh/act and arXiv 2304.13705.
- VLA-JEPA: arXiv 2602.10098, github.com/ginwind/VLA-JEPA and `lerobot/VLA-JEPA-Pretrain`.
- Pi0.5: openpi `src/openpi/training/config.py` and `optimizer.py`.

"Our run" values:

- ACT: from the local checkpoint.
- Pi0.5 and VLA-JEPA: from `write_foundation_configs.py` plus the Hub configs it copies. Their effective
  `train_config.json` files are on the H100 and were not read.

One pass over HE ≈ 1.69M samples after the hold-out.

| Policy   | Official batch × steps (samples)                                                  | Our batch × steps (samples)                       | Other deviations                                                                                                                                                                                                                                                                                                                    |
| -------- | --------------------------------------------------------------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ACT      | 8; real robot: ≥ 5000 epochs of 1 frame per episode, "3–4× past the loss plateau" | 32 × 40K (1.28M ≈ 0.76 pass)                      | Horizon 100 at 30 Hz = 3.3 s (official 100 at 50 Hz = 2 s). Grad clip 10 (official none). Otherwise the official recipe: lr 1e-5 constant, wd 1e-4, MEAN_STD.                                                                                                                                                                       |
| Pi0.5    | 32 × 30K (960K); LIBERO 256 × 30K; DROID fine-tune 32 × 20K                       | **4** × 40K (160K ≈ 0.09 pass)                    | No EMA (official 0.99). **All parameters bf16** (`precision="bfloat16"` casts the model; openpi notes that bf16 training gives higher loss). No image augmentation (official: crop 95 %, rotate ±5°, ColorJitter). wd 0.01 (official 1e-10). 78D resets the action projections. Quantile normalization matches the official recipe. |
| VLA-JEPA | 256 × 30K (7.68M)                                                                 | **1** × 40K (40K); Unitree retrain 8 × 40K (320K) | **One learning rate for all modules** (`get_optim_params` returns every parameter at 1e-4); official: Qwen 1e-5, action head 1e-4, predictor 5e-4. min_lr 1e-6 comes from the pretrain recipe (fine-tune: 1e-5). The single ego view is duplicated for the 2-view world model. Official actions are deltas; ours are absolute.      |
| GR00T    | 32 × 20K (640K), NVIDIA SONIC guide                                               | **4** × 40K (160K ≈ 0.09 pass)                    | See the section above.                                                                                                                                                                                                                                                                                                              |

Reading:

- **Every foundation model was trained with 4–190× fewer samples than its official recipe.** Only ACT is close to
  its official recipe, and it is the smoothest model open loop.
- GR00T and Pi0.5 got the same batch 4. VLA-JEPA only learned the tokens once its batch rose from 1 to 8.
- The shared small-batch recipe is therefore the common suspect, and the fix to try first is the same for all:
  official batch (32+) and sample count.
- Pi0.5 and VLA-JEPA carry recipe deviations of their own, to fix before their retrains: bf16 weights without EMA;
  one LR for all modules.

## Retraining with the official recipes (started 2026-09-29)

Four HE models were retrained on the H100 (GPUs 0 and 6). Everything else is unchanged from the `*_ho5_full` runs:
the `sonic78_nolimit_sonicstate` dataset, the same held-out exclusions and seed.

Paths:

- Configs: `humanoid_everyday_g1_20260923/configs/*_sonic78sonicstate_ho5_official_{smoke,full}.json`
- Logs and `.exit` files: `logs/`
- Outputs: `runs/`

The code is exported from this branch into its own H100 directories:

- GPU 6 runs `lerobot-g1-groot-fix-20260929` (`dfe36b45`).
- GPU 0 runs `lerobot-g1-official-20260929-ceb40b77`.

Containers: `jihun-lerobot-he-official-gpu{0,6}-20260929`.

| GPU | Model    | Recipe                                                                                                                                                                                                                                                             | Smoke: memory, speed | Estimate |
| --- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------- | -------- |
| 6   | GR00T    | Batch 32 × 20K steps; lr 1e-4 cosine, warmup 5 % (1000); fixed processor (q01/q99, letterbox + crop 0.95 + ColorJitter, state dropout)                                                                                                                             | 36 GB, 0.62 s/step   | ~3.5 h   |
| 6   | ACT      | Batch 8 (official) × 200K steps (1.6M samples); rest as before (lr 1e-5, chunk 100, MEAN_STD)                                                                                                                                                                      | not measured         | ~2 h     |
| 0   | Pi0.5    | 8 × 4 accumulation = 32 per update × 30K updates; fp32 weights with bf16 autocast; EMA 0.99; openpi aug (crop 95 %, rotate ±5°, ColorJitter 0.3/0.4/0.5); AdamW (0.9, 0.95), wd 1e-10, clip 1; cosine 2.5e-5 → 2.5e-6, warmup 1K updates; `foreach=False` (memory) | 72 GB, 1.6 s/update  | ~13 h    |
| 0   | VLA-JEPA | 8 × 4 = 32 per update × 30K updates; fp32 weights; per-module peak lr (base 3e-5, Qwen 1e-5, action head 1e-4, predictor 5e-4), warmup 5K updates, cosine to 1/3 of peak (official `min_lr` 1e-5); MIN_MAX                                                         | 52 GB, 1.8 s/update  | ~15 h    |

Departures from the official recipes, all forced by one GPU per run:

- Pi0.5 and VLA-JEPA use gradient accumulation (8 × 4). **LeRobot steps the scheduler per micro-batch.**
  `AcceleratorConfig` builds `Accelerator(step_scheduler_with_optimizer=False)`, so `steps`, warmup and decay all
  count micro-batches: Pi0.5 warmup 4K / decay 120K, VLA-JEPA 20K / 120K, i.e. 1K / 30K and 5K / 30K updates. EMA
  updates once per optimizer update.
  - The first Pi0.5 launch had the schedule in updates (4× too fast) and was stopped at ~1K micro-steps. Its log
    is kept as `logs/pi05_sonic78sonicstate_ho5_official_full_badsched.log`.
- VLA-JEPA uses batch 32, not the official 256 (8 GPUs × 32).
- Batch 32 in fp32 did not fit on 80 GB for either model: the failed configs are kept as `*_oom_b32.json`.

Not retrained: Diffusion, MolmoAct2 and FastWAM. They were not compared with their official recipes.

Code on this branch:

- `dfe36b45`: GR00T processor.
- `b01a01db`: VLA-JEPA `optimizer_module_lrs`.
- `ceb40b77`: `AdamWConfig.foreach`.

### Full matrix: seven policies × HE and Unitree (queued 2026-09-29)

User decision: retrain every VLA/WAM policy plus ACT and Diffusion Policy on both HE
(`humanoid_everyday_g1_20260923/`, 1 camera) and Unitree (`/run-output/`, 2 cameras, 2.59M frames).

Batch × updates follow each official recipe. The Pi0.5 recipe is the fallback where none exists, but every policy
had one.

Queues:

- Runner: `examples/g1_dex3_training/official_queue.sh`.
- Queue files: `/run-output/queue_official/gpu{0,6}.txt`, one `<root>/<config>` per line, re-read after each job,
  so lines can be appended.
- Each smoke job gates its full job.
- Logs and `.exit` files: `<root>/logs/`.

| Policy    | Official recipe used                                                                                                                    | Per-update batch (micro × accum) | Updates | GPU |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | ------- | --- |
| GR00T     | NVIDIA SONIC guide                                                                                                                      | HE 32 × 1, Unitree 16 × 2        | 20K     | 6   |
| ACT       | tonyzhaozh/act (batch 8)                                                                                                                | 8 × 1                            | 200K    | 6   |
| Diffusion | Chi et al. image recipe: batch 64, AdamW, EMA (power 0.75), ResNet18 from scratch + GroupNorm, 240×320 crop 0.9                         | 64 × 1                           | 200K    | 6   |
| Pi0.5     | openpi (see above)                                                                                                                      | HE 8 × 4, Unitree 4 × 8          | 30K     | 0   |
| VLA-JEPA  | official per-module LRs (see above)                                                                                                     | 8 × 4                            | 30K     | 0   |
| MolmoAct2 | paper Table 16 real-world fine-tune: full fine-tune, official image augmentation (no blur)                                              | 8 × 2 = 16                       | 50K     | 0   |
| FastWAM   | paper real-robot 30K steps at per-GPU batch 16; FastWAM trainer optimizer (AdamW 0.9/0.95, wd 0.01, clip 1, 5 % warmup, cosine to 1e-6) | 8 × 2 = 16                       | 30K     | 0   |

Notes:

- Diffusion keeps LeRobot's horizon 64 / 32 actions for the streamer (official 16 / 8 at 10 Hz).
- MolmoAct2's recipe was chosen by the user over the README's 64 × 50K. FastWAM's recipe was chosen over the LIBERO
  128 × 20K.
- The large models need GPU 0: FastWAM and full-fine-tune MolmoAct2 hold ≥ 48 GB before activations.
  - GPU 6 has ~45 GB free next to other users. Its queue should finish around 2026-09-30 noon.
  - GPU 0's queue runs about 8–9 more days by rough estimate (MolmoAct2 and FastWAM not yet measured).

## Mitigations in place (not a fix)

Options, off by default: `--noise-seed` (server), `--chunk-blend-s`, `--replan-s`, `--max-token-step` (streamer). In
sim (HE GR00T, planner start, held-out episodes 1293 / 1300), seed + blend 0.3 s cut palm jerk p95 from 126 / 166 to
44 / 43 m/s³ and the 1–2 Hz share from 2.9 / 3.3 % to 1.6 / 1.0 %, at +1–1.5 cm median palm error.

## Closed-loop finding: the chunk switch, not the plan change (2026-10-06)

Five closed-loop rounds of the 28D HE GR00T (`groot_joint28_ho5_official_full`) through SONIC and the decoupled WBC,
6 held-out episodes x 3 repeats each (details: `docs/research/2026-10-02-g1-wbc-backends.md`, "Closed loop: results"),
measured from the streamer's 50 Hz arm targets:

- Across a chunk switch the arm target moves 0.08 rad (p50) / 0.45-0.52 rad (p95) over the next 0.3 s, in every
  round. The recorded HE demos move just as much: 0.12 / 0.50 rad over 0.3 s (p99 0.68), more than 0.3 rad in 17 % of
  windows. **The plan changes between chunks are normal-sized motion, not abnormal jumps.**
- The jerk comes from executing that change in one tick. With chunks switched at once the commanded arm speed peaks
  at ~19-22 rad/s (median over runs), which trips the 6 rad/s watchdog in 6-10 of 18 runs. A 0.3 s blend brings it to
  3-4 rad/s, closed-loop RTC (`--rtc`) to 5-8 rad/s; both remove almost all trips.
- Noise scale 0 alone does not remove trips (its jumps repeat on the same episodes) but halves the motion inside a
  chunk away from switches (p95 0.36 -> 0.23 rad per 0.3 s).
- RTC keeps following its own plan: valid runs 18/18 (28D through SONIC and C) but the motion moves further from the
  demo (SONIC palm p50 6.4 vs 4.5 cm with noise 0 + blend), and the 78D token GR00T drifts badly (palm vs demo p50
  19.7 cm vs 4.1 cm), as RTC already did open loop. Do not use RTC with the token model.

So for inference the fix is a smooth switch (noise scale 0 + `--chunk-blend-s 0.3`: 53 of 54 runs valid, closest to
the demo). A training-side fix would have to make consecutive chunks agree on their overlap, which the official
recipe does not target.

## Next steps

- [x] Compare the training pipeline with Isaac-GR00T (section above); processor fallback fixed on this branch.
- [ ] (running, see the retraining section) Retrain HE GR00T with NVIDIA's recipe (batch 32, ≥ 20K steps, warmup 5 %, fixed processor); compare the open-loop smoothness
      metrics above before any sim or robot run.
- [x] Check GR00T's action normalization statistics for the SONIC tokens: sane, but min/max instead of q01/q99 (fixed).
- [ ] Try more flow-matching steps / sample averaging at inference.
- [ ] Before retraining Pi0.5 / VLA-JEPA: official batch (32+), Pi0.5 with fp32 weights (or bf16 compute only) and
      EMA 0.99, VLA-JEPA with per-module LRs (Qwen 1e-5, head 1e-4, predictor 5e-4) and fine-tune min_lr 1e-5.
- [ ] Re-evaluate in order: open loop → sim (planner start, table 25 cm) → robot.
