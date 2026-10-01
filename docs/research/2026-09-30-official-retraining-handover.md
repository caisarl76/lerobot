# Handover: retraining every G1 SONIC policy with its official recipe (2026-09-30)

**Start here to continue the work.** It follows the open GR00T jerk issue
([`2026-09-29-issue-groot-jerky-predictions.md`](./2026-09-29-issue-groot-jerky-predictions.md)); that note holds the
comparisons and the reasons for every setting below. This repository is public: addresses, user names and credentials
are left out on purpose.

## Why

The HE GR00T predicted jerky SONIC chunks although the dataset replays smoothly. Two comparisons explained it.

**GR00T against NVIDIA's Isaac-GR00T:**

- Our run trained on 4× fewer samples at 8× smaller batch (4 × 40K) than NVIDIA's recipe.
- The LeRobot processor silently fell back to non-NVIDIA preprocessing when `base_model_path` was a Hub id:
  - min/max normalization instead of q01/q99
  - no image augmentation
  - the 4:3 frame squashed to a square
  - no raw-state dropout

**The other policies against their official recipes:** every foundation model was trained on 4–190× fewer samples.
On top of that:

- Pi0.5 trained with bf16 weights and no EMA.
- VLA-JEPA used one learning rate for all modules.
- FastWAM used a constant LR with betas 0.999.

The user decided to retrain every policy on both datasets with its official batch × updates. The Pi0.5 recipe is the
fallback when a policy has none, but every policy had one.

## Code (branch `feat/g1-combined-eval`, not pushed)

| Commit                             | Change                                                                                                                                                                                                                                                                                                    |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dfe36b45`                         | GR00T processor: without checkpoint sidecars (Hub id), use the base checkpoint's processor values: q01/q99, state dropout 0.2, letterbox + crop 0.95 + ColorJitter (new), embodiment map filled so `new_embodiment` is slot 10, not 0. Test `tests/policies/groot/test_groot_new_embodiment_defaults.py`. |
| `b01a01db`                         | VLA-JEPA `optimizer_module_lrs` (per-submodule peak LR). Test `tests/policies/vla_jepa/test_optim_groups.py`.                                                                                                                                                                                             |
| `ceb40b77`                         | `AdamWConfig.foreach` (False avoids multi-tensor temporaries; needed for fp32 Pi0.5 + EMA on 80 GB).                                                                                                                                                                                                      |
| `4b55559b`                         | `examples/g1_dex3_training/official_queue.sh`: per-GPU queue runner.                                                                                                                                                                                                                                      |
| `65893b97`, `69719330`, `c44b3242` | Issue-note sections: comparisons, retraining, full matrix.                                                                                                                                                                                                                                                |

Tests pass locally:

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest pytest tests/policies/groot tests/policies/vla_jepa
```

- `env -u PYTHONPATH` keeps ROS's pytest plugins out.
- The venv has no pytest; do not `uv sync`, which would drop installed extras.
- The two full-model GR00T tests OOM on the workstation's 12 GB GPUs.

## What runs on the H100

Layout under `/mnt/data01/jhkim/model_weight/g1_dex3_20260922` (= `/run-output` in the containers):

- HE: `humanoid_everyday_g1_20260923/{configs,logs,runs}`, 1 camera, 1.78M frames.
- Unitree: `{configs,logs,runs}` at the top level, 2 cameras, 2.59M frames.
- Dataset in both: `datasets/sonic78_nolimit_sonicstate`, with the ho5 exclusions.
- Jobs are named `<policy>_sonic78sonicstate_ho5_official_{smoke,full}`.

**Containers.** All use image `4cbe2a3f7fc6`, venv `/run-output/environment/venv`, HF cache mounted,
`HF_HUB_OFFLINE=1`.

| Container                                 | GPU | Code mounted at `/workspace/lerobot`                                | Role (2026-10-01 06:00 UTC)                                                                                     |
| ----------------------------------------- | --- | ------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `jihun-lerobot-official-gpu0c-20261001`   | 0   | `/mnt/data01/jhkim/code/lerobot-g1-official-20260929-ceb40b77`      | GPU 0 queue runner (`gpu0.txt`) and the hourly prune loop. `--memory 300g`.                                     |
| `jihun-lerobot-official-gpu6b-20261001`   | 6   | `/mnt/data01/jhkim/code/lerobot-g1-groot-fix-20260929` (`dfe36b45`) | Watcher, then the GPU 6 queue runner (`gpu6.txt`). `--memory 250g`.                                             |
| `jihun-lerobot-he-official-gpu6-20260929` | 6   | same as `gpu6b`                                                     | GPU-less since ~03:30 UTC, but still finishes Unitree GR00T, which holds its GPU. Remove it once that run ends. |

Each code directory is a `git archive` export of the named commit. They are not git checkouts. Start any new training
container like `gpu0c`: with the `/source-datasets` mount, the cache variable and a memory cap, then check
`nvidia-smi -L` inside it.

**Queues:**

- Runner: `/run-output/queue_official/official_queue.sh` (the same file as in the repo).
- Queue files: `gpu6.txt` and `gpu0.txt`, runner logs `gpu{6,0}.runner.log`.
- Each line is `<root>/<job>`; the file is re-read after every job, so lines can be appended.
- A job is done when `<root>/logs/<job>.exit` exists.
- A `_full` job is skipped (exit `skip`) if its `_smoke` job failed.

**State at 2026-09-30 ~02:10 UTC (H100 clock):**

| Policy    | HE (1 camera)                                                                           | Unitree (2 cameras)                               |
| --------- | --------------------------------------------------------------------------------------- | ------------------------------------------------- |
| GR00T     | Done (exit 0, 20K); being judged open-loop and served by session `f189249f` (see below) | Training on GPU 6: smoke started 01:58, then full |
| ACT       | Done (exit 0, 200K); not yet evaluated                                                  | Queued on GPU 6, next                             |
| Diffusion | Queued on GPU 6, from scratch (the leaked run is kept as `runs/..._leaked`)             | Queued on GPU 6, last                             |
| Pi0.5     | Training on GPU 0: 62 %, ~5.5 h left                                                    | Queued on GPU 0, 2nd                              |
| VLA-JEPA  | Queued on GPU 0, 1st (after HE Pi0.5)                                                   | Queued on GPU 0, 3rd                              |
| MolmoAct2 | Queued on GPU 0, 4th                                                                    | Queued on GPU 0, 6th                              |
| FastWAM   | Queued on GPU 0, 5th                                                                    | Queued on GPU 0, 7th                              |

- `gpu6.txt` order: Unitree groot → Unitree act → HE diffusion → Unitree diffusion. The HE groot and act lines are done
  and skipped.
- `gpu0.txt` order: HE vla_jepa → Unitree pi05 → Unitree vla_jepa → HE molmoact2 → HE fastwam → Unitree molmoact2 →
  Unitree fastwam.
- Earlier versions of the queue files are kept as `gpu{0,6}.txt.bak-20260930`.
- Failed attempts are kept under renamed files:
  - Unitree groot, act and diffusion smoke tests from 01:48, which failed on the missing mount:
    `logs/*.{exit,log}.failed-nomount`
  - the leaked HE Diffusion run: `HE/logs/diffusion_..._full.{exit,log}.leaked`

**Workstation (not the H100).** Session `f189249f` (listed as `lerobot-16`) is doing three things:

- copying the new HE GR00T checkpoint to `/mnt/data/jihun/g1_models/he_groot_sonic78sonicstate_ho5_official_full`
- running `openloop_smooth.py` on workstation GPU 1, against the old HE GR00T, with a bf16 backbone
- then serving the new model on workstation GPU 0, port 5561

The held-out episodes 91, 102, 1208, 1219, 1293 and 1300 are copied to
`/mnt/data/jihun/datasets/he_sonic78_nolimit_sonicstate_heldout6`. H100 checkpoint files are root-owned with mode 0600,
so copy them out through `docker exec <container> tar c`.

## Recipes (details and sources in the issue note)

| Policy    | Batch per update (micro × accum) | Updates | Key settings                                                                                                                                      |
| --------- | -------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| GR00T     | HE 32 × 1, Unitree 16 × 2        | 20K     | lr 1e-4 cosine, 5 % warmup via `policy.max_steps` (= `steps`); fixed processor                                                                    |
| ACT       | 8 × 1                            | 200K    | official batch 8; rest as before                                                                                                                  |
| Diffusion | 64 × 1                           | 200K    | AdamW (0.95, 0.999) wd 1e-6 no clip, cosine warmup 500, EMA (power 0.75), ResNet18 from scratch + GroupNorm, 240×320 crop 0.9, horizon 64/32 kept |
| Pi0.5     | HE 8 × 4, Unitree 4 × 8          | 30K     | fp32 weights + bf16 autocast, EMA 0.99, openpi aug, AdamW (0.9, 0.95) wd 1e-10 clip 1, `foreach=False`, cosine 2.5e-5 → 2.5e-6 warmup 1K updates  |
| VLA-JEPA  | 8 × 4                            | 30K     | fp32 weights, per-module LR (base 3e-5, Qwen 1e-5, head 1e-4, predictor 5e-4), warmup 5K, floor = 1/3 peak, MIN_MAX                               |
| MolmoAct2 | 8 × 2                            | 50K     | full fine-tune (user choice: paper real-world recipe), preset per-module LRs, warmup 200, decay to 10 %, official aug without blur                |
| FastWAM   | 8 × 2                            | 30K     | user choice: paper real-robot 30K at batch 16; AdamW (0.9, 0.95) wd 0.01 clip 1, cosine 1e-4 → 1e-6, 5 % warmup, bf16, gradient checkpointing     |

Timing (rough):

- GPU 6 should have finished around 2026-09-30 noon.
- GPU 0 runs about 8–9 days in total (MolmoAct2 and FastWAM speeds are not measured yet).
- None of the GPU 0 jobs fit GPU 6: other users hold ~28–33 GB there.
- A third 80 GB GPU would cut about 4 days; move the Unitree MolmoAct2 and FastWAM lines to its own queue.

## Incident 2026-09-30: torchcodec decoder leak froze the H100

**What happened.** The HE Diffusion full run leaked about 800 GB of host RAM in its 8 DataLoader workers (74–130 GB
each).

- It trained to step ~99.7K of 200K (checkpoint `050000` saved), then stalled for ~9.5 h while the host thrashed
  (1 TB RAM, no swap, load average 64).
- Every container and `docker exec` hung, and other users on the H100 were affected too.
- Stopping the GPU 6 container freed the memory, and Pi0.5 on GPU 0 resumed.

**Cause**, measured with a CPU-only probe (random HE samples with 2 frames, 60 GB-capped container):

- torchcodec's `VideoDecoderCache` holds 100 decoders by default. HE has **4064 video files** (one per episode), so
  random sampling evicts on almost every sample.
- Evicted decoders do not return their memory: +0.7 MB per sample after the cache is full.
- With the cache at 5000 (no eviction), memory grows only while new files are opened; the bound is about the file
  count × 2 MB per worker.
- PyAV stayed flat but was far too slow.
- Unitree has 114 files, so it evicts too, just less often.

**Fix in place, no code change:**

- All training containers run with `LEROBOT_VIDEO_DECODER_CACHE_SIZE=5000` and a memory cap, so a leak kills only its
  own job. See the containers table.
- Diffusion (HE and Unitree) was moved to the end of the GPU 6 queue, as the user asked.
- HE Diffusion reruns from scratch: rerunning costs ~7 h, and resuming from `050000` would only save ~1.7 h.

**Follow-up the same night: every Unitree job failed.** The Unitree videos are symlinks into `/source-datasets`
(`/mnt/data01/jhkim/datasets/unitreerobotics`), which the new containers did not mount.

- `LeRobotDataset`'s `reader.try_load()` then failed. The dataset fell back to a Hub download, which
  `HF_HUB_OFFLINE=1` turns into `OfflineModeIsEnabled`.
- GPU 6 was recreated with the mount.
- GPU 0 could not be, without killing Pi0.5. The second container `gpu0b` was added instead.

**Next time:** start every container with the cache variable, a memory cap, and `-v
/mnt/data01/jhkim/datasets/unitreerobotics:/source-datasets:ro`.

## Unitree data fix hold (2026-09-30 / 10-01)

Session `lerobot-e2` (Psi0/XR-1 work) found 266 corrupt right-hand state frames, with values up to ±3000, in 39
Unitree episodes. They are also baked into the stored statistics. The user decided to fix the datasets and recompute
the stats for all policies.

- The Unitree jobs not yet run are held in `queue_official/pending_unitree_after_datafix.txt`, with the GPU 0 and
  GPU 6 blocks. `lerobot-e2` appends them after the fix and restarts any runner that has exited.
- The user decided to delete the two Unitree models trained on the uncorrected data, `runs/{groot,act}_..._official_full`,
  on 2026-10-01. Their logs are kept as `logs/*.corruptdata`, and both are re-queued in the pending GPU 6 block.
- A watcher died in the old GPU 0 container: HE Pi0.5's `.exit` was never written, so GPU 0 sat idle from 07:22 to
  09:45 UTC on 2026-09-30. The exit was then written by hand.
- **The fix was applied on 2026-10-01 ~01:20 UTC by session `lerobot-15` (formerly `lerobot-e2`), with the user's
  approval:**
  - Datasets: `/run-output/datasets/{joint28,sonic78_nolimit,sonic78_nolimit_sonicstate,combined_sonicstate_1cam}`.
  - 267 frames in 40 episodes, state dims 24–27 only, linearly interpolated in time.
  - `observation.state` in `meta/stats.json` and the per-episode stats were recomputed.
  - Backups: `*.bak-corruptstate-20260930`. A re-scan finds no corrupt frames.
  - Both pending blocks were appended to the queues, and the GPU 6 runner was restarted. The Unitree GR00T rerun
    started at ~01:30 UTC on the corrected data.
  - The GPU 0 Unitree block runs after HE MolmoAct2 and FastWAM.
- The workstation copies trained on uncorrected Unitree data were deleted on 2026-10-01 at the user's request:
  `~/work/g1_models/groot_combined_sonicstate_1cam_ho5_full` (combined Unitree + HE) and
  `/mnt/data/jihun/g1_models/groot_sonic78nolimit_ho5_full`. See memory `h100-old-weights-deleted`.

## Open-loop smoothness of the finished HE models (2026-10-01)

`openloop_smooth.py` on the six held-out HE episodes (91, 102, 1208, 1219, 1293, 1300), random noise, on a workstation
RTX 3060. Units are per-dimension action std. "Disagree" is consecutive-chunk disagreement over the frames the chunks
share; "seam" is the jump at each switch. Replanning is every 12 frames unless noted.

| Model (new = official recipe)  | Tokens: disagree | Tokens: seam | Tokens: step (rec. 0.019) | Tokens: curvature (rec. 0.038) | Hands: disagree | Hands: seam |
| ------------------------------ | ---------------- | ------------ | ------------------------- | ------------------------------ | --------------- | ----------- |
| ACT new (200K)                 | **0.085**        | 0.087        | 0.017                     | 0.008                          | 0.101           | 0.120       |
| ACT old                        | 0.096            | 0.092        | 0.016                     | 0.014                          | 0.108           | 0.129       |
| Pi0.5 new, EMA, bf16           | 0.150            | 0.140        | 0.028                     | 0.032                          | 0.246           | 0.245       |
| GR00T new, bf16 backbone       | 0.170            | 0.139        | 0.023                     | 0.026                          | 0.242           | –           |
| Diffusion new, EMA             | 0.205            | 0.164        | 0.035                     | 0.043                          | 0.315           | 0.270       |
| GR00T old                      | 0.241            | 0.215        | 0.034                     | 0.049                          | 0.324           | –           |
| VLA-JEPA new, bf16, replan 4\* | 0.068            | 0.065        | 0.028                     | 0.037                          | 0.142           | 0.136       |

\* VLA-JEPA's chunk is 7 frames, so it replans every 4 frames and consecutive chunks share only 3 frames. Its numbers are
not comparable with the 12-frame rows, and on the robot it would need a new prediction every 0.13 s.

**Reading:**

- ACT is deterministic, and the retrain improved it slightly.
- Every stochastic generative policy, flow or diffusion, sits at 0.15–0.21 token disagreement, about 2× ACT, even with
  the official recipes.
- Within-chunk smoothness is fine for all of them. The common problem is sample variance at chunk switches, so the
  inference-side fixes (fixed noise seed, RTC, chunk blending) are worth trying on all of them, not only GR00T.

**Eval notes:**

- `ChunkPolicy` now always loads on the CPU first, then moves to the device, so 12 GB GPUs can run Pi0.5 and VLA-JEPA
  with a bf16 config.
- `openloop_smooth.py` takes `REPLAN=<frames>`.
- Local copies are in `/mnt/data/jihun/g1_models/he_*_official_full`. In those copies Pi0.5's `dtype` and VLA-JEPA's
  `torch_dtype` are set to `bfloat16`.
- **VLA-JEPA checkpoint bug:** the saved `model.safetensors` lacks `model.qwen.model.model.language_model.embed_tokens.weight`.
  It is tied to `lm_head`, so safetensors kept only one copy, and `from_pretrained` fails strict loading.
  - The local copy was patched by adding the key, cloned from `lm_head.weight`.
  - Any server or eval on the original checkpoint needs the same patch, or a loader fix.

## Incident 2026-10-01: containers lost their GPU

Around 2026-10-01 03:30 UTC both training containers lost GPU access: `nvidia-smi` in them gives
`Failed to initialize NVML: Unknown Error`, and `torch.cuda.is_available()` is False. This is the known Docker/NVIDIA
issue where a host-side cgroup reset (e.g. `systemctl daemon-reload`) drops a running container's device permissions.

- Processes that had already opened the GPU kept working. Unitree GR00T on GPU 6 trained on normally.
- New jobs fell back to the CPU without any error. LeRobot only warns "Device 'cuda' is not available. Switching to
  'cpu'".
  - HE MolmoAct2's smoke run passed (exit 0) on the CPU, then its full run ran ~2 h at ~50 s/step.

**Fix:**

- **Runner guard:** `official_queue.sh` now checks `torch.cuda.is_available()` before each job and stops the queue
  if CUDA is missing (`27ff296b`).
- **GPU 0:** `gpu0b` and the old `gpu0` container were removed. New container
  `jihun-lerobot-official-gpu0c-20261001` (same mounts, environment and 300 GB cap) runs the GPU 0 queue and the
  hourly prune loop. The CPU attempts are kept as `HE/logs/molmoact2_..._official_{smoke,full}.*.cpu`.
- **GPU 6:** the old runner was stopped, but Unitree GR00T keeps training in the old GPU-less container
  (`jihun-lerobot-he-official-gpu6-20260929`), which still holds its GPU. New container
  `jihun-lerobot-official-gpu6b-20261001` runs a watcher. When the GR00T log says "End of training", or stays quiet
  for 30 min, the watcher writes `logs/groot_..._official_full.exit` and starts the GPU 6 runner. Remove the old
  container after that.
- **Check:** `docker exec <c> nvidia-smi -L` in each container. Recreate any container that fails it.

## Disk: finished runs are pruned (2026-10-01)

`/mnt/data01` was at 96 %. `examples/g1_dex3_training/prune_finished_runs.sh` (on the H100 as
`/run-output/queue_official/prune_finished_runs.sh`) keeps only the final checkpoint's `pretrained_model/` and
`pretrained_model_ema/` of runs that exited 0 with "End of training". It deletes:

- the intermediate step folders
- the final step's resume-only files: optimizer state, `ema_state.pt`, RNG state

So **finished runs can no longer be resumed**, and their intermediate checkpoints are gone.

First pass, by hand, on HE ACT, Diffusion, GR00T and Pi0.5, plus the abandoned `diffusion_..._leaked` run: freed
289 GB, from 446 GB to 735 GB free.

An hourly loop (now in container `gpu0c`) prunes the seven policies' `*_sonic78sonicstate_ho5_official_full` runs on both
datasets as they finish. Its log is `queue_official/prune.log`. It never touches running jobs, failed jobs, or the
XR-1/Psi0 runs of session `lerobot-15`; those keep their intermediate checkpoints, 28 GB each.

## Pitfalls learned (keep)

- **The LeRobot scheduler steps per micro-batch.** `AcceleratorConfig` builds
  `Accelerator(step_scheduler_with_optimizer=False)`.
  - Under accumulation K, `steps`, warmup and decay are all micro-batch counts (× K). EMA updates per optimizer step.
  - The first HE Pi0.5 launch got this wrong. It was stopped; its log is kept as
    `HE/logs/pi05_..._official_full_badsched.log`.
  - The logged `lr` is a window average.
- **Presets overwrite `optimizer`/`scheduler`.** With `use_policy_training_preset=true` they come from the policy config.
  - To set `foreach` or other betas, set the preset to false and give both explicitly.
  - With presets off, `policy.parameters()` is used instead of `get_optim_params()`, so per-module groups are lost.
    VLA-JEPA and MolmoAct2 therefore keep the preset.
- **Batch 32 in fp32 OOMs** on 80 GB for Pi0.5 (in multi-tensor Adam) and VLA-JEPA. Those configs are kept as
  `HE/configs/*_oom_b32.json`.
- **GR00T's warmup preset uses `policy.max_steps`**, not `--steps`: set `max_steps` = `steps`.
- **The containers have no `pkill`/`pgrep`/`ps`.** To stop a job, find PIDs through `/proc/*/cmdline` with the venv's
  Python inside the container, and stop the queue loop before the trainer and its `multiprocessing.spawn` workers.
  `docker top <c> -eo pid,etime,args` works for listing.
- **The earlier `openloop_smooth` baselines are gone from the H100.** All older runs there were deleted on 2026-09-29,
  see memory `h100-old-weights-deleted`. Local copies exist only for:
  - HE GR00T and `groot_sonic78nolimit_ho5_full` in `/mnt/data/jihun/g1_models/`
  - HE ACT and the combined GR00T in `~/work/g1_models/`

  The other policies' old numbers exist only in the docs.

- H100 rules (memory `g1-real-robot-eval-setup`): nothing on the host itself, only containers and mounts. Write to
  `/mnt/data01` through a container, for example `git archive HEAD | ssh h100 docker run -i --rm -v …:/code … tar x`.

## Status check

```bash
ssh h100 'docker exec jihun-lerobot-official-gpu0c-20261001 bash -lc "cd /run-output; \
  for e in humanoid_everyday_g1_20260923/logs/*official*.exit logs/*official*.exit; do echo \"\$e: \$(cat \$e)\"; done; \
  for l in humanoid_everyday_g1_20260923/logs/*official_full.log logs/*official_full.log; do \
    [ -f \${l%.log}.exit ] || echo \"\$l: \$(tail -c 400 \$l | tr \"\\r\" \"\\n\" | grep -a Training: | tail -1 | cut -c1-80)\"; done"; \
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 0,6'
```

Checkpoints land in `<root>/runs/<job>/checkpoints/<step>/pretrained_model`, with EMA copies next to them for Pi0.5
and Diffusion. A run's folder appears at its first checkpoint.

## Next steps

State at 2026-10-01 06:10 UTC:

- HE: GR00T, ACT, Diffusion, Pi0.5 and VLA-JEPA are done and pruned. HE MolmoAct2 full started on GPU 0 after its
  GPU smoke run (0.8 s per micro-step, 50 GB, ~22 h), then HE FastWAM.
- Unitree GR00T is ~52 % on GPU 6, ~5 h left. Then Unitree ACT and Diffusion on GPU 6. The Unitree Pi0.5, VLA-JEPA,
  MolmoAct2 and FastWAM block runs on GPU 0 after HE FastWAM.

1. **Confirm the GPU 6 handover.** When Unitree GR00T ends, `gpu6b`'s watcher should write its `.exit` and start
   Unitree ACT on the GPU. Then `docker rm` the old GPU 6 container.
2. **MolmoAct2 logs `grdn:0.000`.** Its optimizer clips per group internally, so the train loop reports no norm.
   Check that the loss keeps falling in the full run.
3. **The first FastWAM smoke run** (GPU 0, after HE MolmoAct2): its memory is unmeasured.
4. **Check every `.exit` file.** `skip` means the smoke run failed: read the smoke log, fix the config, delete both
   `.exit` files.
5. **Judge each finished model** with `examples/g1_dex3_training/openloop_smooth.py` (tokens and hands, in std units)
   before any sim or robot run.
   - New HE GR00T: consecutive-chunk disagreement is 0.17 (old 0.24, ACT 0.10). Within-chunk smoothness is now at or
     below the recording's.
   - Next for GR00T, open loop first: a fixed noise seed, RTC (native GR00T overlap guidance), then seed plus chunk
     blending on the robot.
   - Not yet evaluated: HE ACT, Diffusion (EMA), Pi0.5 (EMA), VLA-JEPA.
6. Then sim (planner start, table 25 cm), then robot, per the first-runs note.
7. GPU 0 is the long pole (several days). If another 80 GB GPU frees up, move the Unitree MolmoAct2 and FastWAM lines to
   its own queue and container, set up like `gpu0c`.
8. Optional upstream reports:
   - the torchcodec cache default of 100 leaks on datasets with many video files
   - the GR00T processor fallback for Hub ids (fixed on this branch)
   - LeRobot silently falling back to the CPU when CUDA disappears
9. The branch is not pushed.
