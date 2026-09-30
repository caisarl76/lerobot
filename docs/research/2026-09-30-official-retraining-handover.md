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

| Commit     | Change                                                                                                                                                                                                                                          |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dfe36b45` | GR00T processor: without checkpoint sidecars (Hub id), use the base checkpoint's processor values: q01/q99, state dropout 0.2, letterbox + crop 0.95 + ColorJitter (new), embodiment map filled so `new_embodiment` is slot 10, not 0. Test `tests/policies/groot/test_groot_new_embodiment_defaults.py`. |
| `b01a01db` | VLA-JEPA `optimizer_module_lrs` (per-submodule peak LR). Test `tests/policies/vla_jepa/test_optim_groups.py`.                                                                                                                                  |
| `ceb40b77` | `AdamWConfig.foreach` (False avoids multi-tensor temporaries; needed for fp32 Pi0.5 + EMA on 80 GB).                                                                                                                                             |
| `4b55559b` | `examples/g1_dex3_training/official_queue.sh`: per-GPU queue runner.                                                                                                                                                                           |
| `65893b97`, `69719330`, `c44b3242` | Issue-note sections: comparisons, retraining, full matrix.                                                                                                                                                      |

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

**Containers** (image `4cbe2a3f7fc6`, venv `/run-output/environment/venv`, HF cache mounted, `HF_HUB_OFFLINE=1`):

| Container                                 | GPU | Code mounted at `/workspace/lerobot`                          |
| ----------------------------------------- | --- | ------------------------------------------------------------- |
| `jihun-lerobot-he-official-gpu6-20260929` | 6   | `/mnt/data01/jhkim/code/lerobot-g1-groot-fix-20260929` (`dfe36b45`) |
| `jihun-lerobot-he-official-gpu0-20260929` | 0   | `/mnt/data01/jhkim/code/lerobot-g1-official-20260929-ceb40b77` |

Each code directory is a `git archive` export of the named commit. They are not git checkouts.

**Queues:**

- Runner: `/run-output/queue_official/official_queue.sh` (the same file as in the repo).
- Queue files: `gpu6.txt` and `gpu0.txt`, runner logs `gpu{6,0}.runner.log`.
- Each line is `<root>/<job>`; the file is re-read after every job, so lines can be appended.
- A job is done when `<root>/logs/<job>.exit` exists.
- A `_full` job is skipped (exit `skip`) if its `_smoke` job failed.

**Order and state**, as last seen on 2026-09-29 09:23 UTC (H100 clock):

- **GPU 6:**
  - `HE groot` full: running at the time, ETA ~10:10. It was launched by hand, outside the queue.
  - Then the queue: HE act smoke → full, HE diffusion smoke → full, Unitree groot, act, diffusion (smoke → full each).
  - The queue started when `HE/logs/groot_..._official_full.exit` appeared.
- **GPU 0:**
  - A hand-launched loop: HE pi05 full (running, ETA ~21:30), then HE vla_jepa full.
  - The queue waits for `HE/logs/vla_jepa_..._official_full.exit`, then runs, each smoke → full: Unitree pi05, vla_jepa;
    HE molmoact2, fastwam; Unitree molmoact2, fastwam.
- **Unchecked at handover:** at 2026-09-30 00:31 UTC GPU 0 showed **80.9 GB used and 0 % utilization**, GPU 6 40.8 GB at
  0 %. This session could not look further because its tool permission check failed.
  - HE vla_jepa should have been running on GPU 0 then (52 GB in its smoke run). A full GPU at 0 % may be a hang or
    OOM. **Check this first.**

## Recipes (details and sources in the issue note)

| Policy    | Batch per update (micro × accum)  | Updates | Key settings                                                                                                                                     |
| --------- | --------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| GR00T     | HE 32 × 1, Unitree 16 × 2          | 20K     | lr 1e-4 cosine, 5 % warmup via `policy.max_steps` (= `steps`); fixed processor                                                                   |
| ACT       | 8 × 1                             | 200K    | official batch 8; rest as before                                                                                                                 |
| Diffusion | 64 × 1                            | 200K    | AdamW (0.95, 0.999) wd 1e-6 no clip, cosine warmup 500, EMA (power 0.75), ResNet18 from scratch + GroupNorm, 240×320 crop 0.9, horizon 64/32 kept |
| Pi0.5     | HE 8 × 4, Unitree 4 × 8            | 30K     | fp32 weights + bf16 autocast, EMA 0.99, openpi aug, AdamW (0.9, 0.95) wd 1e-10 clip 1, `foreach=False`, cosine 2.5e-5 → 2.5e-6 warmup 1K updates  |
| VLA-JEPA  | 8 × 4                             | 30K     | fp32 weights, per-module LR (base 3e-5, Qwen 1e-5, head 1e-4, predictor 5e-4), warmup 5K, floor = 1/3 peak, MIN_MAX                               |
| MolmoAct2 | 8 × 2                             | 50K     | full fine-tune (user choice: paper real-world recipe), preset per-module LRs, warmup 200, decay to 10 %, official aug without blur             |
| FastWAM   | 8 × 2                             | 30K     | user choice: paper real-robot 30K at batch 16; AdamW (0.9, 0.95) wd 0.01 clip 1, cosine 1e-4 → 1e-6, 5 % warmup, bf16, gradient checkpointing   |

Timing (rough):

- GPU 6 should have finished around 2026-09-30 noon.
- GPU 0 runs about 8–9 days in total (MolmoAct2 and FastWAM speeds are not measured yet).
- None of the GPU 0 jobs fit GPU 6: other users hold ~28–33 GB there.
- A third 80 GB GPU would cut about 4 days; move the Unitree MolmoAct2 and FastWAM lines to its own queue.

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
ssh h100 'docker exec jihun-lerobot-he-official-gpu0-20260929 bash -lc "cd /run-output; \
  for e in humanoid_everyday_g1_20260923/logs/*official*.exit logs/*official*.exit; do echo \"\$e: \$(cat \$e)\"; done; \
  for l in humanoid_everyday_g1_20260923/logs/*official_full.log logs/*official_full.log; do \
    [ -f \${l%.log}.exit ] || echo \"\$l: \$(tail -c 400 \$l | tr \"\\r\" \"\\n\" | grep -a Training: | tail -1 | cut -c1-80)\"; done"; \
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 0,6'
```

Checkpoints land in `<root>/runs/<job>/checkpoints/<step>/pretrained_model`, with EMA copies next to them for Pi0.5
and Diffusion. A run's folder appears at its first checkpoint.

## Next steps

1. **Check GPU 0** (0 % utilization with full memory at 00:31 UTC). Look at the HE vla_jepa log and `gpu0.runner.log`.
   Fix and re-queue: delete a failed job's `.exit` file and the queue reruns it.
2. **Check every `.exit` file.** `skip` means the smoke run failed: read the smoke log, fix the config, delete both
   `.exit` files.
3. **Judge each finished model** with `examples/g1_dex3_training/openloop_smooth.py` (tokens and hands, in std units)
   before any sim or robot run.
   - Target for GR00T: consecutive-chunk disagreement near ACT's ~0.10, down from 0.24. Compare with the old HE
     GR00T's local copy.
4. Then sim (planner start, table 25 cm), then robot, per the first-runs note.
5. Optionally move the Unitree MolmoAct2 and FastWAM lines to a third GPU if one frees up.
