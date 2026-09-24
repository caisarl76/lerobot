# G1 Dex3 training execution plan

**Goal:** Train the requested VLA and world-model policies on all 13 specified Unitree G1 Dex3 datasets together, in 28D joint-action and 78D SONIC-token-plus-hand formats.

**Architecture:** Reuse verified source datasets without changing them. Create combined LeRobot datasets with source episode provenance and shared video files. Encode the 78D targets with the official GEAR-SONIC v1.1 G1-mode encoder. Run joint28 on H100 GPU 3 and sonic78 on GPU 7, in their existing dedicated Docker containers.

**Constraints:** Host Hugging Face cache `/mnt/data01/huggingface` mounts at `/root/.cache/huggingface`; all model outputs live under `/mnt/data01/jhkim/model_weight`. Preserve only the final verified model and its loading configuration/processors after each successful run. No changes to unrelated jobs, datasets, caches, or checkpoints. Preparation uses CPU while GPU 3 is serving. All seven named models remain in scope unless the user selects a different model subset. User explicitly requires all installation and modifications on H100 to occur through Docker containers; never install host packages or directly edit the H100 host filesystem. Transfer updates through `docker exec -i` and write only the task's mounted paths.

**Training budget:** User selected 40,000 steps per model, applied to each of the two action-format runs. Smoke tests remain 20 steps. This supersedes the initial 100k baseline plan. Full runs include all episodes in their sampling pool; 40k steps is not a promise of visiting every frame.

**Approved recovery exception (2026-09-23):** The user selected exclusion of only PickCharger source episode 57, merged episode 2202, after the video audit below. Both action formats retain all 13 collections, including the other 199 PickCharger episodes. The training pool is 3,151 episodes / 2,586,948 rows. Original and derived dataset files remain unchanged; the exclusion is applied through `dataset.exclude_episodes=[2202]`.

**Action contracts:** 28D = left arm7, right arm7, left hand7, right hand7, preserving the dataset's asymmetric finger ordering. 78D = actual SONIC64 + left hand7 + right hand7. Source actions lack legs, waist, and root orientation; record the fixed nominal standing reference assumption explicitly. Do not use SMPL mode for joint-only data, invent motion tokens, or add the live adapter's entry/settling frames to training episodes. Sample future references at official 50 Hz with step-five windows, aligned to the source's 30 Hz video timestamps and bounded within each episode.

## Execution

- [x] Inventory all 13 source datasets; verify rows, episodes, cameras, tasks, finite 28D actions, source metadata and revisions.
- [x] Prepare CPU-only Docker environment from pinned repository/image with Python 3.12 and locked dependencies. Record package and source versions.
- [x] Build combined 28D dataset with all episodes and tasks, unique global indices, reused videos, and correctly aggregated statistics. Smoke selection includes episodes from every source.
- [x] Implement and test SONIC conversion against the official encoder input configuration and native reference ordering; verify real encoder output shape, finiteness, quantization, timing, hand order, and episode boundaries.
- [x] Build combined 78D dataset and verify every episode plus token provenance. Preserve 30 Hz image/action alignment.
- [x] Prepare per-model configurations (ACT, diffusion, Pi0.5, GR00T N1.7, MolmoAct2, VLA-JEPA, FastWAM) for both formats. World model training must retain its prediction objective. Disable gripper-specific action transforms for Dex3.
- [x] Coordinate GPU 3 release with user. Recheck authoritative GPU process state before launching training.
- [x] For each model/format run subset forward/backward/optimizer/save/reload smoke tests with finite losses and correct action shape.
- [ ] Train on all episodes together, record dataset coverage and losses, verify final weights reload, then remove intermediate checkpoints of these runs only.
- [ ] Audit all requested runs and final model artifacts before claiming completion.

## Current evidence

- Local source HEAD: `240ea44c`, worktree `/tmp/lerobot-g1-dex3-training`.
- H100 GPU 3 UUID: `GPU-2a173997-b199-6012-14f7-b2ac9fdd7b60`; user released GPU 3 and the dedicated training container is using it.
- Cache entries for the 13 datasets contain revision markers only. Existing source copies are at `/mnt/data01/jhkim/datasets/unitreerobotics`, 173 GiB including extra single-camera derivatives; contents under audit.
- Approximately 479 GiB free on data disk at initial inspection. Avoid video duplication.
- LeRobot multi-dataset factory is explicitly disabled; use one merged dataset for training.
- Existing ACT image is Python 3.10; this repository requires Python 3.12. Prepare a separate environment.
- Native trainer supports `save_freq=0` for final-only saves. For long runs, retained resumable checkpoints may be used and pruned only after final reload validation.
- CPU preparation container `jihun-lerobot-g1-dex3-prep-20260922` is running, no GPU DeviceRequests. Source at `/workspace/lerobot`, outputs `/run-output`, original datasets `/source-datasets:ro`, SONIC `/sonic-model:ro`. Python3.12.13 / PyTorch2.11.0+cu128 installed using locked uv extras in `/run-output/environment/venv`.
- Six datasets have four cameras; seven picking datasets have only stereo head cameras and HWC shapes mislabeled CHW. Training projection explicitly keeps the common head stereo pair, normalizes shape metadata, and preserves every episode/frame/action. Strict inventory checks discovered and now cover this mismatch.
- Inventory: 3,152 episodes, 2,587,515 frames. Camera axis names in picking data are `height,width,channel` (singular); normalize by verified RGB shape.
- Actual source defects: ToastedBread episode396 points to the previous data shard; episodes397–417 ranges are shard-local although parquet indices are global. All seven picking datasets have a copied red-cup episode instruction while their task tables and row task indices correctly describe their objects. Derived metadata repairs are inferred only after verifying every payload row, recorded in provenance, and never written into original datasets.
- SONIC CPU check encoded 1,585 source-aligned samples across complete episode0 references from all13 sources in 3.81s, with finite78D output and real quantized tokens. Encoder SHA `fb97de22819b2057b41459802128d91723d91a25f0ad73e7bfc41a9cf8365bae`; config SHA `4a67713b310932e50aca81f19188c8d76013148e98b15c8b5bbea995f12e59f0`; report `/run-output/sonic-encoder-smoke.json`. This proves encoder/data contract, not model training.
- Focused Docker tests passed25 cases plus8subtests before the episode task repair; rerun updated tests before claiming final preparation. Existing native aggregation regression suite:14passed in1159.17s.
- Policy adaptation remains necessary: Pi0.5 has default32D action projections, and MolmoAct2 explicitly fixes32D in current code. Their78D runs require verified projection adaptation, not silently truncated targets or random fallback initialization.

### Execution recorded at the initial commit (historical)

- User explicitly released GPU3. Immediately before launch:1MiB/81559MiB,0%utilization,no compute processes. Dedicated container `jihun-lerobot-g1-dex3-train-gpu3-20260922` exposes only GPU3 UUID; PyTorch sees one H100 and CUDA tensor check passed. All setup and writes remain inside Docker.
- Full28D preparation passed `/run-output/logs/prepare-joint28-r5.exit=0`; provenance at `/run-output/datasets/joint28/meta/provenance.json`. All3152episodes/2587515rows payload-equivalent after allowed reindexing;757MiB data, videos symlinked. Source robot labels normalized only in projections. Missing global statistics counts discovered; action/state/timestamp statistics recomputed from actual rows, including exact per-source quantiles.
- Native reader decoded52samples(first/last frames in26episodes across13sources), correct chunk100x28/state28/twoRGBimages. `/run-output/joint28-reader-smoke.json`,exit0.
- Full78D conversion running CPU at ~600frames/s, `/run-output/logs/prepare-sonic78-r2.log`; original launch wrong encoder filename failed before writes. Correct model `/sonic-model/model_encoder.onnx`. Publishes final only on full verification.
- Pi05 opt-in78D action projections implemented; loading failures now propagate.45native Docker tests passed including MEM/reload, `/run-output/logs/pi05-action-dimension-tests.log`.
- MolmoAct2 explicit continuous78D adaptation implemented and reviewed; 134 Docker tests passed, with one skip. Both adaptations preserve backbone weights, reset only necessary projections, and must retain learned heads on final reload.
- ACT and diffusion joint28 completed 20-step smokes and strict checkpoint reload, finite forward loss, prediction, and postprocessing checks. Their optimizer states were removed after verification. Final model reports are stored in each run directory.
- Pi0.5, GR00T, and MolmoAct2 joint28 completed 20-step smokes and strict reload/prediction checks. Predicted shapes: Pi0.5 (1,50,28), GR00T (1,40,28), MolmoAct2 (1,30,28). Their optimizer states were removed only after verification. MolmoAct2 native group clipping leaves the generic logged grad norm at zero; saved nonzero optimizer moments and step20 were verified separately in `/run-output/molmoact2-smoke-optimizer-check.json`. No full 40,000-step run has started yet.
- All 28 smoke/full configuration files parse and validate their feature contracts; full configs target 40,000 steps. VLA-JEPA retains world-model training, and FastWAM retains both video and action objectives.
- VLA-JEPA fine-tuning now rebuilds processors from current embodiment settings and dataset stats, preventing the base LIBERO processor from binarizing G1 wrist channel 6. Resume/inference preserves saved processors. The 25 targeted Docker processor tests passed.

## Verification risks

- Shared video references must resolve correctly from merged and converted datasets.
- Metadata counts alone cannot prove parquet or video completeness.
- Source names contain asymmetric left/right finger ordering.
- SONIC future windows must never cross episode boundaries; non-active encoder features must be zero.
- Pretrained action projections may require explicit resizing for 78D; successful model construction alone is not a smoke test.

### Training launch and retention gates

- `train_full_queue.py` checks all14 verified smoke reports and each full config (40000steps, all episodes, no evaluation split, correct policy/state/action dimensions) before launching any fullrun. Sequential GPU subprocesses stop on training or verification errors. A process lock prevents duplicate queues; each run requires80GiB free.
- `finalize_baseline.py` supports allseven policies, strict saved-model reload, finite weights/loss, native dataset sample, prediction and saved postprocessing, and SHA256 recording. Numeric intermediate checkpoints and final optimizer state are pruned only after success.
- `prune_smoke.py` removes only the matching smoke checkpoints after the full model is verified, with exact run names, paths, hash, report and exit-marker checks. Reports/logs survive. Full-run retention and queue tests:24passed in Docker.
- VLA-JEPA first real load correctly rejected four obsolete predictor weights. Git commit `d451fe4f` removed the never-used state encoder and made unused extrinsics encoding conditional. Compatibility now discards only these exact four absent keys with warnings; unknown keys and missing active weights still fail. Tests prove nonzero world-model loss and predictor gradients. Combined suite62passed/9skipped, then11 focusedcases including `reinit_modules=None` passed. Actual GPU smoke retry is running.
- Pi0.5 and FastWAM saved base processors were inspected: neither contains an unwanted G1 gripper binarization/toggle step. FastWAM base weights are revision-pinned. Wan VAE and UMT5 weights were downloaded without duplicating the unused original-format T5/DiT files.

### Recovery audit on 2026-09-23 (KST)

- Workstation access: `ssh h100`, host `mncsvr11`. Both training containers are running, but their original queues exited with code 1. Both ACT full runs stopped after step 1,139 / 40,000 in video decoding; no full-run checkpoints exist because `save_freq=0` was configured. Recovery must restart these two runs from step zero.
- Containers: `jihun-lerobot-g1-dex3-train-gpu3-20260922` (joint28), `jihun-lerobot-g1-dex3-train-gpu7-20260922` (sonic78). Shared run root: `/mnt/data01/jhkim/model_weight/g1_dex3_20260922` on the host, `/run-output` in Docker. Shared source: `/workspace/lerobot` in Docker. The deployed source has additional changes beyond this checkout, so transfer only reviewed task files.
- Both derived datasets contain 3,152 episodes / 2,587,515 rows before the approved runtime exclusion. All 14 saved smoke reports are verified at step 20 with the appropriate 28D/78D action shape. VLA-JEPA reports nonzero world-model loss; FastWAM reports both video and action losses.
- Deterministic CPU replay: seed 1000, batch size 32, sampler batch index 1140, merged row 1762531. PyAV fails on the right-head video at requested timestamp 1422.2666673024494 s. Sequential decoding and single-frame probes isolate missing source video frames 42664 and 42669 (episode-local frames 561 and 566 of source episode 57).
- A complete sequential decode audit covered all 114 unique shared training videos. Exactly one file failed: `G1_Dex3_PickCharger_Dataset/videos/observation.images.cam_right_high/chunk-000/file-000.mp4`; 61,202 of its 61,204 frames decoded. The other 113 files decoded all advertised frames. Reports are in `/run-output/recovery/video-audit-20260923/`; `episode-impact.json` maps the failures to merged episode 2202, task `Put the charger into the plate.`
- The bad file's SHA-256 is `658fa04b9ec04c0ffd19c1e0c9aa12716280f674c9a7c444f6896f8cfbe1eb4a`, identical to the published Hugging Face LFS object at revision `4c3c42844256552f2f95f2ce9a48e60943270507`. The original per-episode video at revision `2f672bc11876a017514cf4d1e69a6329df525f7a` also fails to decode completely, despite exact action/state/timestamp parity. Re-downloading does not repair the data.
- TorchCodec reads the original failing sample and the first valid frame of the following episode, but also rejects the truly missing frames. Thus switching decoders alone is insufficient. The approved recovery combines the episode exclusion with `video_backend="torchcodec"`, preserving valid neighboring episodes that PyAV's backward seek can otherwise fail on.
- Candidate-reader verification passed 180 boundary samples: ACT, VLA-JEPA, and FastWAM, in both action formats, across all 13 sources and both neighboring episodes. The tests verify the retained 3,151 episodes / 2,586,948 rows, action/state widths, finite actions, and native temporal image windows. Report: `/run-output/recovery/candidate-reader-validation.json`. These are offline reader checks, not completed training or hardware validation.
- The queue's `--exclude-episodes` argument must exactly match every selected full configuration. Omitting it preserves the original no-exclusion requirement. `--action-space` supports one queue per GPU; omitting it acquires both format locks and processes both formats sequentially.

### Recovery deployment and restart on 2026-09-23 (KST)

- Deployed only the reviewed queue file into the shared Docker source; SHA-256 `cc44fdc7be618ab60777297f7e60e66db312a8593ea3e2bd6ffbe043c3b476e0`. The 17 queue tests pass, including action-format selection, exact approved exclusions, validation before launching, and overlapping queue locks. Standards and specification reviews found no issues. Other workstation source changes were preserved.
- Updated all 14 full-run configurations to exclude only merged episode 2202 and use TorchCodec. All retain 40,000 steps and their original action contracts. Enabled checkpoints every 1,000 steps for ACT, 5,000 for diffusion, and 20,000 for the remaining models. Final verification and retention gates remain unchanged.
- Original configurations, queue source, and failed-run logs/exit markers are preserved under `/run-output/recovery/restart-20260922T155217Z/`, with checksums and exact changes in `manifest.json`. No dataset files were edited or deleted.
- Relaunched each queue with `/run-output/environment/uv run --no-project /run-output/environment/venv/bin/python -u -m examples.g1_dex3_training.train_full_queue --run-root /run-output --action-space <joint28|sonic78> --exclude-episodes 2202`. Queue logs are `/run-output/logs/full-training-<space>.log`; per-policy logs remain `/run-output/logs/<policy>_<space>_full.log`. A corresponding `.exit` marker is written when a process finishes.
- The first joint28 relaunch exposed a separate container-runtime issue: host GPU 3 was healthy, but container NVML returned `Unknown Error`, PyTorch reported zero CUDA devices, and the trainer fell back to CPU. Restarted only the dedicated GPU 3 container, then verified its expected UUID and a real CUDA tensor allocation. Archived the nine-step CPU attempt under the recovery backup's `cpu-fallback-attempt/` directory before relaunching joint28. GPU 7 and unrelated jobs were not interrupted.
- Verified at 2026-09-23 01:08:13 KST: joint28 ACT reached step 2,707 and sonic78 ACT reached step 4,822, both beyond the previous failure at 1,139, with finite logged losses and live queue/trainer processes. Both logs confirm the filtered pool of 3,151 episodes / 2,586,948 rows. Both step-1,000 checkpoints contain model weights, processors, optimizer and RNG state; saved training configs confirm CUDA, correct 28D/78D outputs, TorchCodec, and exclusion `[2202]`. Evidence: `/run-output/recovery/restart-20260922T155217Z/resume-verification.json`. This verifies healthy resumption and checkpoint creation, not completion of the 40,000-step runs or final model reload validation. No simulation or robot hardware was run.
