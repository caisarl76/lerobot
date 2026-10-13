# G1 data efficiency on a balanced own dataset: design

Date: 2026-10-13. Status: approved in conversation (collection, training grid, evaluation); written spec under review.

## Goal

Show whether pretraining on Humanoid Everyday (HE) reduces how many of our own teleoperation episodes a VLA needs to
do a task on the real G1. The measure is **robot success rate against the number of own episodes per task**, for a
start from the public weights (`base`) and from the HE checkpoints (`he`), fine-tuned on the same episodes.

Example of the intended result: `base` with 50 episodes per task succeeds 8/10, with 25 episodes 5/10; `he` with 25
episodes succeeds 8/10, i.e. HE pretraining halves the episodes needed.

## Why a new dataset

The first robot trials of the earlier grid (`docs/research/2026-10-09-g1-pretrain-finetune-data-efficiency.md`,
robot runs 42–56 on 2026-10-13) could not answer this:

- **No baseline.** Nobody knew what `base` can do with all of our data, so a failure of a pretrained model at low
  data had nothing to be compared with.
- **Unbalanced data.** The 86-episode WBT multi-task draft has 1–12 episodes per task. The test task "Place the apple
  from the tray onto the plate." has 2 episodes (one training, one held out), both left-handed, and none in the 25 %
  split. On the robot both GR00T `he` 25 % and `base` 100 % closed the right hand (the hand of the 6 "apple plate →
  basket" episodes) and never reached for the apple: 0/7 recorded runs, a data gap rather than a model difference.

## Decisions (user, 2026-10-13)

| topic        | decision                                                                       |
| ------------ | ------------------------------------------------------------------------------ |
| episodes     | 50 per task                                                                    |
| tasks        | 2 right-hand pick-and-place tasks                                              |
| VLAs         | GR00T and Pi0.5                                                                |
| layout       | small grid: 5 marked positions per task                                        |
| structure    | staged, with a baseline gate (approach A)                                      |
| subsets      | 10 / 25 / 50 episodes per task, nested, stratified 2 / 5 / 10 per position     |
| budget       | same sample budget at every subset size                                        |
| scoring      | blind (model codes), unmasked after scoring                                    |

## 1. Collection protocol

- **Tasks** (one fixed prompt each, right hand):
  - T1 `Place the apple from the plate into the basket.`
  - T2 `Place the yellow block into the gray cup.`
- **Recording**: the GR00T-WBC teleop setup of the earlier WBT recordings (50 Hz, SONIC motion tokens, D435
  `ego_view` 640×480), so the sonic78 conversion (state 28 = arms + Dex3 hands by joint name; action 78 = motion
  token 64 + hand targets 14; camera renamed `observation.images.egocentric`) applies unchanged.
- **Scene**: the table, table distance, camera and lighting of the robot trials; table distance as the streamer's
  planner start expects.
- **Positions**: 5 tape marks per task for the **picked object** (plate with apple; yellow block), ~10 cm apart within
  right-arm reach. The **target** (basket; cup) stays on one fixed mark. Marks are photographed and numbered p1–p5.
- **Episodes**: 10 per position, recorded **round-robin** (p1 … p5, p1 …) so session breaks and operator drift spread
  over positions.
- **Start**: every episode starts in the robot's rest stance with **open hands**, the state the policy sees on the
  robot after the streamer's `--hold-rest-s`.
- **Content**: reach, grasp, place, return to rest, stop. Failed or fumbled demonstrations are re-recorded, not kept.
- **Labels**: each episode stores task and position id (annotation or episode metadata), for stratified subsets.
- **Not used**: the 86 earlier WBT multi-task episodes stay out of these fine-tunes, so "N episodes" means N.
- The user collects the data and posts the dataset location in the session.

## 2. Training grid

- **Subsets**: 10 / 25 / 50 episodes per task, nested (10 ⊂ 25 ⊂ 50), stratified 2 / 5 / 10 per position, drawn
  with a fixed seed (not collection order).
- **One multi-task model per cell**: each fine-tune trains on both tasks at its size (20 / 50 / 100 episodes total).
- **Inits**:
  - `base`: public weights of the VLA.
  - `he`: HE official checkpoints `*_sonic78sonicstate_ho5_official_full` (Pi0.5: EMA weights), retargeted to our
    normalization stats with `retarget_init_checkpoint.py`.
- **Recipes**: those of the earlier dex grid, with the same sample budget at every size (smaller sizes see more
  passes):
  - GR00T: batch 32 × 5,000 steps = 160K samples.
  - Pi0.5: micro 4 × accumulation 8 × 60,000 steps = 240K samples, EMA (robot uses `pretrained_model_ema`).
- **Normalization**: one stats file from the 50-episode set, used by every cell.
- **Cells**: 3 sizes × 2 inits × 2 VLAs = 12 fine-tunes.
- **Stages**:
  - Stage 1: `base` 50 for GR00T and Pi0.5 (2 runs).
  - Stage 2, after the gate (section 3): the other 5 cells per VLA (10 runs).
- **Hosts**: h100_174 GPUs 4 and 5 first, then h100 GPUs 0 and 6; never h100 GPU 2. Per-GPU `official_queue.sh`
  queues with `.exit` files, training in containers only (`LEROBOT_VIDEO_DECODER_CACHE_SIZE=5000`, `--memory` cap).
- **To the robot**: each finished run is uploaded to a private `mncai` HF repo (`model_registry.py upload`) and
  downloaded to the workstation with xet disabled; Pi0.5 also uploads `pretrained_model_ema/`. The model registry is
  re-rendered.

## 3. Evaluation protocol

- **Prerequisite**: find and fix the `right_wrist_roll` (arm joint 11) speed spike at the planner → policy handoff.
  It ended two runs by watchdog in the first ticks (old run42 at 6.8 rad/s, run48 at 6.6 rad/s) and reaches 2–4 rad/s
  in other runs during "blend in". Fixed and checked on the robot before any scored trial.
- **Robot setting** (every cell): `g1_groot_real_run.sh` reference: noise 0, `--chunk-blend-s 0.3`, planner start
  and end, `POLICY_FPS=50`, `--max-token-step 0.05`, `--hold-rest-s 1.5`, server on workstation GPU 1 port 5560
  (Pi0.5: `POLICY_SUBDIR=pretrained_model_ema BACKBONE_DTYPE=`). Episode limit: 2 × the 95th-percentile demo length,
  fixed once the data exists.
- **Trials**: 10 per task per cell, 2 per position, fixed order p1 … p5, p1 … p5.
- **Outcome per trial**:
  - success: the object ends in the basket / cup;
  - partial: grasped, not placed (recorded; the headline counts successes only);
  - fail: no grasp, a drop, or a watchdog stop during the task;
  - void: a watchdog stop within the first 0.5 s of the episode (handoff, not policy); rerun once and noted.
- **Blind scoring**: servers are started under codes (M1, M2, …); the code → model map is kept by the session and
  unmasked after the cell is scored.
- **Order**:
  - Gate: `base` 50 for GR00T and Pi0.5. A task continues for a VLA only if `base` 50 scores ≥ 6/10 on it;
    otherwise the task or data is fixed first (more episodes, easier layout) before stage 2.
  - Stage 2: `he` 50, then 25 and 10 for both inits. A cell is skipped only if the larger size of the same curve
    (VLA, init, task) scored 0/10; it is then recorded as 0/10, untested.
- **Records**: run names `runNN_<vla>_<code><size>_<task>_p<pos>_t<k>` (code until unmasked); one CSV row per trial
  (run, VLA, init, size, task, position, outcome, note); PC2 logs pulled for smoothness and watchdog counts
  (`robot_run_smoothness.py`).
- **Analysis**: success rate with 95 % Wilson intervals per (VLA, task, init, size).
  - Headline: the smallest size at which `he` matches `base` 50, and `he` vs `base` at each size.
  - With 10 trials only large gaps are clear (8/10 vs 3/10 is; 6/10 vs 4/10 is not). If a headline comparison
    falls inside the intervals, 10 more trials are run for that cell only.
- **Write-up**: a results doc in `docs/research/` with the data table, curves and runs, linked from `AGENTS.md`.

## Dependencies

The tools exist on unmerged branches and must reach the implementation branch first:

- `feat/g1-combined-eval`: `retarget_init_checkpoint.py`, `prepare_wbt_sonic78.py`, `wbt_joint28_reference.py`,
  streamer `--policy-fps`, `--hold-rest-s`; launcher options.
- `feat/g1-pretrain-finetune` (depends on the above): `prepare_wbt_multitask_sonic78.py`, `wbt_splits.py`,
  `make_finetune_grid.py`, `data_efficiency_table.py`.

New code is limited to: position-stratified 10 / 25 / 50 subsets (extend `wbt_splits.py`), a grid entry for the
new dataset (extend `make_finetune_grid.py`), the handoff fix in the streamer, and a small trial-result table script.

## Out of scope

- XR-1, Psi0 and the HE + Unitree (`heu`) init: added only if the GR00T / Pi0.5 result warrants it.
- Closed-loop sim and open-loop scores as the decision metric (both disagreed with the robot before); they may be
  run as sanity checks only.
- The earlier 86-episode grid: its 7 apple-tray robot runs (42–46, 52, 56) stay on PC2 and are reported as a data
  gap, not counted.
