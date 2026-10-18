# G1 VLA / WAM candidates and training plan (pretrain on HE/Unitree → fine-tune on our teleop data)

Date: 2026-10-14. Goal (see the pretrain → fine-tune study): show that pretraining on open G1 data (Humanoid Everyday
"HE", Unitree) lets a policy reach a given robot success rate with far fewer of our own teleop episodes.
All training uses the sonic78 format (state 28 = arms + Dex3 hands; action 78 = SONIC motion token 64 + Dex3 hand
targets 14; one head camera `observation.images.egocentric`; 50 Hz for our data).

## 1. Policies in this repo

24 `policy.type`s are registered. Six are not imitation VLAs/WAMs and are left out (`tdmpc`, `gaussian_actor` (RL);
`vqbet`; the optimizer/scheduler presets of `lawam` and `molmoact2`). ACT and Diffusion are kept as small-model
controls.

### Already trained on G1 data

| policy (`type`)                        | kind                                        | HE pretrain | Unitree pretrain | HE + Unitree ("heu") | fine-tuned on our data                                                                        | real robot                                                     |
| -------------------------------------- | ------------------------------------------- | ----------- | ---------------- | -------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| GR00T N1.7 (`groot`)                   | VLA, flow DiT head                          | yes         | yes              | yes                  | handover 20/41 × base/he; multitask 25/50/100 % × base/he/heu                                 | handover: base 3.5/5, 4/5; he 2/5, 5/5 (20/41 ep)              |
| Pi0.5 (`pi05`)                         | VLA, PaliGemma + flow expert                | yes         | yes              | yes                  | same as GR00T                                                                                 | handover: base 0/3, 1/5; he 3/5, 3/5                           |
| Psi0 (`psi0`, ours)                    | VLA                                         | yes         | yes              | no                   | handover 41 × base/he                                                                         | first runs; he session s04 next                                |
| Xiaomi-Robotics-1 (`xiaomi_robotics`)  | VLA                                         | yes         | yes              | yes                  | handover 41 × base/he; multitask grid                                                         | zero-shot only (laptop 2/3 with no-state variant); he stalls in sim |
| VLA-JEPA (`vla_jepa`)                  | VLA + JEPA latent prediction                | yes         | yes              | no                   | no                                                                                            | no                                                             |
| MolmoAct2 (`molmoact2`)                | VLA, action reasoning                       | yes         | yes              | no                   | no                                                                                            | no (12.7 GB, does not fit the 12 GB robot-server GPU as is)    |
| FastWAM (`fastwam`)                    | world action model (video + action)         | yes         | yes              | no                   | no                                                                                            | no (lost balance in sim, 0.6–0.7 s per chunk > 0.4 s replan)   |
| ACT (`act`)                            | small transformer, no VLM                   | yes         | yes              | no                   | no                                                                                            | no                                                             |
| Diffusion (`diffusion`)                | small U-Net, no VLM                         | yes         | yes              | no                   | no                                                                                            | no (lost balance in sim)                                       |

Every checkpoint is listed in [`g1-model-registry.md`](./g1-model-registry.md) (HF: `mncai/g1-*`).

### In the repo, never trained on G1

All of them predict actions narrower than our 78-D (defaults: 20–32), so their action input/output projections are
re-initialised, as was done for Pi0.5.

| policy (`type`)            | backbone / size                                  | public init                         | kind                                          | fit for our study                                                                                                                                   | priority |
| -------------------------- | ------------------------------------------------ | ----------------------------------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| X-VLA (`xvla`)             | Florence-2 + soft prompts, 0.9B                  | X-VLA-0.9B (290K episodes, 7 robots) | VLA, flow                                     | built for cross-embodiment pretraining and few-demo adaptation: one soft prompt per dataset (HE / Unitree / ours) maps directly onto our question; small and fast to serve | **A**    |
| LaWAM (`lawam`)            | Qwen3-VL-2B + latent world model + flow head     | `jialei02/lawam-pretrain-lerobot`   | latent world action model                     | a second WAM next to FastWAM, but it predicts latent dynamics instead of video, so it should be cheap enough for 0.4 s replanning on the robot; repo added 09-17, untested | **A**    |
| SmolVLA (`smolvla`)        | SmolVLM2, 0.45B                                  | `lerobot/smolvla_base`              | VLA, flow                                     | the small-model point of the data-efficiency curve; trains and serves cheaply                                                                       | B        |
| EVO1 (`evo1`)              | InternVL3-1B, ~1B                                | InternVL3 only                      | VLA, flow                                     | same role as SmolVLA with no robot pretraining of its own, so HE is its only robot prior: the cleanest test of the HE effect, but it duplicates SmolVLA's role | B        |
| EO-1 (`eo1`)               | Qwen2.5-VL-3B                                    | EO-1 release                        | VLA, interleaved vision-text-action, flow    | overlaps Pi0.5/GR00T in size and design                                                                                                             | C        |
| WALL-OSS (`wall_x`)        | Qwen2.5-VL-based, ~4B                            | `x-square-robot/wall-oss-flow`      | VLA, flow (+FAST)                             | large; overlaps Pi0.5                                                                                                                               | C        |
| Pi0 (`pi0`)                | PaliGemma 3B                                     | `lerobot/pi0_base`                  | VLA, flow                                     | superseded by Pi0.5 on the same backbone                                                                                                            | C        |
| Pi0-FAST (`pi0_fast`)      | PaliGemma 3B                                     | `lerobot/pi0fast_base`              | VLA, autoregressive FAST tokens               | 78 dims × 50 Hz chunks give long token sequences, so autoregressive decoding will likely not fit the 0.4 s replan                                                       | C        |
| LingBot-VA (`lingbot_va`)  | Wan2.2 ~5B video DiT                             | `robbyant/lingbot-va-base`          | autoregressive video-action world model      | heaviest model here; video denoising per chunk will not fit the 0.4 s replan or the 12 GB robot GPU                                                    | C        |
| Multi-task DiT (`multi_task_dit`) | CLIP + DiT, 0.45B                         | CLIP only                           | diffusion/flow policy, no VLM                 | another no-VLM control next to ACT/Diffusion                                                                                                        | C        |

## 2. Data

### Pretraining (done, reuse)

| set                                       | episodes      | notes                                                         |
| ----------------------------------------- | ------------- | ------------------------------------------------------------- |
| HE (`humanoid_everyday_g1_20260923`)      | 226 held out  | main pretraining source                                       |
| Unitree G1, corrected (`unitree`)         | 158 held out  | open loop: adding it to HE changes little (tokens ≤ 0.006)    |
| HE + Unitree (`combined_261009`)          | 7,216 (4.37 M frames) | "heu" init for GR00T, Pi0.5, XR-1                    |

For new models, pretrain on **HE only** (heu ≈ he open loop; halves the pretraining cost).

### Our teleop recordings (`/mnt/data/jihun/datasets/G1_WBT_GR00T`)

All are GR00T-WBC SONIC recordings (LeRobot v2.1, 50 Hz, `action.motion_token` + `action.wbc` + `observation.state`), so each one
converts with `prepare_wbt_sonic78.py` without relabelling. `*_subtask`, `*_full_task`, `*_annotations`, `*_left_only`,
`*_curation` and `*_ver*` are relabelled copies or subsets of a parent, not new episodes.

| recording                         | episodes | frames  | task                                                     | sonic78 version                                      |
| --------------------------------- | -------- | ------- | -------------------------------------------------------- | ---------------------------------------------------- |
| handover_bottle (subset 261002)   | 43       | 18,970  | pick drink bottle and hand over                          | yes (`handover_bottle_sonic78_50hz_261002`, held out 27, 35) |
| pnp_table_261001                  | 102      | 104,391 | multi-task: tray/plate/basket transfer, blocks → cup, basket, pour | yes, 86 prompted (`wbt_multitask_sonic78_50hz_261009`) |
| pnp_table_261013                  | 34       | 79,481  | place OBJECT on the table (newest)                       | no                                                   |
| pnp_bottle_260916                 | 179      | 281,227 | place bottle on the table                                | no                                                   |
| pnp_bottle_260929                 | 28       | 128,200 | place OBJECT on the table                                | no                                                   |
| xr_pick_cup_real_002              | 130      | 125,575 | pick up the cup                                          | no                                                   |
| xr_pick_cup_002                   | 52       | 41,580  | pick up the cup (sim or real: check)                     | no                                                   |
| pnp_trash                         | 92       | 198,846 | pick trash, turn, put in bin (left/right)                | no                                                   |
| pnp_table_260909 (+ 260908, 260910, pnp_table) | 87 + 40 + 35 + 77 | — | approach table, pick bottle/apple       | no                                                   |
| put_object_260914                 | 75       | 28,232  | place object                                             | no                                                   |
| point_block / touch_apple / hyelin / 261008_manip | 29 / 12 / 12 / 13 | — | small probes                                   | no                                                   |

Several recordings use the template prompt "place OBJECT on the table"; their per-episode prompts must be filled in
before VLA training (as `prepare_wbt_multitask_sonic78.py` did from `task_aug`).

## 3. Plan

Fixed across all phases, as in the handover study: same fine-tune sample budget at every data size; `he` init
retargeted to the fine-tune set's stats (`retarget_init_checkpoint.py`); robot cells scored with eval sessions
(`--trials`, one session per model), noise 0 + 0.3 s chunk blend, server on workstation GPU 1.

### Phase 1: complete the existing handover grid (fine-tune only, no new pretraining)

The cheapest new robot evidence: these models already have HE checkpoints.

| runs                                            | count | why                                                               |
| ----------------------------------------------- | ----- | ----------------------------------------------------------------- |
| Psi0 `wbt20` × base/he                          | 2     | Psi0 has only 41-episode cells                                    |
| GR00T, Pi0.5 `wbt10` × base/he (seed 0 subset of the 20) | 4 | the curve's low end: is Pi0.5 he at 10 still above base at 41 (1/5)? |
| VLA-JEPA `wbt{20,50}` × base/he                 | 4     | HE checkpoint exists; 4th VLA family on the robot                 |
| ACT `wbt{20,50}` × scratch/he                   | 4     | no-VLM control: does HE pretraining matter without a VLM? (minutes to train) |

14 runs. MolmoAct2 and FastWAM are held back until their serving problems are solved (memory, latency); FastWAM
fine-tunes add no robot result while one chunk takes 0.6–0.7 s.

### Phase 2: two new models (pretrain on HE, then the handover grid)

1. **X-VLA**: HE pretrain from X-VLA-0.9B at its official fine-tune recipe, then `wbt{20,50}` × base/he (4 runs).
   Optional ablation: frozen backbone + new soft prompt only (the paper's 1 % parameter adaptation) as a fifth cell.
2. **LaWAM**: HE pretrain from `lawam-pretrain-lerobot`, then `wbt{20,50}` × base/he. Measure per-chunk latency on the
   workstation before the fine-tunes; drop it if a chunk takes > 0.4 s.
3. If GPU time remains: **SmolVLA**, same grid (small-model point).

Each new model: one smoke run (load, one step, save, load-and-predict through `sonic_policy_server.py`) before its
full pretrain.

### Phase 3: a second task with a real data-size curve

The handover set has only 43 episodes, so it cannot show the curve beyond 41. Candidate: **pnp_bottle_260916 (179
episodes)** or xr_pick_cup_real_002 (130). Convert to sonic78, hold out 10 %, fine-tune at 10/25/50/100/all episodes
for the 2–3 best models of phases 1–2 × base/he. Before training: check that the recording matches the current robot
setup (hand calibration, camera, table, which hand), since these were recorded in September. If the balanced
right-hand study's new data (2 tasks × 50 episodes) arrives first, use that instead.

### Not planned

Pi0, Pi0-FAST, WALL-OSS, EO-1, LingBot-VA, Multi-task DiT, Diffusion (reasons in section 1); Unitree-only or heu
pretraining for new models; more open-loop or sim-only comparisons (sim palm error did not predict robot success).

## 4. Open questions for the user

- Phase 3 dataset: pnp_bottle_260916, xr_pick_cup_real_002, or wait for the balanced-study data?
- Is pnp_table_261013 (34 episodes, 10-13) part of the balanced-study collection?
- Is xr_pick_cup_002 real or sim?
- GPU budget: the phase 2 HE pretrains are the long jobs; which GPUs (h100 0/6, h100_174 4/5/7/1)?
