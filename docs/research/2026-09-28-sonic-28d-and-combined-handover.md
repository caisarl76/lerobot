# G1 Dex3 + SONIC: handover for 28D joint policies, combined training and evaluation (2026-09-28)

This page continues [`2026-09-27-sonic-real-robot-handover.md`](./2026-09-27-sonic-real-robot-handover.md) (setup,
first real-robot run) and [`2026-09-23-sonic-roundtrip-audit.md`](./2026-09-23-sonic-roundtrip-audit.md) (the
knowledge note). It covers what was added on 2026-09-24…28: open-loop held-out evaluation of all ho5 models, faster
Humanoid Everyday (HE) training, a combined Unitree + HE dataset, and a path that lets **any VLA that outputs 28D arm +
hand joints drive the SONIC-based G1**.

This repository is public: addresses, user names and credentials are left out on purpose.

## Status in one paragraph

A 28D policy (14 arm + 14 Dex3 hand joints) can now drive the robot through SONIC:
`sonic_policy_streamer.py --action-space joint28` encodes each predicted chunk to SONIC tokens on the robot side with
the official encoder (PR #9). Offline it reproduces the stored `sonic78_nolimit` tokens (99.98% identical for chunks of
40+ frames). In NVIDIA's deploy (MuJoCo, 80 cm table 5 cm away), a 28D GR00T kept balance and matched the 78D GR00T at
the median on two held-out episodes, with a worse left-hand tail on one. All 14 ho5 models passed load-and-predict
checks and have open-loop held-out scores (ACT best open loop; GR00T best closed loop, per the earlier sim ranking).
GR00T, Pi0.5 and MolmoAct2 have also been trained on a combined Unitree + HE dataset (relabelled state, one camera);
they are **not yet evaluated**.

## What was added

Repository (`examples/g1_dex3_training/`, merged in PR #9):

| File                                              | What it does                                                                                             |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `sonic_targets.joint_chunk_to_sonic()`            | Online counterpart of `prepare_sonic_dataset.py`: `[N,28]` joints → `[N,78]` tokens + hands, limits off. |
| `sonic_policy_streamer.py --action-space joint28` | Runs the conversion on each chunk; needs `--encoder-model`, `--observation-config`, `--robot-xml`.       |
| `sonic_joint28_stream_check.py`                   | Offline check of the conversion against the stored `sonic78_nolimit` tokens.                             |

H100 (`/run-output` = `/mnt/data01/jhkim/model_weight/g1_dex3_20260922`; audit/sim dir
`A=/mnt/data01/jhkim/model_weight/sonic_roundtrip_20260923`):

| Path                                                                                  | What                                                                                               |
| ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `/run-output/runs/groot_joint28_ho5_full`                                             | 28D GR00T (config = `groot_sonic78nolimit_ho5_full` with dataset `joint28` and a 28D action).      |
| `/run-output/runs/{groot,pi05,molmoact2}_combined_sonicstate_1cam_ho5_full`           | Combined Unitree + HE models (40K steps each, exit 0, not yet evaluated).                          |
| `/run-output/datasets/combined_sonicstate_1cam`                                       | Combined dataset (below). Build script: `/run-output/combined/build_combined.py`.                  |
| `/run-output/configs/heldout_combined_sonicstate_5pct.json`                           | Its held-out split (union of the two ho5 splits; HE episodes offset by 3152).                      |
| `/run-output/humanoid_everyday_g1_20260923/datasets/sonic78_nolimit_g2`, `videos_g2/` | HE with re-encoded video (below). All HE ho5 models except the first ACT attempt train on it.      |
| `/run-output/eval_ho5/`                                                               | `finalize_*.log` (load-and-predict checks) and `openloop_*.json` (open-loop held-out scores).      |
| `/run-output/eval_ho5/ho5_openloop_eval.py`                                           | Open-loop evaluator for any of the 7 policy types, Unitree or HE.                                  |
| `$A/J28_groot_ep{6,2155}_gap5`                                                        | Sim runs of the 28D GR00T; compare with `$A/FINAL_groot_ep{6,2155}_gap5` (78D).                    |
| `$A/code_joint28`                                                                     | Copy of `$A/code` plus the PR #9 files, used for the 28D sim runs (`$A/code` itself is unchanged). |

## 28D policies through SONIC

How it works: SONIC's encoder (mode 0) takes absolute arm joint targets with a 0.9 s look-ahead (10 future poses,
5 ticks apart at 50 Hz); legs and waist are the nominal standing pose, exactly as the dataset converter does. The
streamer treats each 28D chunk like a short episode (holding the last pose past its end), encodes it on CPU, passes the
14 hand values through, and hands the resulting 78D chunk to the unchanged 50 Hz resampler, startup/shutdown path and
gates. Because the conversion runs robot-side, the policy server (H100) is unchanged and returns 28D.

- Chunks should cover the replan interval + 0.9 s: GR00T (40 frames) and Pi0.5 (50) are fine; MolmoAct2 (30) is
  slightly short (the check below shows the cost is small).
- 28D fits every pretrained action head without resizing (GR00T N1.7 ≤ 132, RLDX-1 ≤ 64, Xiaomi-Robotics-0 and
  Pi0.5 ≤ 32), which is the point of this path.

Offline check (`sonic_joint28_stream_check.py`, 20 Unitree episodes incl. held-out 2155 and 6, same encoder and robot
XML hashes as the `sonic78_nolimit` build):

| Streaming setting                        | Tokens identical to the dataset | Largest difference | Hands        |
| ---------------------------------------- | ------------------------------- | ------------------ | ------------ |
| Whole episodes                           | 100%                            | 0                  | identical    |
| 40, 50 or 100-frame chunks, replan 0.4 s | 99.98%                          | one 1/16 grid step | ≤ 2.4e-7 rad |
| 30-frame chunks                          | 99.57%                          | one 1/16 grid step | ≤ 2.4e-7 rad |

Sim (NVIDIA deploy in MuJoCo, table 5 cm away, table-safe startup `r20`, recorded images and state from
`sonic78_nolimit`, scorer `sonic_stream_eval.py`):

| Held-out episode | GR00T | Palm error median / p95 / max | Palm p95 left / right | Wrist orient. median | Max tilt | Min foot contacts |
| ---------------- | ----- | ----------------------------- | --------------------- | -------------------- | -------- | ----------------- |
| 6                | 78D   | 3.6 / 13.8 / 18.2 cm          | 15.2 / 8.3 cm         | 10.2°                | 2.0°     | 8                 |
| 6                | 28D   | 4.9 / 12.5 / 17.5 cm          | 14.1 / 11.3 cm        | 16.7°                | 2.6°     | 8                 |
| 2155             | 78D   | 8.7 / 10.7 / 18.3 cm          | 10.5 / 10.8 cm        | 24.6°                | 3.9°     | 8                 |
| 2155             | 28D   | 8.8 / 20.4 / 25.5 cm          | 22.2 / 11.0 cm        | 27.6°                | 4.6°     | 8                 |

Chunk time including encoding: median 0.28–0.31 s, max 0.36 s (inside the 0.4 s replan). Two episodes cannot separate
a real 28D-vs-78D difference from run-to-run noise.

Running it (sim): copy `sonic_official_sim_eval.sh`, replace `$A/code` with `$A/code_joint28`, and add
`-v \$M:/sonic-model:ro` to the streamer container; then pass
`--action-space joint28 --encoder-model /sonic-model/model_encoder.onnx --observation-config /sonic-model/observation_config.yaml --robot-xml /run-output/environment/g1_29dof_with_hand.xml`
(`/run-output/environment/g1_29dof_with_hand.xml` is the XML the converter used; SHA-256 `8b68d8f0…`). On the real
robot, add the same four options to the streamer command in the 2026-09-27 handover; the robot PC then also needs
`onnxruntime` and the SONIC v1.1 encoder files.

## Open-loop held-out evaluation (all 14 ho5 models)

`ho5_openloop_eval.py` feeds recorded observations every 5 frames and compares the predicted chunk with the recorded
78D chunk, on the 5% held-out episodes and an equal-size training sample. Error = |error| / per-dimension std;
“hold previous” repeats the last recorded action. Values: held-out (training sample in the JSON files).

| Dataset | Policy    | Tokens step 0 | Tokens step 29 | Hands step 0 | Hands step 29 |
| ------- | --------- | ------------- | -------------- | ------------ | ------------- |
| Unitree | ACT       | **0.184**     | **0.263**      | 0.154        | **0.215**     |
| Unitree | MolmoAct2 | 0.216         | 0.325          | **0.145**    | 0.235         |
| Unitree | Diffusion | 0.230         | 0.398          | 0.215        | 0.326         |
| Unitree | GR00T     | 0.232         | 0.327          | 0.201        | 0.269         |
| Unitree | FastWAM   | 0.261         | 0.423          | 0.263        | 0.391         |
| Unitree | Pi0.5     | 0.263         | 0.318          | 0.256        | 0.311         |
| Unitree | VLA-JEPA  | 2.219         | —              | 1.143        | —             |
| Unitree | hold prev | 0.012         | 0.270          | 0.010        | 0.196         |
| HE      | ACT       | **0.192**     | **0.267**      | **0.266**    | **0.364**     |
| HE      | MolmoAct2 | 0.211         | 0.333          | 0.298        | 0.433         |
| HE      | Diffusion | 0.272         | 0.478          | 0.319        | 0.523         |
| HE      | GR00T     | 0.275         | 0.366          | 0.375        | 0.477         |
| HE      | Pi0.5     | 0.281         | 0.342          | 0.430        | 0.505         |
| HE      | FastWAM   | 0.334         | 0.505          | 0.355        | 0.544         |
| HE      | VLA-JEPA  | 2.098         | —              | 1.769        | —             |
| HE      | hold prev | 0.019         | 0.339          | —            | —             |

- Open-loop ranking does not predict closed-loop results: ACT is best here, GR00T is best in the sim.
- Unitree held-out ≈ training (no memorisation); HE held-out is 5–15% worse.
- Every model is far behind “hold previous” for the first steps and only matches it around 1 s ahead: the models do not
  recover the current action from image + state alone.
- **VLA-JEPA did not learn** (action loss 1.15 → 0.85 over 40K steps; error ~2 std on training data too). Short
  tests below.
- VLA-JEPA strict loading fails when the checkpoint config keeps `reinit_modules`: the tied `embed_tokens` is not in the
  file (only `lm_head` is) and the custom check in `VLAJEPAPolicy._load_as_safetensor` does not account for tied weights.
  With `cfg.reinit_modules = None` (as `finalize_baseline.py` does) the load is correct. Not fixed.

### VLA-JEPA on SONIC tokens: short tests

The same recipe (`lerobot/VLA-JEPA-Pretrain`, batch 1, lr 1e-4, ACTION MIN*MAX with `clip_normalized_actions`,
`reinit_modules` = action encoder/decoder + state encoder) learned on 28D joints (`vla_jepa_joint28_full`) but not on
78D tokens (`vla_jepa_sonic78nolimit_ho5_full`); the two configs differ only in dataset and action size. Three 5K-step
tests on Unitree ho5 (warmup 500, cosine decay over 40K so lr stays near peak; configs
`/run-output/configs/\_vjtest*{A,B,C}.json`, runs `/run-output/runs/_vjtest\_\_`, scores
`/run-output/eval*ho5/openloop_vjtest*_.json`, stride 10):

| Test | Change          | Action loss at 5K  | Tokens step 0 held-out / train | Hands step 0 held-out |
| ---- | --------------- | ------------------ | ------------------------------ | --------------------- |
| A    | none (control)  | 0.937              | 2.284 / 2.287                  | 1.144                 |
| B    | batch size 8    | **0.607**          | 1.896 / 1.896                  | 0.869                 |
| C    | ACTION MEAN_STD | 1.454 (other norm) | 0.957 / 0.963                  | 0.970                 |

On this metric a constant prediction of the mean scores ≈ 0.8.

- **Batch size 1 is the main problem.** B's action loss fell like joint28's did at its break (per 1K steps:
  A 1.02 → 0.98 → 0.94 → 0.95 → 0.94; B 0.93 → 0.91 → 0.85 → 0.73 → 0.62). Its open-loop hands improved (0.87 vs
  1.14), tokens only a little, because the MIN_MAX effect below still dominates.
- **MIN_MAX is a second problem.** Under MIN_MAX the unnormalized token outputs land ~2–3× further off than the mean
  would; with MEAN_STD (C) they drop to the level of an uninformative prediction. Step 0 and step 6 errors are the
  same in every test: the outputs are not yet conditioned on the observation.
- **5K steps is short.** Action loss per 1K steps of the two full runs:

  | Step              | 1K   | 3K   | 5K   | 8K   | 10K  | 12K  | 14K  | 20K  |
  | ----------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
  | joint28 (learned) | 0.94 | 0.86 | 0.79 | 0.64 | 0.52 | 0.36 | 0.15 | 0.09 |
  | sonic78 (failed)  | 1.03 | 0.97 | 0.98 | 0.95 | 0.93 | 0.92 | 0.91 | 0.90 |

  joint28 broke through between 5K and 14K; at 5K it would also have scored about 1 open-loop. B is at the start of
  such a break; the tests cannot show where it ends.

- Next: retrain `vla_jepa_sonic78nolimit_ho5_full` with **batch 8 + MEAN_STD**, same held-out split, and check the
  open-loop score at ~15K before running the full 40K. Batch 8 runs at 0.58 s/step (≈ 2.4 h per 15K on one H100).
  Not started: GPU 0 went back to the Qwen server.

## Other changes and decisions

- **HE video decoding.** The HE source is H.264 with a keyframe every ~250 frames, so random access was slow. The
  videos were re-encoded with LeRobot's defaults (AV1, `g=2`, crf 30) into `videos_g2/` and exposed as
  `sonic78_nolimit_g2` (data and meta shared; PSNR 42.6 dB mean vs the original). DataLoader throughput with
  torchcodec: 27 → 429 samples/s; HE ACT trained at 8.7 steps/s instead of ~1.3.
- **Quantile stats fixed (2026-09-25).** Wrong global q01/q99 in `sonic78_nolimit` (Unitree state + action, HE action)
  and the sonicstate datasets were recomputed exactly (`augment_joint_quantiles.py`; backups
  `meta/stats.json.bak-badquantile-20260925`). The MolmoAct2/Pi0.5 ho5 runs were retrained; their checkpoint normalizers
  match the fixed stats. `joint28` still has the old approximate quantiles; only quantile-normalized policies
  (MolmoAct2, Pi0.5) would be affected, so fix it before training those on `joint28`.
- **Combined dataset** `combined_sonicstate_1cam`: Unitree `sonic78_nolimit_sonicstate` (3152 episodes) followed by HE
  `sonic78_nolimit_sonicstate` (4064), relabelled state, one camera `observation.images.cam_left_high` (HE egocentric
  renamed to it; Unitree `cam_right_high` dropped). Merged with `aggregate_datasets` without re-encoding; every
  episode's length and task checked against its source; exact quantiles. 383 held out + episode 2202 excluded.
  Training kept 40K steps for comparability (≈ half the passes per episode of the single-dataset runs).
- **Zombie processes in the training containers.** PID 1 there is `sleep infinity`, which never reaps children; a
  finished trainer stays a zombie and `kill -0 <pid>` waiters never fire. Wait on `.exit` files instead.
- **Qwen vLLM on GPU 0** (`jihun-lerobot-qwen36-gpu0`) was stopped for training and for the VLA-JEPA tests, and
  restarted after them; it is running.
- **GPU 0 training container lost the GPU** (“Failed to initialize NVML”, `cuda False`; training silently fell back to
  CPU at 8 s/step). It was recreated with the same settings and `--gpus device=<GPU 0 UUID>`. If a run is unexpectedly
  slow, check `nvidia-smi -L` in the container and the log for “Switching to 'cpu'”.
- An idle eval container `jihun-lerobot-g1-dex3-eval-gpu6-20260925` (GPU 6, spare memory only; other users' processes
  run there) can be reused for evaluation or removed.

## Next steps

1. **Evaluate the combined models.** Run `finalize_baseline.py` and `ho5_openloop_eval.py` on the three
   `*_combined_sonicstate_1cam_ho5_full` runs, split per source (Unitree / HE held-out). Then the sim on Unitree
   held-out 6 and 2155. They use one camera, so the sim streamer feeds only `cam_left_high`; compare with the
   two-camera `groot_sonic78sonicstate_ho5_full`.
2. **More 28D-vs-78D sim episodes.** Repeat 2155 and add 3–5 held-out episodes for both GR00T models to tell a real
   difference (the left-hand tail on 2155) from noise. About 6 min per run.
3. **A state-of-the-art VLA on 28D.** The point of the 28D path: train a model whose pretrained action space fits 28D.
   - Pi0.5 on `joint28` (fix `joint28` quantiles first).
   - RLDX-1 (Qwen3-VL-8B, 6.9B; code Apache 2.0, **weights non-commercial**): reads LeRobot **v2.1** plus
     `meta/modality.json`; use the `GENERAL_EMBODIMENT` slot. Needs its own container (Python 3.10, torch 2.7,
     transformers 4.57.0) and a v3 → v2.1 conversion (or a loader patch). 28D fits `max_action_dim=64` as released.
   - Xiaomi-Robotics-0 (Qwen3-VL-4B, 4.7B, Apache 2.0): own JSON + video data format; the released loader assumes
     dual-arm end-effector data, so G1 needs a new dataset class. Python 3.12, torch 2.8, transformers ≥ 4.57.1.
     28D fits its `[30, 32]` action shape as released.
   - 78D on either needs the action input/output layers resized and trained from scratch (Xiaomi also needs mismatched
     tensors dropped before its `strict=False` load, which still raises on shape mismatches).
   - Start with a 28D smoke run on `joint28` (or a combined joint28 dataset) in a dedicated container, then stream it
     with `--action-space joint28`.
4. **First real-robot run** (unchanged from the 2026-09-27 handover): VPN round trip to the policy server, live head
   camera layout and exposure. The 28D path adds `onnxruntime` and the encoder files on the robot PC.
5. **Closed-loop state gap.** SONIC reaches a slightly different arm pose than commanded (~0.2 rad), so recorded
   joint state drifts in closed loop. Training on relabelled state (sonicstate) helps; a 28D model on relabelled state
   (`joint28` actions + `sonic78_nolimit_sonicstate` state) is not built yet.
6. **Small fixes.** VLA-JEPA: fix the tied-weight check in `_load_as_safetensor` (ignore missing keys that share storage
   with a loaded key), then the batch 8 + MEAN_STD retrain above. Consider feeding the previous action as an input,
   given how far every model is from “hold previous” at the first steps.
