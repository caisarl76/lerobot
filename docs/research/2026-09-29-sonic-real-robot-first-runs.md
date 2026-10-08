# G1 Dex3 + SONIC: first real-robot runs, evaluation pipeline and findings (2026-09-28/29)

This page continues [`2026-09-28-sonic-28d-and-combined-handover.md`](./2026-09-28-sonic-28d-and-combined-handover.md).
It covers the evaluation of the combined Unitree + HE models, the real-robot pipeline (policy server on a workstation
GPU, streamer on the G1's PC2), the head-camera findings, first-deployment safeguards, a start mode for Humanoid
Everyday (HE) policies, and real runs 01–16. VLA-JEPA is covered in
[`2026-09-29-vla-jepa-sonic-tokens.md`](./2026-09-29-vla-jepa-sonic-tokens.md).

This repository is public: addresses, user names and credentials are left out on purpose.

## Status in one paragraph

The pipeline runs on the real G1: NVIDIA's deploy, a camera publisher and the streamer on PC2 (Orin NX), and the
policy server on a workstation RTX 3060 over the LAN. Sixteen real episodes ran end to end with no safety stop
except one (run08, see below). The main blockers are no longer the pipeline: (1) the robot's only head camera is a
RealSense D435i, which matches HE's egocentric camera but not the Unitree stereo camera, so Unitree-trained models see
out-of-distribution images; (2) HE GR00T reaches the right targets (it touched the laptop in "close a laptop g1")
but moves roughly, because every replan draws a new flow-matching sample; HE ACT moves smoothly but ignores the
instruction and does not reach the target. Smoothing options for GR00T are implemented and being tested in sim.

## What was added (`examples/g1_dex3_training/`, branch `feat/g1-combined-eval`)

| Change                                                          | What it does                                                                                                                    |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `ho5_openloop_eval.py`                                          | Open-loop held-out evaluator (was only on H100); a combined dataset is scored per source with each source's action std.         |
| `finalize_baseline.py --episode`                                | Load-and-predict check for datasets without `meta/provenance.json` (aggregated ones).                                           |
| `sonic_policy_server.py` / streamer `--backbone-dtype bfloat16` | Cast the frozen GR00T backbone to bf16 (12.6 → 9.6 GB), action head stays float32.                                              |
| Streamer import fallback                                        | Runs without lerobot/torch on the robot PC (`--policy-server` + `--images zmq`).                                                |
| Streamer first-chunk hold                                       | Keeps streaming the rest pose while the first chunk is computed (was a 0.16–0.28 s input gap).                                  |
| `g1_head_camera_server.py`                                      | Camera-only publisher in ImageServer's format; converts only published frames; `--out-size`/`--fit`.                            |
| `g1_head_camera_viewer.py`                                      | Watch the stream on a workstation through ffplay, correct colours, optional frame saving.                                       |
| `--max-token-step` (opt-in)                                     | Token slew limit for first deployments (0.05); default off, as the policies were trained on speed-limit-free tokens.            |
| Watchdog (default on)                                           | Ends the episode via the normal shutdown path if a measured arm joint exceeds 6 rad/s or leaves the training range by 0.15 rad. |
| `--start planner`                                               | Start and end in NVIDIA's planner stance (no standing token, no table path), for HE policies.                                   |
| `--noise-seed`, `--chunk-blend-s` (opt-in)                      | Smooth chunk switches of stochastic policies (being tested).                                                                    |
| Streamer `--log` stores its arguments                           | Settings of every run are recorded.                                                                                             |
| `sonic_official_sim_eval.sh`                                    | `CODE_DIR`, `JOINT28_DIR`, `SERVER_EXTRA`.                                                                                      |
| `sonic_replay_extract.py`                                       | `SONIC_SUFFIX=_nolimit` replays the conversion the policies were trained on.                                                    |

## Combined Unitree + HE models

All three combined models (`{groot,pi05,molmoact2}_combined_sonicstate_1cam_ho5_full`) passed load-and-predict.
Open-loop held-out error (per-dimension std of each source), step 0 / step 29:

| Model                                                      | Unitree tokens        | Unitree hands         | HE tokens         | HE hands      |
| ---------------------------------------------------------- | --------------------- | --------------------- | ----------------- | ------------- |
| GR00T, relabelled state, 2 cameras (fair Unitree baseline) | **0.209** / **0.311** | **0.200** / **0.268** | —                 | —             |
| GR00T combined                                             | 0.223 / 0.338         | 0.217 / 0.288         | 0.218 / 0.371     | 0.418 / 0.526 |
| Pi0.5 combined                                             | 0.263 / 0.337         | 0.253 / 0.319         | 0.243 / 0.344     | 0.478 / 0.583 |
| MolmoAct2 combined                                         | 0.183 / 0.328         | 0.150 / 0.246         | **0.169** / 0.349 | 0.323 / 0.486 |

Sim (NVIDIA deploy, table 5 cm, `r20` startup), palm error p50 / p95 averaged over held-out episodes 6 and 2155:
78D GR00T 6.2 / 12.3 cm, 2-camera relabelled GR00T 6.2 / 11.5 cm, combined 1-camera GR00T 7.5 / 15.8 cm. Combining
did not beat the single-dataset models; every combined model got worse on hands, especially HE hands.

## Real-robot pipeline

- **Policy server on a workstation RTX 3060 (12 GB).** GR00T checkpoints are float32 (12.6 GB). The fine-tunes freeze
  the backbone (all 1.52 B backbone values are bf16-exact), so `--backbone-dtype bfloat16` loses no weights; outputs
  differ from float32 by ~0.007 std (tokens) against ~0.23 std between two random seeds. 9.6 GB GPU memory, chunk
  0.19–0.22 s; loading on the CPU peaks at ~23 GB RAM.
- **PC2 (Orin NX, MAXN)**, in an isolated uv environment (`numpy`, `pyzmq`, `msgpack`, `opencv-python-headless`):
  streamer 19 % of one core / 70 MB, camera publisher 11–28 % of one core, NVIDIA deploy ~29 % of one core; the whole
  PC2 stays below ~15 % of 8 cores. The LAN link is Wi-Fi (RTT 3.9 ms mean, 11 ms max); chunk round trip 0.25 s.
- **Do not use `robots/unitree_g1/run_g1_server.py --camera` next to the deploy**: it also releases the robot's
  motion mode and forwards `rt/lowcmd`. Use `g1_head_camera_server.py`.

## Head camera

The G1's head camera is a RealSense D435i on **USB 2.0** (V4L2 colour node `/dev/video4`):

| Mode               | Field of view                                                                         | Rate                   |
| ------------------ | ------------------------------------------------------------------------------------- | ---------------------- |
| 640×480            | centre 4:3 crop of the sensor (correlation 0.989 with the 960×720 centre of 1280×720) | 30 fps                 |
| 1280×720           | full width (~69°)                                                                     | 15 fps (USB 2.0 limit) |
| 424×240, 1920×1080 | full width                                                                            | 30 / 8 fps             |

The Unitree datasets use a wide stereo head camera (both forearms, torso and far table edge in view), so
Unitree-trained models see very different images. **HE's `egocentric` camera matches the D435i in 640×480 mode**
(narrow view, no body visible, table edge in the lower third): use 640×480 and name the stream `egocentric` for HE
models.

## Safeguards

- Chunk switches were the source of fast motion: token steps at a switch were 0.09 median, up to 0.25, against
  ≤ 0.038 within a chunk; measured arm speed in sim reached 5.8 rad/s against a teleop p99.9 of 2.6 rad/s.
- `--max-token-step 0.05` (sim, combined GR00T, episodes 6 / 2155): peak arm speed 4.51 → 2.17 and 2.33 → 2.18 rad/s,
  palm error 5.3/14.5 → 5.8/15.1 and 9.6/17.0 → 9.4/16.1 cm, balance unchanged, watchdog never tripped.
- The table-safe startup path sweeps the palms **17–63 cm to each side**, −5…39 cm forward, 61–133 cm high. Keep
  table legs out of ±65 cm: in run01–03 the left hand caught on a table leg (arm burst 3.3 rad/s); the watchdog
  does not cover the startup path.

## HE policies need a different start

HE episodes start with the arms low (palms 68–70 cm high, 13–27 cm in front of the pelvis, close to the planner
stance) at a table of the same height (lowest forward palm ~81 cm for "push duck g1", 80 cm for the Unitree apple
task), so the table edge is ≥ ~30 cm in front of the pelvis. From the Unitree table path with the table 5 cm away,
the HE models drove their hands into the table (GR00T 10 table contacts, tilt 7.5–8.2°; ACT lifted a foot). With
`--start planner` and the table 25 cm further out, all four sim runs (GR00T, ACT; held-out "put dumpling into plate
g1" episodes 1293, 1300) kept all feet down, tilt 2.4–3.7°.

## Real-robot runs

| Runs  | Model                            | Task / setup                                         | Result                                                                                                         |
| ----- | -------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| 01–03 | combined GR00T (1 cam)           | apple; 1280×720 stretched; table path; slew 0.05     | Full cycle; chunks 0.20 s; left hand caught a table leg in the startup; no contact, fingers flexed in the air. |
| 05–07 | HE ACT                           | "push duck g1" (yellow tube); 640×480; planner start | Smooth (arm ≤ 1.33 rad/s, slew 0–6 ticks); right hand approached, no contact.                                  |
| 08    | HE GR00T                         | push duck, 90 s                                      | Stopped by the state-age check (0.22 s) after ~80 s; `.npz` not saved (written only at the end).               |
| 09–10 | HE GR00T                         | push duck (run10 with slew 0.1)                      | Back-and-forth arm motion; every chunk switch at the slew cap; run10 arm peak 5.09 rad/s.                      |
| 11–14 | HE ACT                           | "close a laptop g1"                                  | Right palm 14–33 cm forward against 39–44 cm in training: does not reach the laptop (camera gap, no language). |
| 15–16 | HE GR00T                         | "close a laptop g1"                                  | Right palm 41–43 cm forward, as in training; touched the laptop; palm jerk p95 101–133 m/s³ (ACT 16–26).       |
| 20–21 | HE GR00T official, noise scale 0 | "close a laptop g1"; slew 0.1; 2026-10-02            | Zero-shot result acceptable (user). **Reference GR00T inference setting**: `g1_groot_real_run.sh`.             |

## GR00T smoothness

Two GR00T samples for the same observation differ by ~0.23 std (as much as its error against the recording), so a
fresh sample every 0.4 s makes the target jump at every switch. Options (off by default): `--noise-seed` on the
server, `--chunk-blend-s` on the streamer. Sim, HE GR00T, planner start, table 25 cm, slew 0.05, held-out episodes
1293 / 1300:

| Setting          | Slew-limited ticks | Palm jerk p95 (m/s³) | Arm speed p95 (rad/s) | 1–2 Hz share | Palm error p50 / p95 (cm) |
| ---------------- | ------------------ | -------------------- | --------------------- | ------------ | ------------------------- |
| none             | 35 / 41            | 126 / 166            | 1.92 / 2.03           | 2.9 / 3.3 %  | 4.5/12.4, 3.1/14.6        |
| `--noise-seed 0` | 16 / 20            | 127 / 137            | 1.45 / 1.46           | 1.7 / 1.7 %  | 6.3/15.3, 4.9/15.0        |
| blend 0.3 s      | 0 / 0              | 46 / 46              | 1.84 / 2.10           | 7.0 / 4.4 %  | 3.7/13.3, 3.4/11.9        |
| seed + blend     | 0 / 0              | 44 / 43              | 1.52 / 1.57           | 1.6 / 1.0 %  | 5.9/16.2, 4.1/15.4        |

Seed + blend is the smoothest on every measure, at +1–1.5 cm median palm error. On the robot, blend 0.3 s alone
(run17) cut arm jerk p95 from 923–1018 to 400 rad/s³ but raised the ~1 Hz back-and-forth (1–2 Hz share 11–15 % →
34 %; human demonstrations 4 %); the sim underestimates that swing, as its images do not react to the arm.

**These are mitigations. The cause is likely in the model output itself, possibly from training**: the no-limit dataset replays smoothly, while the model's
open-loop predictions are already jerky. See the open issue
[`2026-09-29-issue-groot-jerky-predictions.md`](./2026-09-29-issue-groot-jerky-predictions.md).

## Known issues

- **Resolved for inference (2026-10-08): GR00T jerky motion.** The jumps came from switching chunks in one tick;
  noise scale 0 plus `--chunk-blend-s 0.3` is smooth on the robot (runs 24-28), and `g1_groot_real_run.sh` defaults to
  it. See [`2026-09-29-issue-groot-jerky-predictions.md`](./2026-09-29-issue-groot-jerky-predictions.md).
- **Open, for future research: the HE GR00T closes the laptop only part way** ("close a laptop g1", runs 20-29).
  - With the smooth setting the arm moves for ~15 s, then stops and stays still to the end of the 30 s episode (mean
    arm speed per 5 s window drops to ~0.01 rad/s in runs 24, 26, 27, 28); the episode length is not the limit.
  - Noise scale 0.5 keeps the arm moving longer (~20 s, run 29) but is jerky inside the chunks, so it is not a fix.
  - Leading hypothesis (user, 2026-10-08): **the visual difference between the HE recordings and the current
    inference scene.** The camera geometry matches (D435i 640×480 = HE `egocentric`, see "Head camera"), so the gap
    would be in appearance: room, lighting, table, laptop model and its position, background. A policy that no longer
    recognises the scene would stall in a "done-looking" pose instead of finishing.
  - Ways to test it: (1) put HE frames of the same task next to live D435i frames (and compare image statistics);
    (2) open loop, feed the recorded robot-run frames to the policy and compare its chunks with those for HE frames
    of the same phase; (3) rearrange the scene closer to HE (laptop model/pose, table, light) and rerun;
    (4) fine-tune on a few robot demonstrations or train with stronger appearance augmentation.
- H100 GPU 7: NVIDIA's sim host segfaults during the TensorRT build (twice, ~4 min after start); GPU 6 works.
- The VPN to H100 stalls intermittently (SSH up to 15 s); copying a 12.6 GB checkpoint took 0.5–2 h.
- `sonic_official_sim_eval.sh` relaunches forever after a sim-host crash and hangs if the streamer dies early.
- The streamer writes its `.npz` only at the end of the cycle; an interrupted run (run08) leaves only the text log.
- VLA-JEPA: its first fix (batch 8 + MEAN_STD, see the VLA-JEPA note) was superseded; it is part of the
  official-recipe retraining below.

## Status after this note (2026-09-29, later)

- All G1 SONIC runs trained before the fix were **deleted on the H100** (wrong recipes). Every policy is being
  retrained with its official recipe (7 policies × HE and Unitree, `*_official_full`); schedule and recipes are in
  the retraining section of [`2026-09-29-issue-groot-jerky-predictions.md`](./2026-09-29-issue-groot-jerky-predictions.md).
  The results in this note (open loop, sim, real runs 01–17) are for the old models; the only remaining copies of
  those models are local to the evaluation workstation.
- The old-model sim runs, the dataset replay (`review_nolimit_20260929/`), `eval_ho5/bf16check/` (with
  `openloop_smooth.py`) and the streamer copy `code_safety/` are kept on the H100 as baselines and tools for
  judging the retrains.
- The smoothing sims finished (seed + blend smoothest, table above); the robot test of seed + blend with the old HE
  GR00T was not run, as the model is superseded.

## Next steps

1. Judge each official-recipe retrain in order: open loop (`openloop_smooth.py`, against the smoothness numbers in
   the issue note), then sim (planner start for HE, table 25 cm), then the robot. Only then decide whether
   `--noise-seed` / `--chunk-blend-s` are still needed.
2. Close the camera gap for the Unitree tasks: record teleop demonstrations with the D435i and fine-tune.
3. Detect a stuck arm during startup/shutdown (measured vs planned path); save the streamer log on abnormal exit;
   stop `sonic_official_sim_eval.sh` from relaunching after a sim-host crash.
4. Memory check on the 3060 for the retrained Pi0.5 (9.4 GB before) and MolmoAct2 (12.7 GB, does not fit as is).
