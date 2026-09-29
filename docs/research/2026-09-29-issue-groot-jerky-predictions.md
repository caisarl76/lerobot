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

## Mitigations in place (not a fix)

Options, off by default: `--noise-seed` (server), `--chunk-blend-s`, `--replan-s`, `--max-token-step` (streamer). In
sim (HE GR00T, planner start, held-out episodes 1293 / 1300), seed + blend 0.3 s cut palm jerk p95 from 126 / 166 to
44 / 43 m/s³ and the 1–2 Hz share from 2.9 / 3.3 % to 1.6 / 1.0 %, at +1–1.5 cm median palm error.

## Next steps

- [ ] Retrain HE GR00T with a realistic batch (e.g. 32+) and/or more steps; compare the open-loop smoothness
      metrics above before any sim or robot run.
- [ ] Check GR00T's action normalization statistics for the SONIC tokens.
- [ ] Try more flow-matching steps / sample averaging at inference.
- [ ] Re-evaluate in order: open loop → sim (planner start, table 25 cm) → robot.
