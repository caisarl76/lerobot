# VLA-JEPA on SONIC tokens: why it did not learn, and the retrain (2026-09-29)

VLA-JEPA is the one ho5 policy that did not learn on the 78D SONIC action (64 SONIC tokens + 14 Dex3 hand joints); see
the open-loop table in [`2026-09-28-sonic-28d-and-combined-handover.md`](./2026-09-28-sonic-28d-and-combined-handover.md)
(held-out token error ≈ 2.2 std, against 0.18–0.28 for the other policies). This note covers three short tests that
found the cause and the full retrain that follows from them.

This repository is public: addresses, user names and credentials are left out on purpose.

## The failure

The same recipe learned on 28D joints but not on 78D tokens:

- Recipe: pretrained `lerobot/VLA-JEPA-Pretrain`, **batch size 1**, lr 1e-4, warmup 5000, cosine decay over 30K to
  1e-6, 40K steps, ACTION normalization **MIN_MAX** with `clip_normalized_actions`, and `reinit_modules` = action
  encoder, action decoder and state encoder.
- `vla_jepa_joint28_full` (28D) and `vla_jepa_sonic78nolimit_ho5_full` (78D) differ only in dataset and action size.

Action loss per 1K steps of the two runs:

| Step              | 1K   | 3K   | 5K   | 8K   | 10K  | 12K  | 14K  | 20K  |
| ----------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| joint28 (learned) | 0.94 | 0.86 | 0.79 | 0.64 | 0.52 | 0.36 | 0.15 | 0.09 |
| sonic78 (failed)  | 1.03 | 0.97 | 0.98 | 0.95 | 0.93 | 0.92 | 0.91 | 0.90 |

joint28 broke through between 5K and 14K steps; sonic78 stayed flat for all 40K.

## Short tests

Three 5K-step runs on Unitree ho5, each changing one thing (warmup 500 and cosine decay over 40K, so lr stays near
peak). Open-loop evaluation with `ho5_openloop_eval.py` (stride 10) on the 5% held-out episodes and an equal-size
training sample. Error = |error| / per-dimension std; on this metric a constant prediction of the mean scores ≈ 0.8.

| Test | Change vs the failed recipe | Action loss at 5K  | Tokens step 0 held-out / train | Hands step 0 held-out |
| ---- | --------------------------- | ------------------ | ------------------------------ | --------------------- |
| A    | none (control)              | 0.937              | 2.284 / 2.287                  | 1.144                 |
| B    | batch size 8                | **0.607**          | 1.896 / 1.896                  | **0.869**             |
| C    | ACTION MEAN_STD             | 1.454 (other norm) | **0.957** / 0.963              | 0.970                 |

Action loss per 1K steps: A 1.02 → 0.98 → 0.94 → 0.95 → 0.94; B 0.93 → 0.91 → 0.85 → 0.73 → 0.62.

- **Batch size 1 is the main problem.** B's loss falls the way joint28's did at its break, while A stays flat. A likely
  reason (not tested further): with one sample per step the gradient on 78D tokens is too noisy for the reinitialized
  action head, while 28D joints are largely predictable from state and got through anyway.
- **MIN_MAX is a second problem.** Under MIN_MAX the unnormalized token outputs land 2–3× further off than the mean
  would (A: 2.28, B: 1.90), which is why the failed run scored worse than an uninformative prediction. With MEAN_STD
  (C) the error drops to that level (0.96). Step 0 and step 6 errors are equal in every test: after 5K steps no
  output is conditioned on the observation yet.
- **5K steps is too short to see a policy learn.** joint28 would also have scored about 1 at 5K. The tests show which
  settings let the loss move, not the final quality.

`clip_normalized_actions` is ignored under MEAN_STD (the processor logs a warning instead of clamping to 1 std), so
C was not clipped.

## Full retrain

`vla_jepa_sonic78nolimit_ho5_b8meanstd_full`: the failed recipe with exactly two changes, **batch size 8** and
**ACTION MEAN_STD** (`clip_normalized_actions` off). Everything else is unchanged so the result stays comparable with
the other ho5 policies: same dataset (`sonic78_nolimit`), same held-out exclusions, 40K steps, warmup 5000, cosine
decay over 30K. The old run keeps its name.

- Batch 8 uses ~32 GB on one H100 and runs at 0.69 s/step (0.58 in test B): ≈ 7.7 h for 40K steps, 8× the samples of
  the batch-1 run.
- Checkpoints every 15K steps (16 GB each), at 15K, 30K and 40K.
- Checkpoint at 15K: open-loop evaluation, where joint28 had already learned (loss 0.15 at 14K). If the token error is
  not clearly below 0.8 there, stop and look at the action head itself (the 1/16-grid token values,
  `reinit_modules`, `repeated_diffusion_steps`) instead of finishing the run.
- After 40K: `finalize_baseline.py` and the open-loop evaluation at stride 5, to fill the VLA-JEPA row of the
  handover table.

Status (2026-09-29 08:40 KST): **running** on H100 GPU 0 since 08:36 KST (the Qwen server on GPU 0 is stopped for
it). Expected: 15K checkpoint ≈ 11:30 KST (its evaluation ≈ 11:40), 40K ≈ 16:20 KST. The two evaluations are queued in
the evaluation container and wait for the checkpoint and the `.exit` file; progress is in
`/run-output/eval_ho5/vla_jepa_sonic78nolimit_ho5_b8meanstd_full.status`, scores in
`/run-output/eval_ho5/openloop_vla_jepa_sonic78nolimit_ho5_b8meanstd_full{_15k,}.json`.

## Loader bug (not fixed)

VLA-JEPA strict loading fails when the checkpoint config keeps `reinit_modules`: the tied `embed_tokens` is not in the
file (only `lm_head` is), and the custom check in `VLAJEPAPolicy._load_as_safetensor` does not account for tied
weights. With `cfg.reinit_modules = None` before loading (as `finalize_baseline.py` and `ho5_openloop_eval.py` do) the
load is correct.
