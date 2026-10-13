# G1 real-robot evaluation logger (2026-10-13)

Operator-judged trials on the G1, exported as a LeRobot dataset with the operator's verdicts, then labelled
(failed subtask and failure frame) by a VLM and reviewed by a human in the dataset visualizer. Spec:
`docs/superpowers/specs/2026-10-13-g1-eval-logger-design.md`.

Run the launcher commands from `examples/g1_dex3_training/` on the workstation.

## 1. Start

```bash
./g1_groot_real_run.sh sync      # once per code change: copies sonic_policy_streamer.py, stream_backends.py, eval_session.py, sonic_targets.py, wbc_common.py, sonic_token_stream.py to PC2
./g1_groot_real_run.sh server    # workstation policy server; wait for "ready on port"
./g1_groot_real_run.sh deploy    # PC2 NVIDIA SONIC deploy; wait for "Init Done"
./g1_groot_real_run.sh camera    # PC2 D435i head camera
```

## 2. Session

```bash
TRIALS=10 ./g1_groot_real_run.sh session s01_<model>_<task> "<task text>"
```

`TRIALS` defaults to 10, `MAX_S` (per-trial cap) to 120. Extra streamer args go after the task text, e.g.
`--failure-reasons "a,b,c"` replaces the default failure menu (default: `eval_session.DEFAULT_FAILURE_REASONS`). The remote shell re-splits the arguments, so reasons containing spaces need nested quoting: `--failure-reasons "'grasp failed,dropped it'"`.

Operator keys:

- Enter at "Init Done".
- Per trial: Enter when the robot is in position, Enter to stop the trial, then `s` (success), `f` (fail), or `r` (redo, the trial is discarded and run again).
- After `f`: the failure reason number from the menu (after a watchdog stop, Enter takes "safety stop (watchdog)"), then a free-text note (Enter to skip).

Each finished trial is written at once. A crash or Ctrl-C keeps the finished trials; the same command continues at the next trial (`results.jsonl` line count = next trial number). A resume with another task, action space or policy (server/path) than trial 1 is refused before the robot moves.

## 3. Pull and convert

```bash
export SESSIONS=/mnt/data/jihun/g1_eval_sessions   # only defaulted inside the launcher
./g1_groot_real_run.sh pull s01_<model>_<task>       # PC2 -> $SESSIONS/s01_.../raw
uv run python eval_session_to_lerobot.py --session-dir $SESSIONS/s01_.../raw --out $SESSIONS/s01_.../dataset \
    --subtasks "reach,grasp,..."
```

`--subtasks` is the task's ordered subtask list (the VLM labels against it). Options: `--repo-id` (default `local/<out dir name>`), `--fps` (30), `--overwrite`. One episode per kept trial over the policy-controlled phase; trials that never reached it (or have no camera frames) are listed under `skipped` with their full `results.jsonl` line plus `why`, so their verdicts still count. Trials with differing task/action space/policy are rejected; camera gaps over 0.2 s, including at the episode start and end, are printed as warnings.

## 4. Label and review (dataset visualizer, branch feat/eval-labels)

1. Start the backend and UI as in the visualizer's `docs/local-setup.md` (backend with a configured VLM endpoint).
2. `/annotate` -> prepare the dataset directory (local path) -> open the draft -> **Evaluation** panel.
3. The panel header shows the session summary: success rate, per-reason counts, disagreements (VLM verdict differs from the operator), labelled, reviewed. There is no separate dataset view.
4. **Run VLM labels** (bulk) labels only episodes that have no eval label yet (checked again right before each write, so a label saved during the run is never overwritten); to relabel an episode use an explicit episode list (API `episode_indices`). Per-episode failures (e.g. a non-JSON reply, with the raw reply) and skips are listed in the job result: shown in the panel after a run and persisted in `EXPORT_ROOT/jobs/<id>.json`. Raw replies are also snapshotted in `meta/annotation_predictions/` when at least one episode was labelled. Other episodes are unaffected.
5. Per episode: check the subtask segments on the timeline, set the failed subtask, move the player and **Set to playhead** (uses the episode-relative player time) for the failure frame, **Save label** (source becomes `human`), then **Mark reviewed**. Changing the label afterwards makes the episode unreviewed again.

Backend HTTP API (all take `local_path=<dataset>`): `GET /api/eval/episodes/{i}`, `POST /api/eval/episodes/{i}/label`, `POST /api/eval/label` (job; poll `/api/annotation/jobs/{id}`), `GET|POST /api/episodes/{i}/review`.

## 5. Files

```
$SESSIONS/s01_.../
  raw/        trial_000.npz, trial_000_frames.npz, ..., results.jsonl   (PC2 session, as pulled)
  s01_....log streamer log
  dataset/    LeRobot v3.0 dataset
    meta/eval_session.json  task, subtasks, failure_reasons, model, date, fps, streamer_args,
                            episodes{<ep>: trial, verdict, reason, note, episode_end, wall_start, wall_stop}, skipped
    meta/eval_labels.json   per episode: vlm_verdict, failed_subtask, failure_frame, rationale, source (vlm|human),
                            model, prompt_version
```

Labels and reviews live in the visualizer workspace draft (`EXPORT_ROOT/drafts/<job>/`, which `/annotate` prepare copies the dataset into) and reach a dataset via **Save dataset**. A re-convert with `--overwrite` does not touch them.

The VLM's subtask starts are stored as `subtask` atoms of the visualizer draft (not in `eval_labels.json`); the operator's verdict is never overwritten.

## 6. Limits

- Label accuracy is about 0.5 s (2 fps) for trials up to 32 s; longer trials are capped at 64 frames (e.g. ~1.9 s spacing at 120 s).
- The VLM is shown the operator's verdict, so its verdict is a consistency check, not an independent one.
- The VLM job needs the VLM endpoint from `/annotate`'s configuration; without it each episode fails with the endpoint error.

## 7. Checks

End-to-end check (2026-10-13, no VLM): the visualizer backend on a converted fake-rig session (2 episodes, episode 1 a fail "grasp failed / missed", note "x"): summary 1 of 2 success with the reason count; saving a human label gives source `human`; review goes reviewed, then unreviewed after a changed label; an all-episodes VLM job without a reachable VLM fails per episode with the error and skips the human-labelled episode. Not yet exercised: **Run VLM labels** against a real VLM and the UI in a browser.

Tests in this environment need `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` (a ROS pytest plugin breaks collection), e.g. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/... -q`.

First real session (user-run): one short session with `TRIALS=2`; check that Enter stops a trial, the verdict prompt appears after the hand-back, both trials are in `results.jsonl`, and the converted dataset opens in the visualizer.
