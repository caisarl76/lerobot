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

Each finished trial is written at once. A crash or Ctrl-C keeps the finished trials; the same command continues at the next trial (`results.jsonl` line count = next trial number). A resume with another task, action space or policy (server/path) than trial 1 is refused before the robot moves. Each trial records the checkpoint that served it (`model`, asked from the policy server's `info` at the trial start; servers started before 2026-10-14 do not report it, so restart them). A different checkpoint behind the same port than trial 1's ends the session before the next takeover: use a new session name per model.

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
4. **Run VLM labels** (bulk) labels only episodes that have no eval label yet (checked again right before each write, so a label saved during the run is never overwritten); to relabel an episode use an explicit episode list (API `episode_indices`). Per-episode failures (e.g. a non-JSON reply, with the raw reply) and skips are listed in the job result: shown in the panel after a run and persisted in `EXPORT_ROOT/jobs/<id>.json`. Raw replies are also snapshotted in `meta/annotation_predictions/`, even when no episode was labelled. The endpoint is the run's saved `/annotate` VLM setting, else the backend config (`deployment/annotation-h100.json`: `127.0.0.1:34002`, which needs `deployment/annotation-vlm-tunnel.sh` running; without it every episode fails with `APIConnectionError`). Other episodes are unaffected.
5. Per episode: check the subtask segments on the timeline, set the failed subtask, move the player and **Set to playhead** (uses the episode-relative player time) for the failure frame, **Save label** (source becomes `human`), then **Mark reviewed**. Changing the label afterwards makes the episode unreviewed again.

Backend HTTP API (all take `local_path=<dataset>`): `GET /api/eval/episodes/{i}`, `POST /api/eval/episodes/{i}/label`, `POST /api/eval/label` (job; poll `/api/annotation/jobs/{id}`), `GET|POST /api/episodes/{i}/review`.

## 5. Files

```
$SESSIONS/s01_.../
  raw/        trial_000.npz, trial_000_frames.npz, ..., results.jsonl   (PC2 session, as pulled)
  s01_....log streamer log
  dataset/    LeRobot v3.0 dataset
    meta/eval_session.json  task, subtasks, failure_reasons, model, date, fps, streamer_args,
                            episodes{<ep>: trial, verdict, reason, note, episode_end, wall_start, wall_stop, model}, skipped
    meta/eval_labels.json   per episode: vlm_verdict, failed_subtask, failure_frame, rationale, source (vlm|human),
                            model, prompt_version
```

Labels and reviews live in the visualizer workspace draft (`EXPORT_ROOT/drafts/<job>/`, which `/annotate` prepare copies the dataset into) and reach a standalone dataset via **Review & export → Export & publish** (Rich annotations, HF repository blank for a local export), which carries `eval_session.json` / `eval_labels.json` and the reviews (renumbered and clipped with the export). A re-convert with `--overwrite` does not touch them.

The VLM's subtask starts are stored as `subtask` atoms of the visualizer draft (not in `eval_labels.json`); the operator's verdict is never overwritten.

## 6. Limits

- Label accuracy is about 0.5 s (2 fps) for trials up to 32 s; longer trials are capped at 64 frames (e.g. ~1.9 s spacing at 120 s).
- The VLM is shown the operator's verdict, so its verdict is a consistency check, not an independent one.
- The VLM job needs the VLM endpoint from `/annotate`'s configuration; without it each episode fails with the endpoint error.

## 7. Checks

End-to-end check (2026-10-13, no VLM): the visualizer backend on a converted fake-rig session (2 episodes, episode 1 a fail "grasp failed / missed", note "x"): summary 1 of 2 success with the reason count; saving a human label gives source `human`; review goes reviewed, then unreviewed after a changed label; an all-episodes VLM job without a reachable VLM fails per episode with the error and skips the human-labelled episode. Not yet exercised: **Run VLM labels** against a real VLM and the UI in a browser.

Tests in this environment need `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` (a ROS pytest plugin breaks collection), e.g. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/... -q`.

First real session (2026-10-14, `s00_test_he_groot_laptop`, HE GR00T, "close a laptop g1", `TRIALS=2 MAX_S=60`): both trials fail "stalled / froze" (laptop half closed); converted to 2 episodes of 60 s; Qwen3.6-27B labelled both as fail in "push the lid down" (frames 274 and 357), agreeing with the operator; both labels saved and reviewed in the browser.

## 8. Future work: safe hand settle

**Status: not started (marked by the user, 2026-10-21).**

- **Problem:** at every episode end (operator stop, time cap, "no valid chunk", watchdog, stale state or camera)
  `SonicBackend.finish()` (`stream_backends.py`) blends from the last sent token straight to the rest token in 1 s,
  then hands back to PLANNER mode. The blend is in token space and knows nothing about the table, so a raised right
  hand sweeps down through whatever is below it: on 2026-10-21 it hit the black box. The line
  `token slew limit active on N ticks` printed just before it is only a counter, not the trigger.
- **Interim fix (requested 2026-10-21, session lerobot-11):** hold the current pose at the episode end, ask the operator
  to clear the space and press Enter, then run the existing blend and handback.
- **Safe hand settle (this item):** return to the rest pose automatically and without contact, so the operator does not
  have to clear the space. Candidates:
  1. _Retrace:_ play the episode's own streamed tokens in reverse (every tick is logged) back to the start pose, then
     the usual handback. The hand came in along that path without hitting anything; the streamer already plays a
     reversed path at the table (`rev_tokens`, "table shutdown"). Needs a speed cap, skipping stalled or oscillating
     stretches, and a rule for the hands (keep holding, or open only at a safe point).
  2. _Lift, then lower:_ first move the hand up and back to a clearance waypoint above the table, then descend to rest.
  3. Choose the route from the measured arm pose and the known table height.
- **Done when:** no contact with the table or the box from varied end poses (mid-reach, holding in the handover
  posture, watchdog stop), checked in sim with the table scene first (Isaac handover eval), then on the robot.
