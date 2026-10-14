# G1 real-robot evaluation logger (design)

Date: 2026-10-13. Status: approved in chat (sections 1–3), written spec under review.

## Goal

Evaluate a VLA on a G1 manipulation task in sessions of 5–10 trials, where a human decides success, and keep a
complete record of every trial:

1. Run the evaluation process (policy server on the workstation, streamer on PC2, as today).
2. The operator puts the robot and the scene into the starting position.
3. The operator starts the trial; robot state, actions and the head camera are recorded.
4. The operator stops the trial when it is done.
5. In the streamer terminal the operator enters success / fail, and for a failure a reason from a menu.
6. – 7. Reset and repeat for the requested number of trials.
7. The session becomes a LeRobot dataset that `~/work/lerobot-dataset-visualizer` opens.
8. A VLM labels each trial: subtask start frames, the subtask where a failed trial failed and the failure frame, and
   its own success verdict as a check on the operator's.
9. A human reviews and corrects these labels in the visualizer.

Success: one session yields one dataset in which every trial has the operator's verdict and reason, and a
human-reviewed subtask / failure-step label.

## Decisions

- **Scope: the G1 SONIC pipeline only** (`sonic_policy_streamer.py`, `--backend sonic`). No generic LeRobot
  robot/policy API.
- **Steps 1–8 in this repo, steps 9–10 in the visualizer repo** (approach A): the visualizer already has the timeline,
  `subtask` atoms, the VLM endpoint configuration, prediction snapshots and the review mechanism.
- **Failure label = subtask + frame**, against a fixed, ordered subtask list per session.
- **The operator's verdict is authoritative.** The VLM's verdict only flags a disagreement; it never changes the
  operator's verdict.
- **Nothing new is installed on PC2** (its isolated uv env has numpy, pyzmq, msgpack, opencv-python-headless).
  PC2 stores raw trials; the workstation converts them.
- The visualizer's workflow decision Success / Failed / Pending is a curation keep/delete decision (Failed = excluded
  from export). It is **not** used for the evaluation verdict.

## Part 1 — trial recording on PC2 (`examples/g1_dex3_training/sonic_policy_streamer.py`)

### CLI

| Flag                        | Meaning                                                                          |
| --------------------------- | -------------------------------------------------------------------------------- |
| `--trials N`                | Run N trials in one process. Without it the streamer behaves exactly as today.   |
| `--session-dir DIR`         | Session output directory (required with `--trials`; replaces `--log`).           |
| `--failure-reasons "a,b,…"` | Failure menu. Default below.                                                     |
| `--duration-s`              | With `--trials`: the per-trial safety cap (the operator normally stops earlier). |

`--trials` requires `--images zmq`, `--backend sonic` and `--end planner` (the deploy must keep running in planner
mode between trials); the parser rejects other combinations.

Default failure reasons: `did not reach the object`, `grasp failed / missed`, `dropped the object`,
`wrong object / target`, `collision (table, knocked over)`, `stalled / froze`, `safety stop (watchdog)`, `other`.

### Backend split

`SonicBackend.start()` today does both one-time and per-trial work. It is split into:

- `connect()` — once: `deploy_ready` gate, `start` command in PLANNER mode.
- `take_over()` — per trial: `settled` gate, read the planner's current token, PLANNER → POSE handoff blend.

`start()` stays as `connect()` + `take_over()` so the single-episode path is unchanged. `finish()` already hands
back to the planner with `--end planner`.

### Per-trial loop

The policy is warmed up once; `InferenceWorker`, sockets and `LiveImages` live for the whole session. For trial `i`:

1. `take_over()`; the `settled` prompt names the trial (`trial 3/10: move the robot into position…`).
2. Reset per-trial state (log, history, latencies, chunk slots); `--hold-rest-s` hold, first chunk, `begin()`.
3. Record: the existing 50 Hz log, plus every head-camera JPEG the `LiveImages` receiver thread receives, kept as the
   received bytes with `time.time()` receive time (recording is switched on/off by the main loop; no re-encoding).
4. The episode loop ends on: operator stop (Enter: the 50 Hz loop checks `select.select([sys.stdin], [], [], 0)`
   once per tick, no extra thread), the duration cap, or the existing watchdog / state-age / chunk-age stops.
   `episode_end` records which (`operator stop`, `completed`, or the existing reasons).
5. `finish()` (blend back, hand back to the planner).
6. Verdict prompt: `[s]uccess / [f]ail / [r]edo`. Fail → numbered reason menu (a watchdog stop pre-selects
   `safety stop (watchdog)`; Enter accepts it) → optional free-text note (Enter skips). Redo discards the trial's
   files and repeats trial `i`.
7. Write the trial immediately (atomic: temp file + `os.replace`), so Ctrl-C or a crash keeps every finished trial:
   - `trial_{i:03d}.npz`: today's log keys, `episode_end`, `args`, `verdict`, `reason`, `note`, `wall_start`,
     `wall_stop` (first / last episode-phase tick).
   - `trial_{i:03d}_frames.npz`: `jpeg` (object array of bytes), `t_recv` (float64), `camera` (stream name). The
     JPEGs hold RGB data (`g1_head_camera_server.py` encodes RGB arrays without conversion).
   - one JSON line appended to `results.jsonl`: trial, verdict, reason, note, episode_end, wall times.

Cost: ~40 KB × 30 fps ≈ 70 MB per 60 s trial in memory and on disk; no extra CPU for encoding.

### Gate-dir mode (sim, unattended tests)

With `--gate-dir`, trial `i` waits for `settled.{i}`, stops on `stop.{i}`, and reads its verdict from
`verdict.{i}` (JSON: `{"verdict": "fail", "reason": 2, "note": ""}`). This lets a sim run of `--trials 2` exercise the
whole loop with a driver that writes these files.

### Launcher

`g1_groot_real_run.sh session SESSION [TASK] [extra args]`: as `stream`, with `--trials ${TRIALS:-10}
--session-dir ../sessions/SESSION --duration-s ${MAX_S:-120}` instead of `--log`.
`g1_groot_real_run.sh pull SESSION`: rsync `pc2_222:~/g1_sonic_eval/sessions/SESSION/` to the workstation
(`${SESSIONS:-/mnt/data/jihun/g1_eval_sessions}/SESSION/raw/`).

## Part 2 — conversion to a LeRobot dataset (workstation, `examples/g1_dex3_training/eval_session_to_lerobot.py`)

`eval_session_to_lerobot.py --session-dir RAW --out ROOT --repo-id local/<session> --subtasks "reach,grasp,…"
[--fps 30]` writes a LeRobot **v3.0** dataset with `LeRobotDataset.create` / `add_frame` / `save_episode`.

- One episode per kept trial (redo trials do not exist on disk), episode index = order of trial index.
- Episode span: the policy-controlled phase only (`phase` rows of the episode, first predicted chunk to the stop);
  the handoff and handback blends are excluded.
- Frames at `--fps` (default 30, the camera rate): frame `k` at `wall_start + k / fps`.
  - `observation.images.<camera>` (e.g. `egocentric`): latest JPEG with `t_recv ≤` frame time, decoded, encoded to
    video by LeRobot.
  - `observation.state` (28D): latest 50 Hz log tick with `wall ≤` frame time.
  - `action`: the command that was sent — `token` + `hands` (78D) for `sonic78`; `joint_ref` (28D / 31D) for joint
    action spaces.
  - `task`: the session's `--task`.
- A camera gap > 0.2 s inside an episode is printed as a warning with its time and length; conversion continues.
- `meta/eval_session.json`:
  ```json
  {"task": "...", "subtasks": ["reach", "grasp", "lift", "handover"], "failure_reasons": ["..."],
   "model": "<--policy-server / checkpoint from args>", "streamer_args": {...}, "date": "2026-10-13",
   "episodes": {"0": {"trial": 0, "verdict": "fail", "reason": "grasp failed / missed", "note": "",
                      "episode_end": "operator stop", "wall_start": 0.0, "wall_stop": 0.0}}}
  ```
- Layout: `pull` puts the raw session in `<SESSIONS>/<session>/raw/`, the converter writes `<SESSIONS>/<session>/dataset/`
  (its `--out`); the converter never modifies `raw/`.
- The converter prints the session summary: success n/N and counts per failure reason.

## Part 3 — VLM labels and human review (`~/work/lerobot-dataset-visualizer`)

### Labeling: `backend/eval_labels.py`, `POST /api/eval/label`

Input: a dataset (local path) and optionally episode indices (default: all). Per episode:

- Frames sampled at 2 fps (at most 64), each annotated with its frame index and timestamp.
- Prompt context from `meta/eval_session.json`: task, ordered subtask list, operator verdict, reason and note.
- VLM endpoint: the one configured for `/annotate` (`config.vlm`, `LEROBOT_VLM_API_KEY` / `GENON_API_KEY`); no new
  model service.
- Required JSON reply:
  ```json
  {
    "subtask_starts": [
      { "subtask": "reach", "frame": 0 },
      { "subtask": "grasp", "frame": 212 }
    ],
    "vlm_verdict": "fail",
    "failed_subtask": "grasp",
    "failure_frame": 268,
    "rationale": "fingers close beside the bottle; the bottle stays on the table"
  }
  ```
  Validation: subtasks from the list and in list order, frames inside the episode and increasing,
  `failed_subtask` / `failure_frame` present iff `vlm_verdict == "fail"` (`failure_frame` ≥ that subtask's start).
  Frames are snapped to the episode's exact frame timestamps. An invalid reply fails that episode explicitly; its raw
  reply is kept in the job result. No silent repair.
- Writes:
  - `subtask_starts` → the existing `subtask` persistent atoms (start timestamp per subtask), through the same save
    path as manual edits, so they show and edit on the existing timeline.
  - failure label → `meta/eval_labels.json`, per episode: `vlm_verdict`, `failed_subtask`, `failure_frame`,
    `rationale`, `source` (`vlm` / `human`), model id, prompt version.
  - the raw replies → a snapshot in `meta/annotation_predictions/` (existing mechanism).

### Review: Evaluation panel

In the episode view, shown when `meta/eval_session.json` exists:

- Operator verdict, reason, note (read-only).
- VLM verdict with a **disagreement** badge when it differs from the operator.
- Failed subtask (dropdown from the session list) and failure frame, with **Set failure frame = playhead**; the
  failure frame is a marker on the existing timeline. Saving sets `source: human`.
- **Mark reviewed** is the existing review flow. Its hash (`annotation_history`) is extended to cover the episode's
  `eval_labels.json` entry, so editing subtask atoms or the failure label invalidates the review.

Session summary (dataset view): success rate, counts per failure reason, VLM / operator disagreements, reviewed
count.

Export (**Save dataset**) carries `meta/eval_session.json` and `meta/eval_labels.json`.

### Not in scope

- A second, zoomed VLM pass around the failure (labels are ~0.5 s accurate at 2 fps; the reviewer sets the exact
  frame). Add when reviewers often move the failure frame by more than 1 s.
- Robot-state plots in the VLM prompt.
- Other robots and other backends (`--backend decoupled`).

## Error handling

- Streamer: a trial is written before the next one starts; files are written atomically. A watchdog / stale-state
  stop still asks for the verdict. A failed policy warm-up or lost deploy connection ends the session with the
  finished trials on disk.
- Converter: refuses a session without `results.jsonl` or with trials whose `.npz` / frames file is missing (names
  them); refuses an existing `--out` unless `--overwrite`.
- Labeler: per-episode failures do not stop the job; the job result lists them with the raw replies.

## Testing

- Converter: one pytest builds a synthetic two-trial session (fake `.npz` logs and a few JPEGs with known receive
  times) and checks episode count, frame ↔ tick and frame ↔ JPEG alignment, action width and `eval_session.json`.
- Streamer: a sim run with `--trials 2 --gate-dir` driven by a small script that writes `settled.i`, `stop.i`,
  `verdict.i`; check two trials, `results.jsonl`, and that the converter accepts the session.
- Visualizer backend: pytest with a fake VLM client (canned replies): parsing, validation rejects (unknown subtask,
  wrong order, frame out of range, missing failure fields), atom and label writes, review invalidation after a label
  edit.
- Visualizer frontend: one `bun test` for the panel (renders verdicts, disagreement badge, playhead → failure
  frame), plus the repo's `bun run format && bun run validate`.
- Real robot: one short session (2–3 trials) end to end before relying on it.
