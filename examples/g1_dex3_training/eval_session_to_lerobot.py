"""Convert an evaluation session of sonic_policy_streamer.py --trials (pulled from PC2) to a LeRobot v3.0 dataset.

One episode per kept trial (results.jsonl order) over the policy-controlled phase, wall_start..wall_stop (the first /
last "episode" tick); trials that never reached it (or have no camera frames) are skipped and listed under
"skipped" with their full results.jsonl line plus "why", so the verdict still counts. Frame k is at
wall_start + k / fps and takes the latest camera JPEG and the latest 50 Hz log tick at or before that time:
  observation.images.<stream>  the head camera as received (the JPEGs hold RGB data, so no colour conversion)
  observation.state (28)       measured arms + Dex3 hands (sonic_targets.ACTION_NAMES order)
  action                       what the streamer sent: token 64 + hands 14 (sonic78), or joint_ref 28 / 31
meta/eval_session.json carries the operator's verdicts and the session settings for the visualizer's Evaluation
panel (the subtask list is given here: the VLM labels the failure step against it).

Usage: eval_session_to_lerobot.py --session-dir RAW --out DATASET --subtasks "reach,grasp,lift,handover"
                                  [--repo-id local/NAME] [--fps 30] [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from eval_session import load_frames, parse_reasons, session_mismatch, summarize
from sonic_targets import ACTION_NAMES

from lerobot.datasets.lerobot_dataset import LeRobotDataset

WAIST_NAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")  # motors 12-14
MAX_CAMERA_GAP_S = 0.2


def frame_rows(times: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Index of the latest of the sorted `times` at or before each of `at` (the first one before the first)."""
    return np.clip(np.searchsorted(times, at, side="right") - 1, 0, len(times) - 1)


def action_rows(log, action_space: str) -> tuple[np.ndarray, list[str]]:
    if action_space == "sonic78":
        names = [f"sonic_token_{i}" for i in range(64)] + list(ACTION_NAMES[14:])
        return np.concatenate([log["token"], log["hands"]], axis=1), names
    width = 31 if action_space == "joint31" else 28
    return log["joint_ref"][:, :width], (list(ACTION_NAMES) + list(WAIST_NAMES))[:width]


def convert(session_dir: Path, out: Path, repo_id: str, subtasks: list[str], fps: int = 30) -> dict:
    session_dir, out = Path(session_dir), Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists (use --overwrite)")
    lines = [json.loads(s) for s in (session_dir / "results.jsonl").read_text().splitlines()]
    trials = []
    for line in lines:
        base = session_dir / f"trial_{line['trial']:03d}"
        for path in (base.with_suffix(".npz"), Path(f"{base}_frames.npz")):
            if not path.exists():
                raise FileNotFoundError(f"missing {path.name} for trial {line['trial']}")
        trials.append((line, np.load(base.with_suffix(".npz")), load_frames(Path(f"{base}_frames.npz"))))
    args = json.loads(str(trials[0][1]["args"]))
    for line, log, _ in trials[1:]:
        if bad := session_mismatch(args, json.loads(str(log["args"]))):
            raise ValueError(f"trial {line['trial']}: {', '.join(bad)} differ from trial 0 (a mixed session)")
    if len(models := {line.get("model") for line, _, _ in trials} - {None}) > 1:
        raise ValueError(f"trials were served by different models {sorted(models)} (a mixed session)")
    action_space = args.get("action_space", "sonic78")
    skipped, kept = [], []
    for line, log, frames in trials:
        if not np.isfinite(line["wall_start"]):
            print(f"[convert] trial {line['trial']}: no episode ticks, skipped")
            # NaN wall times become null: strict JSON readers (the visualizer's API) reject NaN
            skipped.append({**line, "wall_start": None, "wall_stop": None, "why": "no episode ticks"})
        elif not len(frames[2]):
            print(f"[convert] trial {line['trial']}: no camera frames, skipped")
            skipped.append({**line, "why": "no camera frames"})
        else:
            kept.append((line, log, frames))
    if not kept:
        raise ValueError("no trial reached the episode phase")
    cameras = sorted(set(kept[0][2][1]))
    shapes = {
        c: cv2.imdecode(
            np.frombuffer(kept[0][2][2][list(kept[0][2][1]).index(c)], np.uint8), cv2.IMREAD_COLOR
        ).shape
        for c in cameras
    }
    _, action_names = action_rows(kept[0][1], action_space)
    features = {
        "observation.state": {"dtype": "float32", "shape": (28,), "names": list(ACTION_NAMES)},
        "action": {"dtype": "float32", "shape": (len(action_names),), "names": action_names},
        **{
            f"observation.images.{c}": {
                "dtype": "video",
                "shape": shapes[c],
                "names": ["height", "width", "channels"],
            }
            for c in cameras
        },
    }
    ds = LeRobotDataset.create(
        repo_id, fps=fps, root=out, robot_type="unitree_g1_dex3_sonic", features=features
    )
    episodes = {}
    for ep, (line, log, (t_recv, names, jpegs)) in enumerate(kept):
        n = int(np.floor((line["wall_stop"] - line["wall_start"]) * fps)) + 1
        at = line["wall_start"] + np.arange(n) / fps
        rows = frame_rows(log["wall"], at)
        state = log["state"][rows].astype(np.float32)
        action = action_rows(log, action_space)[0][rows].astype(np.float32)
        per_cam = {}
        for c in cameras:
            sel = np.flatnonzero(names == c)
            t_c = t_recv[sel]
            ws, we = line["wall_start"], line["wall_stop"]
            # the JPEG shown at wall_start (or wall_start itself if none came before), those inside, then wall_stop
            before = t_c[t_c <= ws][-1:]
            times = np.concatenate([before if len(before) else [ws], t_c[(t_c > ws) & (t_c <= we)], [we]])
            for gap_at, gap in zip(times[:-1], np.diff(times), strict=True):
                if gap > MAX_CAMERA_GAP_S:
                    print(
                        f"[convert] trial {line['trial']} {c}: camera gap {gap:.2f} s at {gap_at - ws:.2f} s"
                    )
            per_cam[c] = sel[frame_rows(t_c, at)]
        for k in range(n):
            frame = {"observation.state": state[k], "action": action[k], "task": args["task"]}
            for c in cameras:
                frame[f"observation.images.{c}"] = cv2.imdecode(
                    np.frombuffer(jpegs[per_cam[c][k]], np.uint8), cv2.IMREAD_COLOR
                )
            ds.add_frame(frame)
        ds.save_episode()
        episodes[str(ep)] = {
            k: line[k]
            for k in ("trial", "verdict", "reason", "note", "episode_end", "wall_start", "wall_stop", "model")
            if k in line
        }
    ds.finalize()
    meta = {
        "task": args["task"],
        "subtasks": subtasks,
        "failure_reasons": parse_reasons(args.get("failure_reasons")),
        "model": next(iter(models), None) or args.get("policy_server") or args.get("policy_path"),
        "streamer_args": args,
        "date": datetime.fromtimestamp(kept[0][0]["wall_start"]).date().isoformat(),
        "fps": fps,
        "episodes": episodes,
        "skipped": skipped,
    }
    (out / "meta/eval_session.json").write_text(json.dumps(meta, indent=2))
    print(f"[convert] {out}: {len(kept)} episodes; {summarize(session_dir)}")
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session-dir", type=Path, required=True, help="raw session (g1_groot_real_run.sh pull)")
    p.add_argument("--out", type=Path, required=True, help="new dataset directory")
    p.add_argument("--subtasks", required=True, help="the task's ordered subtasks, comma-separated")
    p.add_argument("--repo-id", help="default: local/<out directory name>")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--overwrite", action="store_true", help="replace an existing --out")
    a = p.parse_args()
    subtasks = [s.strip() for s in a.subtasks.split(",") if s.strip()]
    if not subtasks:
        p.error("--subtasks is empty")
    if a.overwrite and a.out.exists():
        out, sess = a.out.resolve(), a.session_dir.resolve()
        if out == sess or out in sess.parents:
            p.error(
                "--out is the session directory or one of its parents; refusing to delete the raw session"
            )
        if not (sess / "results.jsonl").exists():
            p.error(f"{a.session_dir} has no results.jsonl; refusing --overwrite")
        shutil.rmtree(a.out)
    convert(a.session_dir, a.out, a.repo_id or f"local/{a.out.name}", subtasks, a.fps)


if __name__ == "__main__":
    main()
