"""Bottle nominal position and handover zone for tasks/handover_bottle.json from MuJoCo backend A replays of the
held-out handover episodes (spec: Bottle position and handover zone). Usage:
python derive_handover_task.py RUN_DIR [RUN_DIR ...] --table-gap-cm 18 --out tasks/handover_bottle.json
The bottle is the mean grasp point of the runs relative to the pelvis; --table-gap-cm places the table for the trials
(the replays may have run with another gap) and must leave the bottle on the table.
"""

import argparse
import json
from pathlib import Path

import numpy as np

# hand -> (wrist_yaw_link axis, sign) the palm faces: the fingertips and the thumb tip move along it when the hand
# closes (MuJoCo FK of the replays' closed hand pose: left -y, right +y)
PALM_NORMAL = {0: (1, -1.0), 1: (1, 1.0)}
PALM_HALF_THICKNESS = 0.015
RADIUS, HEIGHT, TABLE_TOP = 0.033, 0.20, 0.80


def events(hand_targets, lo, hi, min_closed: int = 25):
    """(grasp hand, close index, release index) of the first close -> reopen cycle of either hand: its mean normalised
    target rises through 50 % (a hand closed from the first row, still the planner-mode fist, counts only once it has
    opened) and later falls below it again (a handover releases; a hand clenched to the end, an idle fist, is not the
    grasp). A close shorter than min_closed records (0.5 s at 50 Hz) is a flicker through the threshold, not a
    grasp. Without any cycle: the first close and the last index."""
    x = (np.asarray(hand_targets) - lo) / np.where(hi > lo, hi - lo, 1.0)
    closed = np.stack([x[:, :7].mean(1) > 0.5, x[:, 7:].mean(1) > 0.5], 1)
    n, best = len(closed), None
    for hand in (0, 1):
        c = closed[:, hand]
        rises = np.flatnonzero(c[1:] & ~c[:-1]) + 1
        for close in rises:
            reopen = np.flatnonzero(~c[close:])
            if reopen.size and reopen[0] < min_closed:
                continue  # flicker
            cycle = (reopen.size > 0, -int(close))
            if best is None or cycle > best[0]:
                best = (cycle, hand, int(close), int(close + reopen[0]) if reopen.size else n - 1)
            break  # first close of this hand only
    if best is None:
        raise ValueError("no hand closes during the episode")
    return best[1], best[2], best[3]


def episode_rows(sim_wall, streamer_wall, phase) -> np.ndarray:
    """Indices of the sim records inside the streamer's "episode" phase (wall clock): the deploy holds the Dex3 hands
    closed in planner mode, so a grasp seen before or after the episode is not one."""
    w = np.asarray(streamer_wall)[np.asarray(phase) == "episode"]
    return np.flatnonzero((np.asarray(sim_wall) >= w.min()) & (np.asarray(sim_wall) <= w.max()))


def bottle_from_palm(palm, wrist_r, offset_m, hand):
    axis, sign = PALM_NORMAL[hand]
    return np.asarray(palm) + sign * offset_m * np.asarray(wrist_r)[:, axis]


def quat_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--table-gap-cm", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    per = []
    for run in a.runs:
        s = np.load(run / "sim" / "sim_state.npz")
        log = np.load(run / "streamer.npz", allow_pickle=True)
        rows = episode_rows(s["wall"], log["wall"], log["phase"])
        hands = s["hand_q"][rows]  # dataset order, left 7 then right 7 (sim_scene.HANDS)
        hand, close, release = events(hands, hands.min(0), hands.max(0))
        close, release = rows[close], rows[release]
        pelvis0 = s["pelvis"][0]
        b = bottle_from_palm(
            s["palm"][close, hand], s["wrist_R"][close, hand], RADIUS + PALM_HALF_THICKNESS, hand
        )
        pel_r = s["pelvis"][release]
        zone = quat_matrix(pel_r[3:7]).T @ (s["palm"][release, hand] - pel_r[:3])
        per.append({"run": run.name, "grasp_hand": hand, "close_t": float(s["t"][close]),
                    "release_t": float(s["t"][release]), "bottle_x_from_pelvis": float(b[0] - pelvis0[0]),
                    "bottle_y_from_pelvis": float(b[1] - pelvis0[1]), "bottle_z": float(b[2]),
                    "zone_center_pelvis": zone.tolist()})  # fmt: skip
    hands = {r["grasp_hand"] for r in per}
    if len(hands) != 1:
        raise SystemExit(f"episodes grasp with different hands: {per}")
    bx = float(np.mean([r["bottle_x_from_pelvis"] for r in per]))
    by = float(np.mean([r["bottle_y_from_pelvis"] for r in per]))
    edge0 = 0.08 + a.table_gap_cm / 100  # robot at the origin after the trial reset
    if bx - RADIUS - 0.01 < edge0:
        raise SystemExit(
            f"bottle at x {bx:.3f} m is not on a table whose edge is at {edge0:.3f} m: use a smaller gap"
        )
    cfg = {
        "table_xy_world": [edge0 + 0.4, 0.0],
        "bottle_xy_world": [round(bx, 4), round(by, 4)],
        "jitter_m": 0.03,
        "judge": {
            "table_top_z": TABLE_TOP,
            "bottle_radius": RADIUS,
            "bottle_height": HEIGHT,
            "zone_center_pelvis": np.mean([r["zone_center_pelvis"] for r in per], 0).round(4).tolist(),
            "zone_radius": 0.08,
            "grasp_hand": hands.pop(),
        },  # fmt: skip
        "table_gap_cm": a.table_gap_cm,
        "per_episode": per,
        "source_runs": [r["run"] for r in per],
    }
    a.out.write_text(json.dumps(cfg, indent=1) + "\n")
    print(json.dumps(cfg["judge"]), cfg["bottle_xy_world"])


if __name__ == "__main__":
    main()
