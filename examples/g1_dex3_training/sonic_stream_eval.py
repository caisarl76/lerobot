"""Score a sonic_policy_streamer.py run recorded by sonic_official_sim_host.py.

Aligns the streamer log with the sim recording by wall-clock time. During the "episode" phase it reports
stability (tilt, foot contacts, leg deviation), the reached palm position and wrist orientation vs. FK of the
dataset's original joint action at the same source frame (pelvis frame, as in the round-trip audit), and the
policy's tokens/hands vs. the dataset's stored ones.
Usage: python sonic_stream_eval.py RUN_DIR JOINT28_ROOT SONIC_ROOT EPISODE
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/code")
from sonic_roundtrip_audit import Robot, v3_episode, v3_meta
from sonic_targets import NOMINAL_BODY

run, src, conv, ep = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
s = np.load(run / "streamer.npz", allow_pickle=True)
sim = np.load(run / "sim" / "sim_state.npz")
rows, info = v3_meta(src)
row = next(r for r in rows if r["episode_index"] == ep)
orig = v3_episode(src, row, info, ["frame_index", "action"])["action"].astype(np.float64)
stored = v3_episode(conv, row, info, ["frame_index", "action"])["action"].astype(np.float64)
robot = Robot()


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def stats(x):
    x = np.asarray(x, float).ravel()
    return {
        "p50": round(float(np.median(x)), 3),
        "p95": round(float(np.percentile(x, 95)), 3),
        "max": round(float(x.max()), 3),
    }


ep_idx = np.flatnonzero(s["phase"] == "episode")
sim_idx = np.clip(np.searchsorted(sim["wall"], s["wall"][ep_idx]), 0, len(sim["wall"]) - 1)
frames = s["frame"][ep_idx]
first = np.r_[True, np.diff(frames) != 0]  # first streamer tick of each source frame
ep_idx, sim_idx, frames = ep_idx[first], sim_idx[first], frames[first]

palm_err, orient_err = [], []
for k, j in zip(frames, sim_idx, strict=True):
    rot = quat_to_rot(sim["pelvis"][j, 3:7])
    reached = (sim["palm"][j] - sim["pelvis"][j, :3]) @ rot  # pelvis frame
    body = np.r_[NOMINAL_BODY[:15], orig[k, :14]]
    intended = robot.fk_pelvis_frame(body)  # also leaves robot.kin at this pose for the wrist frames
    palm_err.append(np.linalg.norm(reached - intended, axis=1))
    for h, wid in enumerate(robot.m.body(n).id for n in ("left_wrist_yaw_link", "right_wrist_yaw_link")):
        r_reached = rot.T @ sim["wrist_R"][j, h]
        r_int = robot.kin.xmat[wid].reshape(3, 3)
        orient_err.append(np.degrees(np.arccos(np.clip((np.trace(r_reached.T @ r_int) - 1) / 2, -1, 1))))
palm_err = np.array(palm_err)

win = (sim["wall"] >= s["wall"][ep_idx[0]]) & (sim["wall"] <= s["wall"][ep_idx[-1]])
q = sim["pelvis"][win, 3:7]
tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1)))
tok = s["token"][ep_idx]
out = {
    "episode": ep, "task": row["tasks"][0], "frames_scored": int(len(frames)), "episode_frames": int(len(orig)),
    "stability": {"tilt_max_deg": round(float(tilt.max()), 2), "floor_contacts_min": int(sim["floor_contacts"][win].min()),
                  "leg_dev_max_rad": round(float(np.abs(sim["body_q"][win, :15] - NOMINAL_BODY[:15]).max()), 3),
                  "pelvis_z_min": round(float(sim["pelvis"][win, 2].min()), 3)},
    "palm_err_vs_original_cm": {k: round(v * 100, 2) for k, v in stats(palm_err).items()},
    "palm_err_p95_cm_left_right": [round(float(np.percentile(palm_err[:, h], 95)) * 100, 2) for h in (0, 1)],
    "wrist_orientation_err_deg": stats(orient_err),
    "policy_token_vs_stored_abs": stats(np.abs(tok - stored[frames, :64])),
    "policy_hands_vs_stored_abs_rad": stats(np.abs(s["hands"][ep_idx] - stored[frames, 64:])),
    "gate_5cm_p95": "PASS" if np.percentile(palm_err, 95) <= 0.05 else "FAIL",
}  # fmt: skip
(run / "stream_eval.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
