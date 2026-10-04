"""Score a sonic_policy_streamer.py run recorded by a sim host (sonic_official_sim_host.py or
decoupled_wbc_sim_host.py).

Validity first (wbc_common.run_validity, same rule for both backends): runs that did not complete, cover < 98% of the
episode or (backend C) held their targets on > 2% of records are reported as {"valid": false} and exit 2.
Then, during the "episode" phase: stability; palm error in the pelvis frame against FK of the recorded action
("original"), of the streamer's 50 Hz joint_ref (controller error; joint runs only) and, for backend C, of the applied
target after clipping/hold; wrist orientation; table contacts; arm speed / jerk and palm jerk; for 31D runs the torso
orientation reached vs commanded and (C) the lower-body policy's rpy command vs FK of the commanded waist; for token
runs the tokens/hands vs the dataset's stored ones. A-native runs (token policies, stored tokens) have no joint_ref.
Usage: python sonic_stream_eval.py RUN_DIR JOINT28_ROOT SONIC_ROOT EPISODE
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/code")
from robot_run_smoothness import windowed_diff
from sonic_targets import NOMINAL_BODY
from wbc_common import log_settings, rpy_from_matrix, run_validity


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def stats(x):
    x = np.asarray(x, float).ravel()
    return {"p50": round(float(np.median(x)), 3), "p95": round(float(np.percentile(x, 95)), 3), "max": round(float(x.max()), 3)}  # fmt: skip


def cm(x):
    return {k: round(v * 100, 2) for k, v in stats(x).items()}


def body_from(arms, waist):
    b = NOMINAL_BODY.copy()
    b[15:29], b[12:15] = arms, waist
    return b


def score(run: Path, src: str, conv: str, ep: int) -> dict:
    for rel in ("streamer.npz", "sim/sim_state.npz"):
        if not (run / rel).exists():
            return {"episode": ep, "valid": False, "reason": f"missing {rel}"}
    from sonic_roundtrip_audit import Robot, v3_episode, v3_meta

    s = np.load(run / "streamer.npz", allow_pickle=True)
    sim = np.load(run / "sim" / "sim_state.npz")
    term_path = run / "sim" / "termination.json"
    term = json.loads(term_path.read_text()) if term_path.exists() else None
    backend, action_space = log_settings(str(s["args"]))
    rows, info = v3_meta(src)
    row = next(r for r in rows if r["episode_index"] == ep)
    orig = v3_episode(src, row, info, ["frame_index", "action"])["action"].astype(np.float64)
    ep_all = np.flatnonzero(s["phase"] == "episode")
    win = np.zeros(len(sim["wall"]), bool)
    if len(ep_all):
        win = (sim["wall"] >= s["wall"][ep_all[0]]) & (sim["wall"] <= s["wall"][ep_all[-1]])
    held = (
        float(np.mean(sim["held"][win]))
        if backend == "decoupled" and "held" in sim.files and win.any()
        else 0.0
    )
    valid, reason = run_validity(term, s["phase"], s["frame"], s["wall"], sim["wall"], len(orig), held)
    head = {"episode": ep, "backend": backend, "action_space": action_space}
    if not valid:
        return {**head, "valid": False, "reason": reason}

    robot = Robot()
    torso_id = robot.m.body("torso_link").id
    wrist_ids = [robot.m.body(n).id for n in ("left_wrist_yaw_link", "right_wrist_yaw_link")]
    sim_idx = np.clip(np.searchsorted(sim["wall"], s["wall"][ep_all]), 0, len(sim["wall"]) - 1)
    frames = s["frame"][ep_all]
    first = np.r_[True, np.diff(frames) != 0]  # first streamer tick of each source frame
    ep_idx, sim_idx, frames = ep_all[first], sim_idx[first], frames[first]
    jref = s["joint_ref"][ep_idx] if "joint_ref" in s.files else None
    has_ref = jref is not None and np.isfinite(jref).all()
    waist31 = action_space == "joint31" and has_ref
    applied = sim["applied_ref"] if "applied_ref" in sim.files else None
    nominal_waist = NOMINAL_BODY[12:15]

    palm_orig, palm_ref, palm_app, orient_err, torso_err, rpy_err, waist_cmd = [], [], [], [], [], [], []
    for i, (k, j) in enumerate(zip(frames, sim_idx, strict=True)):
        rot = quat_to_rot(sim["pelvis"][j, 3:7])
        reached = (sim["palm"][j] - sim["pelvis"][j, :3]) @ rot  # pelvis frame
        waist = jref[i, 28:31] if waist31 else nominal_waist
        palm_orig.append(
            np.linalg.norm(reached - robot.fk_pelvis_frame(body_from(orig[k, :14], waist)), axis=1)
        )
        for h, wid in enumerate(wrist_ids):  # robot.kin is at the recorded-action pose now
            r_reached = rot.T @ sim["wrist_R"][j, h]
            r_int = robot.kin.xmat[wid].reshape(3, 3)
            orient_err.append(np.degrees(np.arccos(np.clip((np.trace(r_reached.T @ r_int) - 1) / 2, -1, 1))))
        if has_ref:
            palm_ref.append(
                np.linalg.norm(reached - robot.fk_pelvis_frame(body_from(jref[i, :14], waist)), axis=1)
            )
            if (
                waist31
            ):  # torso orientation relative to the yaw-only pelvis frame, as upstream defines the command
                cmd_rpy = rpy_from_matrix(robot.kin.xmat[torso_id].reshape(3, 3))
                yaw = np.arctan2(rot[1, 0], rot[0, 0])
                rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
                torso_err.append(np.abs(rpy_from_matrix(rz.T @ sim["torso_R"][j]) - cmd_rpy))
                waist_cmd.append(np.abs(jref[i, 28:31]).max())
                if "rpy_cmd" in sim.files:
                    rpy_err.append(np.abs(sim["rpy_cmd"][j] - cmd_rpy))
        if applied is not None:
            ap = applied[j]
            ap_waist = ap[28:31] if waist31 else nominal_waist
            palm_app.append(
                np.linalg.norm(reached - robot.fk_pelvis_frame(body_from(ap[:14], ap_waist)), axis=1)
            )
    palm_orig = np.array(palm_orig)
    if rpy_err and max(waist_cmd) > 0.05 and np.abs(sim["rpy_cmd"][win]).max() < 1e-3:
        return {
            **head,
            "valid": False,
            "reason": "waist commanded but the lower-body policy's rpy command stayed 0",
        }

    q = sim["pelvis"][win, 3:7]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1)))
    arm, palm = sim["body_q"][win, 15:29], sim["palm"][win]
    jerk = np.abs(windowed_diff(windowed_diff(windowed_diff(arm)))).max(1)
    palm_jerk = np.linalg.norm(windowed_diff(windowed_diff(windowed_diff(palm))), axis=-1).max(1)
    out = {
        **head, "valid": True, "task": row["tasks"][0], "frames_scored": int(len(frames)), "episode_frames": int(len(orig)),
        "stability": {"tilt_max_deg": round(float(tilt.max()), 2), "floor_contacts_min": int(sim["floor_contacts"][win].min()),
                      "leg_dev_max_rad": round(float(np.abs(sim["body_q"][win, :12] - NOMINAL_BODY[:12]).max()), 3),
                      "pelvis_z_min": round(float(sim["pelvis"][win, 2].min()), 3)},
        "palm_err_vs_original_cm": cm(palm_orig),
        "palm_err_p95_cm_left_right": [round(float(np.percentile(palm_orig[:, h], 95)) * 100, 2) for h in (0, 1)],
        "palm_err_vs_joint_ref_cm": cm(palm_ref) if palm_ref else None,
        "palm_err_vs_applied_cm": cm(palm_app) if palm_app else None,
        "wrist_orientation_err_deg": stats(orient_err),
        "table": {"hits_max": int(sim["table_hits"][win].max()), "hit_records": int((sim["table_hits"][win] > 0).sum()),
                  "clear_min_cm": round(float(np.nanmin(sim["table_clear"][win])) * 100, 2) if np.isfinite(sim["table_clear"][win]).any() else None},
        "smoothness": {"arm_speed_p95": round(float(np.percentile(np.abs(np.diff(arm, axis=0)).max(1) / 0.02, 95)), 3),
                       "arm_jerk_p95": round(float(np.percentile(jerk, 95)), 1),
                       "palm_jerk_p95": round(float(np.percentile(palm_jerk, 95)), 1)},
        "torso_err_rad": {a: stats(np.array(torso_err)[:, n]) for n, a in enumerate(("roll", "pitch", "yaw"))} if torso_err else None,
        "rpy_cmd_err_rad": stats(rpy_err) if rpy_err else None,
        "c_counts": term.get("counts") if backend == "decoupled" else None,
        "gate_5cm_p95": "PASS" if np.percentile(palm_orig, 95) <= 0.05 else "FAIL",
    }  # fmt: skip
    if orig.shape[1] >= 28 and "hands" in s.files:  # logged hand target vs the dataset's hands, both backends
        out["policy_hands_vs_dataset_abs_rad"] = stats(np.abs(s["hands"][ep_idx] - orig[frames, 14:28]))
    if backend == "sonic":
        stored = v3_episode(conv, row, info, ["frame_index", "action"])["action"].astype(np.float64)
        if stored.shape[1] == 78:
            out["policy_token_vs_stored_abs"] = stats(np.abs(s["token"][ep_idx] - stored[frames, :64]))
            out["policy_hands_vs_stored_abs_rad"] = stats(np.abs(s["hands"][ep_idx] - stored[frames, 64:]))
    return out


def main():
    run, src, conv, ep = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
    out = score(run, src, conv, ep)
    (run / "stream_eval.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["valid"] else 2)


if __name__ == "__main__":
    main()
