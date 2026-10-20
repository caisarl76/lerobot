"""Test 4b, open-loop visual gap of the Isaac scene: does the policy act on an Isaac head render as on the real frame?

  samples SESSIONS OUT_DIR   pre-grasp samples of real sessions (every 0.5 s from the episode start to the right
                             hand's grasp): logged arm+hand state, frame index, the trial's bottle position (grasp
                             point in front of the palm, MuJoCo FK, torso pitched 6 deg as in the planner stance)
  (then isaac_render_states.py OUT_DIR in the Isaac env renders each sample)
  policy OUT_DIR MODEL       the policy's chunk (first 0.5 s) on the real frame, the render and a black frame, same
                             state (bf16 backbone for GR00T, noise 0, as the robot sessions); policy_gap.json
Gap = mean |action(render) - action(real)|; the black frame is the no-vision control.
Usage (lerobot env): G1_SESSIONS=SESSIONS_ROOT python isaac_visual_gap.py samples s07_wbt_groot_he_handover OUT_DIR
"""

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

SESSIONS = Path(os.environ.get("G1_SESSIONS", "sessions"))  # root of the robot eval session dirs
GEAR = Path.home() / "work" / "GR00T-WholeBodyControl"
TORSO_FROM_PELVIS, STANCE_PITCH = np.array([-0.004, 0.0, 0.054]), np.radians(6.0)
BOTTLE_R, PALM_HALF_THICKNESS = 0.033, 0.015


def samples(sessions: str, out: Path) -> None:
    import mujoco
    from sonic_targets import ACTION_NAMES

    m = mujoco.MjModel.from_xml_path(str(GEAR / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"))
    d = mujoco.MjData(m)
    c, s = np.cos(STANCE_PITCH), np.sin(STANCE_PITCH)
    pitch = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    rows = []
    for sess in sessions.split(","):
        for f in sorted(glob.glob(str(SESSIONS / sess / "raw" / "trial_*[0-9].npz"))):
            z = np.load(f, allow_pickle=True)
            frames = np.load(f.replace(".npz", "_frames.npz"), allow_pickle=True)
            ep = np.flatnonzero(z["phase"] == "episode")
            h = z["state"][ep, 14:]
            closed = ((h - h.min(0)) / np.maximum(h.max(0) - h.min(0), 1e-6))[:, 7:].mean(1) > 0.5
            rises = [i for i in np.flatnonzero(closed[1:] & ~closed[:-1]) + 1 if closed[i : i + 25].all()]
            if not rises:
                continue
            k = ep[rises[0]]  # the right hand's grasp, held >= 0.5 s
            mujoco.mj_resetData(m, d)
            for n, v in zip(ACTION_NAMES[:28], z["state"][k], strict=True):
                d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
            mujoco.mj_forward(m, d)
            tid = m.body("torso_link").id
            rt, pt = d.xmat[tid].reshape(3, 3), d.xpos[tid]
            palm = np.mean(
                [rt.T @ (d.xpos[m.body(f"right_hand_{q}_0_link").id] - pt) for q in ("index", "middle")], 0
            )
            wrist = rt.T @ d.xmat[m.body("right_wrist_yaw_link").id].reshape(3, 3)
            bottle = TORSO_FROM_PELVIS + pitch @ (palm + (BOTTLE_R + PALM_HALF_THICKNESS) * wrist[:, 1])
            for tw in np.arange(z["wall"][ep[0]], z["wall"][k], 0.5):
                fi = int(np.argmin(abs(frames["t_recv"] - tw)))
                row = int(np.argmin(abs(z["wall"] - frames["t_recv"][fi])))
                rows.append({"session": sess, "trial": int(Path(f).stem.split("_")[1]), "frame": fi, "row": row,
                             "t_ep": float(z["t_ep"][row]), "verdict": str(z["verdict"]),
                             "bottle_xy": bottle[:2].tolist(), "state": z["state"][row].tolist()})  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples.json").write_text(json.dumps(rows))
    print(len(rows), "samples")


def policy(out: Path, model: str, horizon: int = 25) -> None:
    import cv2
    from sonic_policy_streamer import ChunkPolicy

    rows, renders = json.loads((out / "samples.json").read_text()), np.load(out / "renders.npz")["rgb"]
    pol = ChunkPolicy(model, "cuda", backbone_dtype="bfloat16" if "groot" in model else None, noise_scale=0.0)
    key = pol.image_keys[0]
    _, h, w = pol.shapes[key]

    def act(img, state, task):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return pol.chunk([state] * pol.n_obs, [{key: img}] * pol.n_obs, task)[:horizon]

    res = []
    for i, s in enumerate(rows):
        raw = SESSIONS / s["session"] / "raw"
        frames = np.load(raw / f"trial_{s['trial']:03d}_frames.npz", allow_pickle=True)
        z = np.load(raw / f"trial_{s['trial']:03d}.npz", allow_pickle=True)
        o = frames["offsets"]
        real = cv2.imdecode(
            frames["jpeg"][o[s["frame"]] : o[s["frame"] + 1]], cv2.IMREAD_COLOR
        )  # RGB bytes: as fed
        task, state = json.loads(str(z["args"]))["task"], np.asarray(s["state"], np.float32)
        a_real, a_sim, a_black = (
            act(real, state, task),
            act(renders[i], state, task),
            act(0 * real, state, task),
        )
        gap = lambda a, sl, ref=a_real: float(np.abs(a[:, sl] - ref[:, sl]).mean())  # noqa: E731
        res.append({"trial": s["trial"], "t_ep": s["t_ep"], "verdict": s["verdict"],
                    "token_sim": gap(a_sim, slice(0, 64)), "token_black": gap(a_black, slice(0, 64)),
                    "hand_sim": gap(a_sim, slice(64, 78)), "hand_black": gap(a_black, slice(64, 78)),
                    "token_real_vs_streamed": float(np.abs(a_real[0, :64] - z["token"][s["row"]]).mean())})  # fmt: skip
    (out / "policy_gap.json").write_text(json.dumps(res, indent=1))
    med = {
        k: round(float(np.median([r[k] for r in res])), 4)
        for k in res[0]
        if k not in ("trial", "t_ep", "verdict")
    }
    med["render_closer_than_black"] = f"{sum(r['token_sim'] < r['token_black'] for r in res)} of {len(res)}"
    print(json.dumps(med))


if __name__ == "__main__":
    if sys.argv[1] == "samples":
        samples(sys.argv[2], Path(sys.argv[3]))
    elif sys.argv[1] == "policy":
        policy(Path(sys.argv[2]), sys.argv[3])
    else:
        raise SystemExit(__doc__)
