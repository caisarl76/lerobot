"""Test 4b: render the Isaac head view at each sample of OUT_DIR/samples.json (isaac_visual_gap.py samples): the
planner-stance pelvis and waist pinned, the sample's logged arm+hand joints, the task's case, the bottle at the
sample's position. Writes OUT_DIR/renders.npz (RGB uint8, the policy's input convention).
Usage (sonic env, GPU 0): python isaac_render_states.py OUT_DIR
"""

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
try:  # Kit keeps the process alive after an exception: always exit hard
    import isaac_scene as isc
    import torch
    from sonic_targets import ACTION_NAMES

    task = json.loads((Path(__file__).parent / "tasks" / "handover_bottle.json").read_text())
    rows = json.loads((out / "samples.json").read_text())
    scene = isc.Scene(out / "isaac", str(task["table_gap_cm"]), dds=False)
    a = np.radians(-2.5)  # pelvis pitch and waist pitch of the deploy's planner stance (torso +6 deg)
    scene.pinned_root = np.array([0.0, 0.0, 0.788, np.cos(a / 2), 0.0, np.sin(a / 2), 0.0])
    names = isc.BODY + isc.DEX3_MOTORS
    base = np.zeros(len(names))
    base[: len(isc.BODY)] = scene.config["DEFAULT_DOF_ANGLES"]
    base[names.index("waist_pitch_joint")] = 0.147
    for _ in range(10):  # the kinematic case is placed only after the first steps
        scene.step()
    scene.place_table()
    imgs = []
    for s in rows:
        q = base.copy()
        for n, v in zip(ACTION_NAMES[:28], s["state"], strict=True):
            q[names.index(n)] = v
        pos = scene.robot.data.default_joint_pos.clone()
        pos[0, scene.body_ids + scene.hand_ids] = torch.tensor(q, dtype=pos.dtype, device=pos.device)
        scene.place_bottle(0, s["bottle_xy"], 0.0)
        for _ in range(30):  # hold the pose; the renderer converges
            scene.robot.write_joint_state_to_sim(pos, torch.zeros_like(pos))
            scene.next_frame = 0.0
            scene.step()
        imgs.append(scene.last_rgb.copy())
    np.savez_compressed(out / "renders.npz", rgb=np.stack(imgs))
    print("rendered", len(imgs), flush=True)  # os._exit skips the flush
except Exception:
    traceback.print_exc()
os._exit(0)
