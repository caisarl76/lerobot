"""Spec test 0a: unpaced real-time factor of the Isaac scene with the head camera rendering at 30 Hz.

The robot is held on the band at standing height (root pinned, feet on the floor), joints PD-held at gear_sonic's
DEFAULT_DOF_ANGLES with its MOTOR_KP / MOTOR_KD (hands 1.5 / 0.1 at 0), the table placed: the per-step work of a
real run without the deploy. Pass: sim time / wall time >= 1.2 in every 1 s window after a 2 s warm-up.
Usage (sonic env): CUDA_VISIBLE_DEVICES=0 python isaac_rtf_probe.py OUT_DIR [--seconds 60] [--stand-z 0.79]
"""

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

import isaac_scene as isc

p = argparse.ArgumentParser()
p.add_argument("out", type=Path)
p.add_argument("--seconds", type=float, default=60.0)
p.add_argument("--stand-z", type=float, default=isc.STAND_Z)
p.add_argument("--device", default="cuda:0", help="physics device: cuda:0 or cpu")
p.add_argument("--no-render", action="store_true", help="diagnostic: camera never rendered")
a = p.parse_args()

try:  # Kit keeps the process alive after an exception or app.close(): always exit hard
    scene = isc.Scene(a.out, "5", device=a.device, render=not a.no_render)
    cfg, nb = scene.config, len(isc.BODY)
    scene.cmd["q"][:nb] = cfg["DEFAULT_DOF_ANGLES"]
    scene.cmd["kp"][:nb], scene.cmd["kd"][:nb] = cfg["MOTOR_KP"], cfg["MOTOR_KD"]
    scene.cmd["kp"][nb:], scene.cmd["kd"][nb:] = 1.5, 0.1
    scene.pinned_root = np.array([0.0, 0.0, a.stand_z, 1.0, 0.0, 0.0, 0.0])
    scene.place_table()

    windows, wall0, t_win, w_win = [], time.monotonic(), 0.0, time.monotonic()
    while scene.t < a.seconds:
        scene.step()
        if scene.t - t_win >= 1.0 - 1e-9:
            now = time.monotonic()
            windows.append({"t": round(scene.t, 3), "rtf": round((scene.t - t_win) / (now - w_win), 3)})
            t_win, w_win = scene.t, now
    steady = [w["rtf"] for w in windows if w["t"] > 2.0]
    feet = int((scene.feet.data.net_forces_w[0].norm(dim=-1) > 1.0).sum())
    result = {
        "pass": bool(steady) and min(steady) >= 1.2,
        "rtf_min": min(steady), "rtf_median": float(np.median(steady)),
        "overall_rtf": round(scene.t / (time.monotonic() - wall0), 3),
        "frame_ms_p95": float(np.percentile(scene.frame_ms, 95)) if scene.frame_ms else None, "frames": len(scene.frame_ms),
        "feet_in_contact": feet, "stand_z": a.stand_z, "device": a.device, "render": not a.no_render, "windows": windows,
    }  # fmt: skip
    (a.out / "rtf.json").write_text(json.dumps(result, indent=1))
    import cv2  # noqa: E402

    if scene.frame_ms:
        cv2.imwrite(str(a.out / "head_cam_last.png"), cv2.cvtColor(scene.last_rgb, cv2.COLOR_RGB2BGR))
    print("RTF " + json.dumps({k: v for k, v in result.items() if k != "windows"}), flush=True)
    scene.close("completed")
except Exception:
    traceback.print_exc()
    os._exit(1)
os._exit(0)
