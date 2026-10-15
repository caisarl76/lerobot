"""isaac_scene.py pure helpers (no Isaac import): camera config and the PD torque law of gear_sonic base_sim."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
import isaac_scene as isc


def test_pd_torques_matches_base_sim_and_clips():
    cmd = {"q": np.array([1.0, 0.0]), "dq": np.zeros(2), "kp": np.array([100.0, 10.0]),
           "kd": np.array([2.0, 1.0]), "tau": np.array([0.5, 0.0])}  # fmt: skip
    q, dq, limit = np.array([0.9, 0.0]), np.array([0.0, 1.0]), np.array([50.0, 50.0])
    np.testing.assert_allclose(isc.pd_torques(cmd, q, dq, limit), [0.5 + 100 * 0.1, -1.0])
    cmd["kp"][0] = 1000.0  # 100 Nm unclipped
    assert isc.pd_torques(cmd, q, dq, limit)[0] == 50.0


def test_load_camera_requires_all_keys(tmp_path):
    cam = isc.load_camera(isc.CAMERA_JSON)
    assert (cam["width"], cam["height"]) == (640, 480)
    assert cam["parent"] == "torso_link" and len(cam["pos"]) == 3 and len(cam["rot_wxyz"]) == 4
    bad = tmp_path / "c.json"
    bad.write_text(json.dumps({k: v for k, v in cam.items() if k != "fx"}))
    with pytest.raises(ValueError, match="fx"):
        isc.load_camera(bad)


def test_effort_limits_reorders_mujoco_actuators_to_body_then_dex3():
    # gear_sonic motor_effort_limit_list is in MuJoCo actuator order: legs, waist, left arm, left hand, right arm,
    # right hand. Give every actuator a distinct value and check each joint gets its own.
    values = np.arange(len(isc.MUJOCO_ACTUATORS), dtype=float)
    out = isc.effort_limits(values)
    order = isc.BODY + isc.DEX3_MOTORS
    assert out.shape == (len(order),)
    for name, v in zip(order, out):
        assert isc.MUJOCO_ACTUATORS[int(v)] == name
