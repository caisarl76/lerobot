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
    for name, v in zip(order, out, strict=True):
        assert isc.MUJOCO_ACTUATORS[int(v)] == name


def test_quat_wxyz_to_matrix():
    np.testing.assert_allclose(isc.quat_wxyz_to_matrix([1, 0, 0, 0]), np.eye(3), atol=1e-12)
    r = isc.quat_wxyz_to_matrix([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])  # +90 deg about z
    np.testing.assert_allclose(r @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_make_obs_has_every_bridge_key_and_splits_hands():
    nb, nh = len(isc.BODY), len(isc.DEX3_MOTORS)
    state = {"q": np.arange(nb + nh, dtype=float), "dq": np.zeros(nb + nh), "ddq": np.zeros(nb + nh),
             "tau": np.zeros(nb + nh), "root_pose": np.r_[0, 0, 0.79, 1, 0, 0, 0], "root_vel": np.zeros(6),
             "root_acc": np.zeros(6), "torso_quat": np.r_[1, 0, 0, 0], "torso_vel": np.zeros(6), "time": 1.5}  # fmt: skip
    obs = isc.make_obs(state)
    for k in ("body_q", "body_dq", "body_ddq", "body_tau_est", "floating_base_pose", "floating_base_vel",
              "floating_base_acc", "secondary_imu_quat", "secondary_imu_vel", "left_hand_q", "left_hand_dq",
              "right_hand_q", "right_hand_dq", "time"):  # fmt: skip
        assert k in obs, k
    assert obs["body_q"].shape == (nb,) and obs["left_hand_q"][0] == nb and obs["right_hand_q"][0] == nb + 7
