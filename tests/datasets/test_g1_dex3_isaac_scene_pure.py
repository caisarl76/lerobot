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


def test_imu_velocities_follow_mujoco_frames():
    # gear_sonic MuJoCo: base = [qvel linear (world), qvel angular (body frame)]; torso IMU = mj_objectVelocity local
    # ([lin, ang] in the torso frame). Isaac gives everything in the world frame.
    yaw90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])  # w x y z: +90 deg about z
    lin_w, ang_w = np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])  # world +x
    base, torso = isc.imu_velocities(yaw90, lin_w, ang_w, yaw90, lin_w, ang_w)
    np.testing.assert_allclose(base[:3], [1.0, 0.0, 0.0], atol=1e-9)  # linear stays in the world frame
    np.testing.assert_allclose(base[3:], [0.0, -1.0, 0.0], atol=1e-9)  # world +x seen from a body yawed +90 deg
    np.testing.assert_allclose(torso, [0.0, -1.0, 0.0, 0.0, -1.0, 0.0], atol=1e-9)  # both torso parts local


def test_composite_keeps_robot_and_bottle_pixels_only():
    rgb, bg = np.full((2, 3, 3), 200, np.uint8), np.full((2, 3, 3), 10, np.uint8)
    seg = np.array([[1, 2, 3], [4, 0, 1]])
    labels = {"1": "/World/envs/env_0/Robot/right_hand_index_0_link/visuals", "2": "/World/envs/env_0/Bottle/neck/geometry/mesh",
              "3": "/World/envs/env_0/Table/geometry/mesh", "4": "/World/ground/Environment", "0": "BACKGROUND"}  # fmt: skip
    out = isc.composite(rgb, seg, labels, bg)
    np.testing.assert_array_equal(out[..., 0], [[200, 200, 10], [10, 10, 200]])  # robot, bottle | case, floor, sky
    out = isc.composite(rgb, seg, labels, bg, keep=("/Robot/",))  # the real bottle in bg stays visible
    np.testing.assert_array_equal(out[..., 0], [[200, 10, 10], [10, 10, 200]])


def test_parse_friction():
    assert isc.parse_friction(None) is None and isc.parse_friction("") is None  # unset: PhysX default
    assert isc.parse_friction("0.3") == (0.3, 0.3)
    assert isc.parse_friction("0.4,0.3") == (0.4, 0.3)
    with pytest.raises(ValueError):
        isc.parse_friction("0.3,0.4")  # dynamic above static


def test_bottle_xy_per_axis_jitter_is_seeded_and_bounded():
    xy = np.array([isc.sample_bottle_xy(s, [0.3675, -0.005], [0.0525, 0.115]) for s in range(200)])
    assert (np.abs(xy[:, 0] - 0.3675) <= 0.0525).all() and (np.abs(xy[:, 1] + 0.005) <= 0.115).all()
    assert np.ptp(xy[:, 1]) > 0.15  # y spreads wider than x could
    np.testing.assert_array_equal(isc.sample_bottle_xy(7, [0.3, 0.0], 0.04), isc.sample_bottle_xy(7, [0.3, 0.0], 0.04))  # paired


def test_case_offset_turns_about_the_pivot_then_shifts():
    pivot = np.array([0.01, 0.0])
    np.testing.assert_allclose(isc.case_offset([0.4, -0.05], pivot, [0.0, 0.0, 0.0]), [0.4, -0.05])
    np.testing.assert_allclose(isc.case_offset([0.41, 0.0], pivot, [0.0, 0.0, 90.0]), [0.01, 0.4], atol=1e-12)
    np.testing.assert_allclose(isc.case_offset([0.41, 0.0], pivot, [-0.01, 0.02, 90.0]), [0.0, 0.42], atol=1e-12)


def test_plane_homography_follows_camera_motion():
    cam = {"fx": 615.0, "fy": 615.0, "cx": 320.0, "cy": 240.0}
    a = np.radians(48)  # pitched down, x forward / z up (Isaac "world" camera convention)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    pos = np.array([0.08, 0.0, 1.26])
    np.testing.assert_allclose(isc.plane_homography(cam, pos, R, pos, R, 0.845), np.eye(3), atol=1e-9)
    # a camera moved 2 cm left sees a point on the plane 2 cm further right than the reference camera
    M = isc.plane_homography(cam, pos, R, pos + [0.0, 0.02, 0.0], R, 0.845)
    p = np.array([0.4, -0.05, 0.845])
    def px(c, Rc):
        q = Rc.T @ (p - c)
        return np.array([cam["cx"] - cam["fx"] * q[1] / q[0], cam["cy"] - cam["fy"] * q[2] / q[0], 1.0])
    ref, cur = px(pos, R), px(pos + [0.0, 0.02, 0.0], R)
    w = M @ ref
    np.testing.assert_allclose(w[:2] / w[2], cur[:2], atol=1e-6)


def test_follow_background_fills_uncovered_pixels_from_the_render():
    bg = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    fill = np.full_like(bg, 7)
    m = np.array([[1.0, 0, 10], [0, 1, 0], [0, 0, 1]])  # the real frame moved 10 px right: columns 0-9 have no data
    out = isc.follow_background(bg, m, fill)
    np.testing.assert_array_equal(out[:, :10], 7)  # the render, not the replicated edge column
    np.testing.assert_array_equal(out[:, 10:], bg[:, :54])


def test_look_at_points_the_camera_x_axis_at_the_target():
    pos, target = np.array([1.3, -0.9, 1.5]), np.array([0.4, -0.05, 0.9])
    R = isc.quat_wxyz_to_matrix(isc.look_at_wxyz(pos, target))
    np.testing.assert_allclose(R[:, 0], (target - pos) / np.linalg.norm(target - pos), atol=1e-9)  # x forward
    assert abs(R[2, 1]) < 1e-9 and R[2, 2] > 0  # no roll: y stays horizontal, z up


def test_robot_look_follows_the_real_g1_finish():
    assert isc.robot_look("right_hand_index_1_link") == isc.robot_look("left_wrist_yaw_link") == "dex3_black"
    assert isc.robot_look("pelvis") == isc.robot_look("left_hip_pitch_link") == "g1_dark"
    assert isc.robot_look("right_elbow_link") == isc.robot_look("torso_link") == "g1_metal"


def test_head_camera_mount_matches_the_official_urdf():
    cam = isc.load_camera(isc.CAMERA_JSON)  # Unitree g1_29dof_with_hand_rev_1_0.urdf d435_joint on torso_link
    assert isc.URDF.name == "g1_29dof_with_hand_rev_1_0.urdf"
    np.testing.assert_allclose(cam["pos"], [0.0576235, 0.01753, 0.42987])
