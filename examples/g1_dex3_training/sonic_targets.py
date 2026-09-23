"""Episode-local G1 joint references for the official SONIC v1.1 encoder.

The input has no measured legs, waist, or root pose. These targets therefore
describe stationary manipulation with the explicitly recorded nominal stance.
Native G1 mode uses [mode_id,0,0,0], ten 29D poses, ten velocities, and ten
row-major 6D heading rotations. All other observation terms are masked to zero.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET  # nosec B405 - parses the local, hash-recorded robot XML only
from pathlib import Path

import numpy as np

ARM_NAMES = tuple(
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
)
ACTION_NAMES = (
    ARM_NAMES
    + tuple(
        f"left_hand_{joint}_joint"
        for joint in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
    )
    + tuple(
        f"right_hand_{joint}_joint"
        for joint in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
    )
)
ISAAC_FROM_MOTOR = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
)
NOMINAL_BODY = np.array(
    [
        -0.312,
        0,
        0,
        0.669,
        -0.363,
        0,
        -0.312,
        0,
        0,
        0.669,
        -0.363,
        0,
        0,
        0,
        0,
        0.2,
        0.2,
        0,
        0.6,
        0,
        0,
        0,
        0.2,
        -0.2,
        0,
        0.6,
        0,
        0,
        0,
    ],
    dtype=np.float64,
)
OBSERVATION_TERMS = (
    ("encoder_mode_4", 4),
    ("motion_joint_positions_10frame_step5", 290),
    ("motion_joint_velocities_10frame_step5", 290),
    ("motion_anchor_orientation_heading_10frame_step5", 60),
    ("motion_anchor_orientation_heading", 6),
    ("motion_joint_positions_lowerbody_10frame_step5", 120),
    ("motion_joint_velocities_lowerbody_10frame_step5", 120),
    ("vr_3point_local_target", 9),
    ("vr_3point_local_orn_target", 12),
    ("smpl_joints_10frame_step1", 720),
    ("smpl_anchor_orientation_heading_10frame_step1", 60),
    ("motion_joint_positions_wrists_10frame_step1", 60),
)


def load_joint_limits(xml_path: Path) -> np.ndarray:
    root = ET.parse(xml_path).getroot()  # nosec B314 - trusted local robot model file
    compiler = root.find("compiler")
    if compiler is None or compiler.get("angle") != "radian":
        raise ValueError("robot XML must explicitly declare radians")
    joints = {joint.get("name"): joint for joint in root.iter("joint") if joint.get("name")}
    return np.asarray([list(map(float, joints[name].attrib["range"].split())) for name in ACTION_NAMES])


def build_encoder_inputs(
    actions: np.ndarray,
    limits: np.ndarray,
    *,
    arm_speed_limit: float | None = 1.0,
    hand_speed_limit: float | None = 2.0,
    fps: float = 30,
):
    """Return source-aligned inputs, filtered hands, and an auditable conversion report.

    By default apply the reference adapter's 1rad/s arm and 2rad/s hand slew limits at 50Hz;
    None disables a limit (joint-limit clipping always applies).
    Training adds no entry/settling interval and never borrows the next episode.
    Source rows (fps, default 30Hz) select nearest 50Hz reference frames (<=6.67ms error at 30Hz).
    """
    actions = np.asarray(actions, dtype=np.float64)
    limits = np.asarray(limits, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 28 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("expected nonempty finite actions [N,28]")
    if limits.shape != (28, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("expected finite increasing limits [28,2]")
    clipped = np.clip(actions, limits[:, 0], limits[:, 1])
    source_times = np.arange(len(actions), dtype=np.float64) / fps
    dense_times = np.arange(int(np.ceil(source_times[-1] * 50)) + 1, dtype=np.float64) / 50
    desired = np.column_stack([np.interp(dense_times, source_times, clipped[:, i]) for i in range(28)])
    filtered = np.empty_like(desired)
    filtered[0] = desired[0]
    max_step = (
        np.r_[
            np.full(14, np.inf if arm_speed_limit is None else arm_speed_limit),
            np.full(14, np.inf if hand_speed_limit is None else hand_speed_limit),
        ]
        / 50
    )
    for i in range(1, len(filtered)):
        filtered[i] = filtered[i - 1] + np.clip(desired[i] - filtered[i - 1], -max_step, max_step)
    body = np.tile(NOMINAL_BODY, (len(filtered), 1))
    body[:, 15:] = filtered[:, :14]
    body = body[:, ISAAC_FROM_MOTOR]
    velocities = np.gradient(body, 0.02, axis=0) if len(body) > 1 else np.zeros_like(body)
    # Terminal reference is explicitly held, including zero reference velocity.
    velocities[-1] = 0
    body = np.vstack([body, np.repeat(body[-1:], 46, axis=0)])
    velocities = np.vstack([velocities, np.zeros((46, 29))])
    reference_rows = np.rint(source_times * 50).astype(np.int64)
    future_rows = reference_rows[:, None] + np.arange(10)[None] * 5
    inputs = np.zeros((len(actions), 1751), dtype=np.float32)
    inputs[:, 4:294] = body[future_rows].reshape(len(actions), 290)
    inputs[:, 294:584] = velocities[future_rows].reshape(len(actions), 290)
    inputs[:, 584:644] = np.tile([1, 0, 0, 1, 0, 0], 10)
    hands = filtered[reference_rows, 14:].astype(np.float32)
    report = {
        "encoder_mode": 0,
        "source_fps": fps,
        "reference_fps": 50,
        "preview_frames": 10,
        "preview_step": 5,
        "lower_body_assumption": "fixed_nominal_standing_legs_and_waist",
        "root_orientation_wxyz": [1, 0, 0, 0],
        "nominal_body_motor_order": NOMINAL_BODY.tolist(),
        "arm_speed_limit_rad_s": arm_speed_limit,
        "hand_speed_limit_rad_s": hand_speed_limit,
        "clipped_values": int(np.count_nonzero(clipped != actions)),
        "rate_limited_values": int(np.count_nonzero(np.abs(filtered - desired) > 1e-9)),
        "max_reference_time_error_s": float(np.max(np.abs(reference_rows / 50 - source_times))),
    }
    return inputs, hands, report


def combine_tokens_and_hands(tokens: np.ndarray, hands: np.ndarray) -> np.ndarray:
    tokens, hands = np.asarray(tokens, dtype=np.float32), np.asarray(hands, dtype=np.float32)
    if tokens.ndim != 2 or tokens.shape[1] != 64 or hands.shape != (len(tokens), 14):
        raise ValueError("expected matching token [N,64] and hand [N,14] arrays")
    if not np.isfinite(tokens).all() or not np.isfinite(hands).all():
        raise ValueError("nonfinite SONIC targets")
    if not np.allclose(tokens * 16, np.rint(tokens * 16), atol=1e-5, rtol=0):
        raise ValueError("SONIC tokens must lie on the official 1/16 grid")
    return np.concatenate((tokens, hands), axis=1)


class SonicEncoder:
    """CPU-only official ONNX encoder, with model/config provenance."""

    def __init__(self, model: Path, observation_config: Path):
        import onnxruntime as ort
        import yaml

        encoder = yaml.safe_load(observation_config.read_text())["encoder"]
        active_names = [term["name"] for term in encoder["encoder_observations"] if term.get("enabled", True)]
        if active_names != [name for name, _ in OBSERVATION_TERMS] or encoder["dimension"] != 64:
            raise ValueError("unexpected SONIC observation configuration")
        mode = next(item for item in encoder["encoder_modes"] if item["mode_id"] == 0)
        if mode["required_observations"] != active_names[:4]:
            raise ValueError("unexpected SONIC G1 mode terms")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if [(v.name, v.shape, v.type) for v in inputs] != [("obs_dict", [1, 1751], "tensor(float)")]:
            raise ValueError("unexpected SONIC encoder input contract")
        if [(v.name, v.shape, v.type) for v in outputs] != [("encoded_tokens", [1, 64], "tensor(float)")]:
            raise ValueError("unexpected SONIC encoder output contract")
        self.provenance = {
            "encoder_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "observation_config_sha256": hashlib.sha256(observation_config.read_bytes()).hexdigest(),
            "onnxruntime_version": ort.__version__,
            "provider": "CPUExecutionProvider",
        }

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 1751 or not np.isfinite(values).all():
            raise ValueError("expected finite encoder inputs [N,1751]")
        return np.concatenate(
            [self.session.run(["encoded_tokens"], {"obs_dict": row[None]})[0] for row in values]
        )
