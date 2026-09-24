"""SONIC round-trip audit: joint actions -> encoder -> token -> decoder (MuJoCo closed loop) -> FK hand.

Read-only on every dataset/model. Decoder layout, permutations, gains and action scale follow
gear_sonic_deploy g1_deploy_onnx_ref.cpp + policy_parameters.hpp (SONIC v1.1). Plant: official
sim2sim scene_43dof.xml at 200 Hz, PD torques computed here, decoder at 50 Hz. Dex3 fingers are
unactuated in sim (they bypass SONIC); the palm point used for FK does not depend on finger joints.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq

sys.path.insert(0, "/code")
from sonic_targets import (
    ISAAC_FROM_MOTOR,
    NOMINAL_BODY,
    SonicEncoder,
    build_encoder_inputs,
    load_joint_limits,
)  # noqa: E402

MODEL = Path("/sonic-model")
SCENE = "/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
CONVERTER_XML = Path("/gear_sonic_deploy/g1/g1_29dof_with_hand.xml")
MOTOR_FROM_ISAAC = np.argsort(ISAAC_FROM_MOTOR)
BODY_NAMES = (
    [
        f"{s}_{j}_joint"
        for s in ("left", "right")
        for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ]
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + [
        f"{s}_{j}_joint"
        for s in ("left", "right")
        for j in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    ]
)
# policy_parameters.hpp
ARM = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}
EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
W = 10 * 2.0 * 3.1415926535
_LEG = ["7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020"]
_ARM = ["5020"] * 5 + ["4010"] * 2
MOTOR_TYPE = _LEG + _LEG + ["7520_14", "5020", "5020"] + _ARM + _ARM
DOUBLE = {4, 5, 10, 11, 13, 14}  # ankles, waist roll/pitch: 2x kp and kd
KP = np.array([ARM[t] * W * W * (2 if i in DOUBLE else 1) for i, t in enumerate(MOTOR_TYPE)])
KD = np.array([2 * 2.0 * ARM[t] * W * (2 if i in DOUBLE else 1) for i, t in enumerate(MOTOR_TYPE)])
SCALE = np.array([0.25 * EFFORT[t] / (ARM[t] * W * W) for t in MOTOR_TYPE])
DEFAULT = NOMINAL_BODY.copy()  # identical to hpp default_angles
DEFAULT_ISAAC = DEFAULT[ISAAC_FROM_MOTOR]
HIST, TOK = 10, 64
WARMUP_TICKS = 50


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Robot:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(SCENE)
        self.m.opt.timestep = 0.005
        self.d = mujoco.MjData(self.m)
        self.kin = mujoco.MjData(self.m)
        jid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in BODY_NAMES]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in jid])
        self.vadr = np.array([self.m.jnt_dofadr[j] for j in jid])
        act_joint = {int(self.m.actuator_trnid[a, 0]): a for a in range(self.m.nu)}
        self.act = np.array([act_joint[j] for j in jid])
        self.ctrl_lo, self.ctrl_hi = self.m.actuator_ctrlrange[self.act].T
        self.palm = {
            s: [
                mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_hand_{f}_0_link")
                for f in ("index", "middle")
            ]
            for s in ("left", "right")
        }
        self.pelvis = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.floor = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.limits = self.m.jnt_range[jid]
        foot_bodies = {
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link")
            for s in ("left", "right")
        }
        # collision geoms only (the foot meshes are visual: contype=conaffinity=0)
        self.foot_geoms = [
            g
            for g in range(self.m.ngeom)
            if self.m.geom_bodyid[g] in foot_bodies and (self.m.geom_contype[g] or self.m.geom_conaffinity[g])
        ]
        assert all(self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX for g in self.foot_geoms), (
            "expected box soles"
        )

    def palms(self, data):
        return np.stack([data.xpos[self.palm[s]].mean(0) for s in ("left", "right")])

    def fk_pelvis_frame(self, q_motor):
        """Palm points in the pelvis frame for body joint angles (motor order)."""
        self.kin.qpos[:] = 0
        self.kin.qpos[3] = 1
        self.kin.qpos[self.qadr] = q_motor
        mujoco.mj_kinematics(self.m, self.kin)
        return self.palms(self.kin)

    def reset(self, q_motor):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[3] = 1
        self.d.qpos[self.qadr] = q_motor
        mujoco.mj_forward(self.m, self.d)
        self.d.qpos[2] -= self.sole_min_z()
        mujoco.mj_forward(self.m, self.d)

    def sole_min_z(self):
        """Lowest corner of the box soles, world z."""
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        return min(
            float(
                (
                    self.d.geom_xpos[g][:, None]
                    + self.d.geom_xmat[g].reshape(3, 3) @ (corners * self.m.geom_size[g]).T
                )[2].min()
            )
            for g in self.foot_geoms
        )

    def floor_contacts(self):
        return sum(
            1 for i in range(self.d.ncon) if self.floor in (self.d.contact[i].geom1, self.d.contact[i].geom2)
        )

    def state(self):
        q = self.d.qpos[self.qadr].copy()
        dq = self.d.qvel[self.vadr].copy()
        quat = self.d.qpos[3:7].copy()
        gyro = self.d.qvel[3:6].copy()  # free-joint angular velocity is body-local
        return q, dq, quat, gyro

    def band_wrench(self):
        """gear_sonic ElasticBand (unitree_mujoco): spring-damper holding the pelvis at (0,0,1), upright."""
        pos, rot = self.d.xpos[self.pelvis], self.d.xmat[self.pelvis].reshape(3, 3)
        lin, ang = self.d.qvel[0:3], rot @ self.d.qvel[3:6]
        f = 10000 * (np.array([0, 0, 1.0]) - pos) - 1000 * lin
        w, v = self.d.qpos[3], self.d.qpos[4:7]
        angle = 2 * np.arctan2(np.linalg.norm(v), w)
        rotvec = v / max(np.linalg.norm(v), 1e-12) * angle
        return np.r_[f, -1000 * rotvec - 10 * ang]

    def step(self, target_motor, n=4, band=False):
        """Returns the per-joint fraction of substeps whose PD torque hit the actuator limit."""
        sat = np.zeros(29)
        for _ in range(n):
            self.d.xfrc_applied[self.pelvis] = self.band_wrench() if band else 0
            q, dq = self.d.qpos[self.qadr], self.d.qvel[self.vadr]
            tau = KP * (target_motor - q) - KD * dq
            sat += (tau <= self.ctrl_lo) | (tau >= self.ctrl_hi)
            self.d.ctrl[self.act] = np.clip(tau, self.ctrl_lo, self.ctrl_hi)
            mujoco.mj_step(self.m, self.d)
        return sat / n


def gravity(quat):
    w, x, y, z = quat / np.linalg.norm(quat)
    # rotate (0,0,-1) by conj(quat)
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return r.T @ np.array([0.0, 0.0, -1.0])


class Decoder:
    def __init__(self):
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = 1
        self.s = ort.InferenceSession(
            str(MODEL / "model_decoder.onnx"), so, providers=["CPUExecutionProvider"]
        )
        self.name = self.s.get_inputs()[0].name
        assert self.s.get_inputs()[0].shape[-1] == 994 and self.s.get_outputs()[0].shape[-1] == 29

    def reset(self):
        z = np.zeros(29)
        self.h = {
            "ang": [np.zeros(3)] * HIST,
            "q": [z] * HIST,
            "dq": [z] * HIST,
            "act": [z] * HIST,
            "g": [np.array([0, 0, -1.0])] * HIST,
        }
        self.last = z.copy()

    def __call__(self, token, q_motor, dq_motor, quat, gyro):
        def push(k, v):
            return self.h.__setitem__(k, self.h[k][1:] + [v])  # oldest -> newest

        push("ang", gyro)
        push("q", q_motor[ISAAC_FROM_MOTOR] - DEFAULT_ISAAC)
        push("dq", dq_motor[ISAAC_FROM_MOTOR])
        push("act", self.last.copy())
        push("g", gravity(quat))
        obs = np.concatenate(
            [token] + self.h["ang"] + self.h["q"] + self.h["dq"] + self.h["act"] + self.h["g"]
        ).astype(np.float32)
        out = self.s.run(None, {self.name: obs[None]})[0][0].astype(np.float64)
        self.last = out
        return DEFAULT + out[MOTOR_FROM_ISAAC] * SCALE  # absolute motor-order targets


def generic_encoder_inputs(body_motor, fps, limits):
    """Encoder mode-0 inputs from an absolute 29D motor-order body trajectory (43D branch / calibration).

    Same layout/timing as sonic_targets.build_encoder_inputs; joint-limit clip + linear resample to 50 Hz,
    no slew limit, identity root orientation.
    """
    n = len(body_motor)
    clipped = np.clip(body_motor, limits[:, 0], limits[:, 1])
    ts = np.arange(n) / fps
    dense = np.arange(int(np.ceil(ts[-1] * 50)) + 1) / 50
    body = np.column_stack([np.interp(dense, ts, clipped[:, i]) for i in range(29)])[:, ISAAC_FROM_MOTOR]
    vel = np.gradient(body, 0.02, axis=0) if len(body) > 1 else np.zeros_like(body)
    vel[-1] = 0
    body = np.vstack([body, np.repeat(body[-1:], 46, 0)])
    vel = np.vstack([vel, np.zeros((46, 29))])
    rows = np.rint(ts * 50).astype(int)
    fut = rows[:, None] + np.arange(10)[None] * 5
    x = np.zeros((n, 1751), np.float32)
    x[:, 4:294] = body[fut].reshape(n, 290)
    x[:, 294:584] = vel[fut].reshape(n, 290)
    x[:, 584:644] = np.tile([1, 0, 0, 1, 0, 0], 10)
    return x, int(np.count_nonzero(clipped != body_motor))


def startup_hang(robot, dec, stand_token, leadin_tokens=None):
    """Official sim2sim start (gear_sonic base_sim + unitree_mujoco ElasticBand):
    1. sim starts at XML qpos0, pelvis held by the elastic band at (0,0,1) m;
    2. deploy INIT: 1 s PD ramp to default_angles, then SONIC runs in POSE/planner mode — approximated
       here by the encoder token of a constant default standing pose;
    3. key 9: band released (0.5 s operator delay); 4. Backspace: mj_resetData -> XML qpos0 on the
       floor, band stays off, controller keeps running with its history; 5. settle 3 s;
    6. optional ≤0.5 rad/s lead-in from the stance to the episode's first pose."""
    mujoco.mj_resetData(robot.m, robot.d)
    mujoco.mj_forward(robot.m, robot.d)
    q0 = robot.state()[0]
    for j in range(100):
        r = min(1.0, (j + 1) / 50)
        robot.step(q0 * (1 - r) + DEFAULT * r, band=True)
    dec.reset()
    for _ in range(50):
        robot.step(dec(stand_token, *robot.state()), band=True)
    for _ in range(25):  # key 9
        robot.step(dec(stand_token, *robot.state()))
    mujoco.mj_resetData(robot.m, robot.d)  # Backspace
    mujoco.mj_forward(robot.m, robot.d)
    info = {"qpos0_sole_min_z_m": robot.sole_min_z(), "qpos0_floor_contacts": robot.floor_contacts()}
    tilt, z = 0.0, []
    for _ in range(150):
        q, dq, quat, gyro = robot.state()
        robot.step(dec(stand_token, q, dq, quat, gyro))
        tilt = max(tilt, float(np.degrees(np.arccos(np.clip(-gravity(quat)[2], -1, 1)))))
        z.append(float(robot.d.xpos[robot.pelvis, 2]))
    info.update(
        settle_tilt_max_deg=tilt,
        pelvis_z_min=min(z),
        pelvis_z_settled=z[-1],
        pelvis_z_speed_end=float(abs(robot.d.qvel[2])),
        floor_contacts_settled=robot.floor_contacts(),
        fell=bool(min(z) < 0.55),
    )
    for tok in leadin_tokens if leadin_tokens is not None else []:
        robot.step(dec(tok, *robot.state()))
    return info


def roundtrip(
    robot,
    dec,
    tokens,
    fps,
    ref_motor,
    raw_motor,
    stand_token=None,
    start="placed",
    leadin_tokens=None,
    interp=False,
):
    """Closed-loop decode of source-rate tokens (held at 50 Hz); sample at source timestamps."""
    n = len(tokens)
    rows = np.rint(np.arange(n) / fps * 50).astype(int)
    last_tick = rows[-1]
    # Default: robot already in the episode's first pose, holding its first token before t=0.
    # stand_token: robot instead stands in SONIC's default pose holding a "stand" token (deploy start).
    startup = None
    if start == "placed":
        robot.reset(ref_motor[0] if stand_token is None else DEFAULT)
        dec.reset()
        for _ in range(WARMUP_TICKS):
            target = dec(tokens[0] if stand_token is None else stand_token, *robot.state())
            robot.step(target)
    else:
        startup = startup_hang(robot, dec, stand_token, leadin_tokens)
    start_pos = robot.d.xpos[robot.pelvis].copy()
    start_rot = robot.d.xmat[robot.pelvis].reshape(3, 3).copy()
    targets, achieved, palms_world, pel_pos, pel_rot = [], [], [], [], []
    fell_at = None
    arm_dq_max, sat_ticks, tilt_max, tgt_jump_max = np.zeros(14), np.zeros(29), 0.0, 0.0
    prev_target = None
    for j in range(last_tick + 1):
        k = min(int(np.floor(j * fps / 50 + 1e-9)), n - 1)
        tok = tokens[k]
        w = j * fps / 50 - k
        if interp == "delayed":  # causal: blend k-1 -> k over the period after token k arrives (+1 frame)
            tok = (1 - w) * tokens[max(k - 1, 0)] + w * tokens[k]
        elif interp:  # look-ahead: blend k -> k+1 (needs the next token, e.g. from the action chunk)
            tok = (1 - w) * tokens[k] + w * tokens[min(k + 1, n - 1)]
        q, dq, quat, gyro = robot.state()
        target = dec(tok, q, dq, quat, gyro)
        targets.append(target)
        achieved.append(q)
        palms_world.append(robot.palms(robot.d))
        pel_pos.append(robot.d.xpos[robot.pelvis].copy())
        pel_rot.append(robot.d.xmat[robot.pelvis].reshape(3, 3).copy())
        if fell_at is None and robot.d.xpos[robot.pelvis, 2] < 0.55:
            fell_at = j / 50
        if not np.isfinite(robot.d.qpos).all():
            fell_at = fell_at or j / 50
            break
        arm_dq_max = np.maximum(arm_dq_max, np.abs(dq[15:]))
        tilt_max = max(tilt_max, float(np.degrees(np.arccos(np.clip(-gravity(quat)[2], -1, 1)))))
        if prev_target is not None:
            tgt_jump_max = max(tgt_jump_max, float(np.abs(target[15:] - prev_target[15:]).max() * 50))
        prev_target = target
        sat_ticks += robot.step(target)
    m = min(len(targets), last_tick + 1)
    sel = rows[rows < m]
    k_ok = len(sel)
    targets, achieved = np.array(targets)[sel], np.array(achieved)[sel]
    palms_world, pel_pos, pel_rot = np.array(palms_world)[sel], np.array(pel_pos)[sel], np.array(pel_rot)[sel]

    def fk(qs):
        return np.stack([robot.fk_pelvis_frame(q) for q in qs])

    ref_p, raw_p, tgt_p = fk(ref_motor[:k_ok]), fk(raw_motor[:k_ok]), fk(targets)
    ach_p = np.einsum("kji,kpj->kpi", pel_rot, palms_world - pel_pos[:, None])  # pelvis frame
    ref_world = np.einsum("ij,kpj->kpi", start_rot, raw_p) + start_pos  # intended, pelvis frozen at t=0
    return {
        "k": k_ok,
        "fell_at_s": fell_at,
        "startup": startup,
        "safety": {
            "arm_joint_speed_max_rad_s": float(arm_dq_max.max()),
            "arm_joint_speed_max_per_joint": arm_dq_max.round(2).tolist(),
            "torque_saturation_frac_arms": float(sat_ticks[15:].sum() / (14 * max(1, len(targets)))),
            "torque_saturation_frac_legs_waist": float(sat_ticks[:15].sum() / (15 * max(1, len(targets)))),
            "torso_tilt_max_deg": tilt_max,
            "arm_setpoint_rate_max_rad_s": tgt_jump_max,
        },
        "targets": targets,
        "achieved": achieved,
        "ref_palm": ref_p,
        "raw_palm": raw_p,
        "tgt_palm": tgt_p,
        "ach_palm": ach_p,
        "world_palm": palms_world,
        "ref_world_palm": ref_world,
        "pelvis_pos": pel_pos,
    }


def err_stats(e):
    e = np.asarray(e).ravel()
    return {
        "mean": float(e.mean()),
        "p50": float(np.median(e)),
        "p95": float(np.percentile(e, 95)),
        "max": float(e.max()),
    }


def best_lag(a, b, max_shift=10):
    """Shift s (source frames) minimising mean |a[k+s]-b[k]|; positive s = a lags b."""
    out = {}
    for s in range(-max_shift, max_shift + 1):
        lo, hi = max(0, -s), min(len(b), len(a) - s)
        if hi - lo > 10:
            out[s] = float(np.linalg.norm(a[lo + s : hi + s] - b[lo:hi], axis=-1).mean())
    s = min(out, key=out.get)
    return s, out[s]


# ---------------------------------------------------------------- dataset adapters
def v3_episode(root, ep_row, info, columns):
    path = Path(root) / info["data_path"].format(
        chunk_index=ep_row["data/chunk_index"], file_index=ep_row["data/file_index"]
    )
    t = pq.read_table(path, columns=columns, filters=[("episode_index", "==", ep_row["episode_index"])])
    order = np.argsort(t["frame_index"].to_numpy())
    return {c: np.asarray(t[c].to_pylist())[order] for c in columns if c != "frame_index"}


def v3_meta(root):
    rows = []
    for f in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet")):
        rows += pq.read_table(
            f, columns=["episode_index", "data/chunk_index", "data/file_index", "tasks", "length"]
        ).to_pylist()
    return rows, json.loads((Path(root) / "meta/info.json").read_text())


def job(args):
    kind, spec = args
    robot, dec = Robot(), Decoder()
    enc = SonicEncoder(MODEL / "model_encoder.onnx", MODEL / "observation_config.yaml")
    res = {"kind": kind, **{k: v for k, v in spec.items() if k != "ep_row"}}
    try:
        if kind == "joint28":
            limits = load_joint_limits(CONVERTER_XML)
            src = v3_episode(spec["src"], spec["ep_row"], spec["src_info"], ["frame_index", "action"])[
                "action"
            ].astype(np.float32)
            nolimit = spec["conv"].endswith(
                "_nolimit"
            )  # reference must match how the stored tokens were made
            inputs, hands, report = build_encoder_inputs(
                src, limits, **({"arm_speed_limit": None, "hand_speed_limit": None} if nolimit else {})
            )
            fresh = enc.encode(inputs)
            stored = None
            try:
                stored = v3_episode(
                    spec["conv"], spec["ep_row"], spec["src_info"], ["frame_index", "action"]
                )["action"]
            except Exception as exc:  # incomplete dataset: file still being written
                res["stored_error"] = repr(exc)[:200]
            fps = 30
            ref_motor = inputs[:, 4:33].astype(np.float64)[:, MOTOR_FROM_ISAAC]  # filtered ref at source rows
            raw_motor = np.tile(NOMINAL_BODY, (len(src), 1))
            raw_motor[:, 15:] = src[:, :14]
            res["encoder_report"] = {
                k: report[k] for k in ("clipped_values", "rate_limited_values", "max_reference_time_error_s")
            }
            if spec.get("noslew"):  # ablation: same raw body, clip only, no slew limiter
                inputs, _ = generic_encoder_inputs(raw_motor, fps, robot.limits)
                ref_motor = inputs[:, 4:33].astype(np.float64)[:, MOTOR_FROM_ISAAC]
                fresh, stored = enc.encode(inputs), None
            if stored is not None:
                assert stored.shape == (len(src), 78), stored.shape
                res["stored_vs_fresh_token_maxabs"] = float(np.abs(stored[:, :64] - fresh).max())
                res["stored_vs_fresh_token_exact_frac"] = float(
                    (np.abs(stored[:, :64] - fresh) < 1e-6).all(1).mean()
                )
                res["dex3_stored_vs_source"] = err_stats(np.abs(stored[:, 64:] - src[:, 14:]))
                res["dex3_stored_vs_filtered"] = float(np.abs(stored[:, 64:] - hands).max())
                tokens = stored[:, :64].astype(np.float32)
                res["token_source"] = "stored"  # nosec B105 - provenance label
            else:
                tokens = fresh
                res["token_source"] = "fresh"  # nosec B105 - provenance label, not a secret
        elif kind == "teleop43":
            import pandas as pd

            df = pd.read_parquet(spec["path"], columns=["frame_index", "action"]).sort_values("frame_index")
            a = np.stack(df["action"].to_numpy()).astype(np.float64)
            names = spec["names"]
            raw_motor = a[:, [names.index(n) for n in BODY_NAMES]]
            fps = spec["fps"]
            inputs, clipped = generic_encoder_inputs(raw_motor, fps, robot.limits)
            ref_motor = inputs[:, 4:33].astype(np.float64)[:, MOTOR_FROM_ISAAC]
            tokens = enc.encode(inputs)
            res["encoder_report"] = {"clipped_values": clipped}
            res["token_source"] = "fresh (no stored 43D conversion exists)"  # nosec B105 - provenance label
        elif kind == "calib":
            if spec["path"] == "stand":
                raw_motor = np.tile(NOMINAL_BODY, (300, 1))
            else:
                jp = np.loadtxt(spec["path"], delimiter=",", skiprows=1)  # IsaacLab order, absolute
                raw_motor = jp[:, MOTOR_FROM_ISAAC]
            fps = spec["fps"]
            inputs, clipped = generic_encoder_inputs(raw_motor, fps, robot.limits)
            ref_motor = inputs[:, 4:33].astype(np.float64)[:, MOTOR_FROM_ISAAC]
            tokens = enc.encode(inputs)
            res["token_source"] = "fresh"  # nosec B105 - provenance label, not a secret
        start = spec.get("start", "placed")
        need_stand = spec.get("init") == "default" or start != "placed"
        stand = (
            enc.encode(generic_encoder_inputs(DEFAULT[None], 50, robot.limits)[0])[0] if need_stand else None
        )
        leadin = None
        if start == "hang_leadin":  # ≤0.5 rad/s smoothstep from default stance to the episode's first pose
            first = ref_motor[0]
            m = int(max(1.0, np.abs(first - DEFAULT).max() / 0.5) * 50) + 1
            a = np.linspace(0, 1, m)[:, None]
            leadin = enc.encode(
                generic_encoder_inputs(DEFAULT + (first - DEFAULT) * (3 * a**2 - 2 * a**3), 50, robot.limits)[
                    0
                ]
            )
        r = roundtrip(
            robot, dec, tokens, fps, ref_motor, raw_motor, stand, start, leadin, spec.get("interp", False)
        )
        k = r["k"]
        arms = slice(15, 29)
        res.update(
            frames=int(len(tokens)),
            frames_simulated=int(k),
            fps=fps,
            fell_at_s=r["fell_at_s"],
            palm_err_decoded_vs_raw_m=err_stats(np.linalg.norm(r["tgt_palm"] - r["raw_palm"], axis=-1)),
            palm_err_decoded_vs_filtered_m=err_stats(np.linalg.norm(r["tgt_palm"] - r["ref_palm"], axis=-1)),
            palm_err_achieved_vs_raw_m=err_stats(np.linalg.norm(r["ach_palm"] - r["raw_palm"], axis=-1)),
            palm_err_world_vs_raw_m=err_stats(np.linalg.norm(r["world_palm"] - r["ref_world_palm"], axis=-1)),
            palm_err_filtered_vs_raw_m=err_stats(np.linalg.norm(r["ref_palm"] - r["raw_palm"], axis=-1)),
            palm_err_per_hand_p95_decoded_vs_raw_m=[
                float(np.percentile(np.linalg.norm(r["tgt_palm"][:, h] - r["raw_palm"][:, h], axis=-1), 95))
                for h in (0, 1)
            ],
            arm_joint_abs_err_decoded_vs_filtered_rad=err_stats(
                np.abs(r["targets"][:, arms] - ref_motor[:k, arms])
            ),
            arm_joint_abs_err_achieved_vs_filtered_rad=err_stats(
                np.abs(r["achieved"][:, arms] - ref_motor[:k, arms])
            ),
            lower_joint_abs_err_decoded_vs_filtered_rad=err_stats(
                np.abs(r["targets"][:, :15] - ref_motor[:k, :15])
            ),
            per_joint_mae_decoded_vs_filtered_rad=np.abs(r["targets"] - ref_motor[:k])
            .mean(0)
            .round(4)
            .tolist(),
            per_joint_p95_decoded_vs_filtered_rad=np.percentile(
                np.abs(r["targets"] - ref_motor[:k]), 95, axis=0
            )
            .round(4)
            .tolist(),
            first_frame_palm_err_m=float(np.linalg.norm(r["tgt_palm"][0] - r["raw_palm"][0], axis=-1).max()),
            last_frame_palm_err_m=float(np.linalg.norm(r["tgt_palm"][-1] - r["raw_palm"][-1], axis=-1).max()),
            pelvis_drift_m=float(np.linalg.norm(r["pelvis_pos"][-1, :2] - r["pelvis_pos"][0, :2])),
            safety=r["safety"],
            startup=r["startup"],
            source_palm_range_m=float(np.ptp(r["raw_palm"].reshape(k, -1), axis=0).max()),
        )
        s, e = best_lag(r["tgt_palm"], r["raw_palm"])
        res["decoded_best_lag_frames"], res["decoded_palm_mean_err_at_best_lag_m"] = s, e
        s, e = best_lag(r["ach_palm"], r["raw_palm"])
        res["achieved_best_lag_frames"], res["achieved_palm_mean_err_at_best_lag_m"] = s, e
        out = Path(spec["out"])
        np.savez_compressed(
            out.with_suffix(".npz"),
            **{k2: v for k2, v in r.items() if isinstance(v, np.ndarray)},
            ref_motor=ref_motor[:k],
            raw_motor=raw_motor[:k],
            tokens=tokens[:k],
        )
        out.write_text(json.dumps(res, indent=1))
    except Exception:
        import traceback

        res["error"] = traceback.format_exc()[-2000:]
        Path(spec["out"]).write_text(json.dumps(res, indent=1))
    return res


def plan(args):
    rng = np.random.default_rng(0)
    jobs = []
    out = Path(args.out)
    if "calib" in args.sets:
        d = "/gear_sonic_deploy/reference/example/dance_in_da_party_001__A464"
        jobs.append(("calib", {"set": "calib", "episode": "stand", "path": "stand", "fps": 30}))
        jobs.append(
            (
                "calib",
                {
                    "set": "calib",
                    "episode": "dance_in_da_party_001__A464",
                    "path": f"{d}/joint_pos.csv",
                    "fps": 50,
                },
            )
        )
    for name, src, conv in (
        ("unitree", "/run-output/datasets/joint28", "/run-output/datasets/sonic78"),
        (
            "humanoid_everyday",
            "/run-output/humanoid_everyday_g1_20260923/datasets/joint28",
            "/run-output/humanoid_everyday_g1_20260923/datasets/sonic78",
        ),
    ):
        if name not in args.sets:
            continue
        rows, info = v3_meta(src)
        by_task = {}
        for r in rows:
            by_task.setdefault(r["tasks"][0], []).append(r)
        picked = {}
        tasks = list(by_task)
        if len(tasks) > args.max_tasks:  # ponytail: uniform task subsample, stratify by source if HE needs it
            tasks = [tasks[i] for i in sorted(rng.choice(len(tasks), args.max_tasks, replace=False))]
        for task in tasks:
            rs = by_task[task]
            for i in rng.choice(len(rs), min(args.per_task, len(rs)), replace=False):
                picked[rs[i]["episode_index"]] = rs[i]
        # extremes from conversion reports (clipping) when available
        prov = Path(conv) / "meta/provenance.json"
        if prov.exists():
            reps = json.loads(prov.read_text()).get("episode_reports", [])
            by_ep = {r["episode_index"]: r for r in rows}
            for rep in sorted(reps, key=lambda r: -r["clipped_values"] / r["frames"])[: args.extremes]:
                picked[rep["episode_index"]] = by_ep[rep["episode_index"]]
            for rep in sorted(reps, key=lambda r: -r["rate_limited_values"] / r["frames"])[: args.extremes]:
                picked[rep["episode_index"]] = by_ep[rep["episode_index"]]
        for ep, r in sorted(picked.items()):
            jobs.append(
                (
                    "joint28",
                    {
                        "set": name,
                        "episode": ep,
                        "task": r["tasks"][0],
                        "src": src,
                        "conv": conv,
                        "src_info": info,
                        "ep_row": r,
                    },
                )
            )
    if "teleop43" in args.sets:
        for ds in sorted(glob.glob("/hf/g1_teleop/*/")):
            info = json.loads(Path(ds, "meta/info.json").read_text())
            files = sorted(glob.glob(f"{ds}/data/*/*.parquet"))
            for i in rng.choice(len(files), min(args.per_task, len(files)), replace=False):
                jobs.append(
                    (
                        "teleop43",
                        {
                            "set": "teleop43",
                            "task": Path(ds).name,
                            "episode": Path(files[i]).stem,
                            "path": files[i],
                            "fps": info["fps"],
                            "names": info["features"]["action"]["names"],
                        },
                    )
                )
    if args.conv_suffix or args.interp:  # same episode picks; stored tokens from a variant dataset
        tag = (args.conv_suffix or "") + {"delayed": "_interpdelayed", True: "_interp", False: ""}[
            args.interp
        ]
        jobs = [
            (
                k,
                {
                    **s,
                    "set": s["set"] + tag,
                    "conv": s["conv"] + (args.conv_suffix or ""),
                    "interp": args.interp,
                },
            )
            for k, s in jobs
            if k == "joint28"
        ]
    if args.start != "placed":
        jobs = [
            (k, {**s, "set": s["set"] + "_" + args.start, "start": args.start})
            for k, s in jobs
            if k == "joint28"
        ]
    if args.init_default:
        jobs = [
            (k, {**s, "set": s["set"] + "_initdefault", "init": "default"}) for k, s in jobs if k == "joint28"
        ]
    if args.noslew:
        if args.noslew == "all":
            jobs += [
                (k, {**s, "set": s["set"] + "_noslew", "noslew": True}) for k, s in jobs if k == "joint28"
            ]
        else:
            worst = json.loads(Path(args.noslew).read_text())
            keep = {(n, w["episode"]) for n in ("unitree", "humanoid_everyday") for w in worst[n]["worst"]}
            jobs = [
                (k, {**s, "set": s["set"] + "_noslew", "noslew": True})
                for k, s in jobs
                if (s["set"], s.get("episode")) in keep
            ]
    for _kind, spec in jobs:
        spec["out"] = str(out / "episodes" / f"{spec['set']}__{spec['episode']}.json")
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/audit/results")
    p.add_argument("--sets", nargs="+", default=["calib", "unitree", "humanoid_everyday", "teleop43"])
    p.add_argument("--per-task", type=int, default=4)
    p.add_argument("--extremes", type=int, default=4)
    p.add_argument("--max-tasks", type=int, default=20)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int)
    p.add_argument("--conv-suffix", help="read stored tokens from <sonic78 dir><suffix>, e.g. _nolimit")
    p.add_argument(
        "--interp",
        nargs="?",
        const=True,
        default=False,
        choices=[True, "delayed"],
        help="linearly interpolate source-rate tokens to 50 Hz (look-ahead); 'delayed' = causal, one frame later",
    )
    p.add_argument(
        "--start",
        choices=["placed", "hang", "hang_leadin"],
        default="placed",
        help="placed: on the floor in the first pose; hang: official elastic-band start then release",
    )
    p.add_argument(
        "--init-default", action="store_true", help="start each episode from SONIC's default standing pose"
    )
    p.add_argument(
        "--noslew",
        help="breakdown.json: rerun its worst episodes without the slew limiter; 'all': run every joint28 job both ways",
    )
    a = p.parse_args()
    (Path(a.out) / "episodes").mkdir(parents=True, exist_ok=True)
    jobs = plan(a)[: a.limit]
    todo = [j for j in jobs if not Path(j[1]["out"]).exists()]
    print(f"{len(jobs)} jobs, {len(todo)} to run", flush=True)
    prov = {
        "encoder_sha256": sha(MODEL / "model_encoder.onnx"),
        "decoder_sha256": sha(MODEL / "model_decoder.onnx"),
        "observation_config_sha256": sha(MODEL / "observation_config.yaml"),
        "scene_sha256": sha(SCENE),
        "sim_robot_xml_sha256": sha(Path(SCENE).parent / "g1_29dof_with_hand.xml"),
        "converter_xml_sha256": sha(CONVERTER_XML),
        "sonic_targets_sha256": sha("/code/sonic_targets.py"),
        "mujoco": mujoco.__version__,
        "onnxruntime": ort.__version__,
        "kp": KP.round(4).tolist(),
        "kd": KD.round(4).tolist(),
        "action_scale": SCALE.round(5).tolist(),
    }
    (Path(a.out) / "provenance.json").write_text(json.dumps(prov, indent=1))
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(job, todo):
            msg = r.get("error", "").splitlines()[-1:] or [
                f"p95 dec/raw={r['palm_err_decoded_vs_raw_m']['p95']:.3f} ach/raw={r['palm_err_achieved_vs_raw_m']['p95']:.3f} world={r['palm_err_world_vs_raw_m']['p95']:.3f} fell={r['fell_at_s']}"
            ]
            print(r["set"], r["episode"], msg[0], flush=True)


def summarize(out, gate_m=0.05):
    """Corpus metrics per dataset (never mixing widths); gate = p95 palm error > gate_m fails."""
    rows = [json.loads(p.read_text()) for p in sorted(Path(out, "episodes").glob("*.json"))]
    summary = {}
    for name in sorted({r["set"] for r in rows}):
        rs = [r for r in rows if r["set"] == name]
        ok = [r for r in rs if "error" not in r]
        npz = [np.load(Path(r["out"]).with_suffix(".npz")) for r in ok]

        def pool(a, b, npz=npz):
            return (
                np.concatenate([np.linalg.norm(d[a] - d[b], axis=-1).ravel() for d in npz])
                if npz
                else np.zeros(1)
            )

        ach, dec = pool("ach_palm", "raw_palm"), pool("tgt_palm", "raw_palm")
        ach_f, flt = pool("ach_palm", "ref_palm"), pool("ref_palm", "raw_palm")
        world = (
            np.concatenate(
                [np.linalg.norm(d["world_palm"] - d["ref_world_palm"], axis=-1).ravel() for d in npz]
            )
            if npz
            else np.zeros(1)
        )
        ep_p95 = sorted(((r["palm_err_achieved_vs_raw_m"]["p95"], r["episode"]) for r in ok), reverse=True)
        summary[name] = {
            "episodes": len(rs),
            "errors": [r["episode"] for r in rs if "error" in r],
            "falls": [r["episode"] for r in ok if r["fell_at_s"] is not None],
            "palm_achieved_vs_raw_m": err_stats(ach),
            "palm_achieved_vs_filtered_m": err_stats(ach_f),
            "palm_filtered_vs_raw_m (clip+slew only)": err_stats(flt),
            "palm_decoded_setpoint_vs_raw_m": err_stats(dec),
            "palm_world_vs_raw_m": err_stats(world),
            "episodes_over_gate_frac": float(np.mean([p > gate_m for p, _ in ep_p95])) if ep_p95 else None,
            "worst_episodes": ep_p95[:5],
            "median_achieved_lag_frames": float(np.median([r["achieved_best_lag_frames"] for r in ok]))
            if ok
            else None,
            "palm_achieved_mean_err_at_best_lag_m": err_stats(
                [r["achieved_palm_mean_err_at_best_lag_m"] for r in ok]
            )
            if ok
            else None,
            "token_stored_vs_fresh_maxabs": max(
                (r.get("stored_vs_fresh_token_maxabs", 0) for r in ok), default=None
            ),
            "dex3_stored_vs_source_max_rad": max(
                (r["dex3_stored_vs_source"]["max"] for r in ok if "dex3_stored_vs_source" in r), default=None
            ),
            "dex3_stored_vs_source_p95_rad": float(
                np.median([r["dex3_stored_vs_source"]["p95"] for r in ok if "dex3_stored_vs_source" in r])
            )
            if any("dex3_stored_vs_source" in r for r in ok)
            else None,
            "verdict_gate": f"p95 achieved palm error vs raw source > {gate_m} m => FAIL",
            "verdict": None if not ok else ("FAIL" if err_stats(ach)["p95"] > gate_m else "PASS"),
        }
    Path(out, "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    if sys.argv[1:2] == ["summarize"]:
        print(json.dumps(summarize(sys.argv[2]), indent=1))
    else:
        main()
