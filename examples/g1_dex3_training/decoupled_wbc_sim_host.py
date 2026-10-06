"""Sim host for sonic_policy_streamer.py --backend decoupled: NVIDIA's GR00T decoupled WBC (lower-body RL for legs +
waist, arm and Dex3 hand joint targets straight to PD) in the same scene as sonic_official_sim_host.py (sim_scene).

Contract (docs/superpowers/specs/2026-10-01-g1-wbc-backends-design.md; upstream NVlabs/GR00T-WholeBodyControl
b042411fae with only decoupled_wbc/ mounted at /upstream/decoupled_wbc; onnxruntime on PYTHONPATH):
  lower body  G1GearWbcPolicy with GEAR_WBC_CONFIG of control/main/teleop/configs/g1_29dof_gear_wbc.yaml (516 obs,
              15 actions) and the Balance/Walk ONNX; walking command 0, height 0.74 (Balance runs throughout)
  upper body  IdentityPolicy: the streamer's 50 Hz targets as they are; every wrapper goal is complete;
              the first goal is the start pose: arms in A's planner-stance pose (START_ARMS), hands and waist at the
              measured pose; the gear-WBC default arm pose reaches table-top height (0.80 m)
  waist       robot model waist_location lower_body: torso command 0, the RL holds waist + legs
  activation  RL output switched on through the lower-body policy (key "]"), never a toggle-only wrapper goal
  gains       body MOTOR_KP / MOTOR_KD of g1_29dof_gear_wbc.yaml; Dex3 kp 1.5 / kd 0.1 (A's deploy defaults,
              dex3_hands.hpp), written into gear_sonic's bridge command slots by joint name (no DDS, no slot swap)
  rates       physics 200 Hz (A's scene; upstream sim_frequency 200), control 50 Hz
  messages    ZMQ SUB "joints" (wbc_common.pack_joint_message) on --action-port; drops messages older than 0.1 s,
              holds the previous target on stale or non-finite ones, clips to the MuJoCo joint ranges (counted);
              ZMQ PUB "g1_debug" (body_q motor order, left/right_hand_q dataset order) on --state-port
Timeline: band on, RL off -> 1 s RL on -> 3 s band released -> 4 s reset onto the ground -> 9 s policy flags checked,
table placed, GATE/deploy_ready + GATE/settled, fall detection armed -> GATE/done + 3 s: stop. Streamer silent for
5 s without GATE/done: aborted.
Usage (in jihun/sonic-vla-sim, cwd gear_sonic_deploy):
  python decoupled_wbc_sim_host.py OUT_DIR GATE_DIR [--selfcheck contract|state|command|stand]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import msgpack
import mujoco
import numpy as np
import yaml
import zmq
from sim_scene import BODY, HANDS, Scene
from wbc_common import (
    REF_NAMES,
    WAIST_NAMES,
    pack_joint_message,
    read_done,
    tilt_deg,
    unpack_joint_message,
    write_termination,
)

UPSTREAM = Path("/upstream/decoupled_wbc")
WBC_YAML = UPSTREAM / "control/main/teleop/configs/g1_29dof_gear_wbc.yaml"
POLICY_DIR = UPSTREAM / "sim2mujoco/resources/robots/g1"
MODEL_PATH = "policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx"
HAND_KP, HAND_KD = (
    1.5,
    0.1,
)  # Dex3Hands defaults in A's deploy (dex3_hands.hpp:396-397), kept by setAllJointsCommand
DECIMATION = 4  # 200 Hz physics -> 50 Hz control
MAX_MSG_AGE_S, SILENCE_S = 0.1, 5.0
FIRST_MSG_S = 600.0  # must exceed policy-server load (MolmoAct2 ~2 min) + streamer startup + first inference
GOAL_CONST = {"base_height_command": np.array([0.74]), "navigate_cmd": np.zeros(3)}
T_ACTIVATE, T_BAND, T_RESET, T_SETTLED = 1.0, 3.0, 4.0, 9.0
# Arms during hang/settle: the SONIC planner stance backend A holds before its episode (mean of the pre-episode records
# of WBC_A28_ep1293_r1 on the second H100 host, std <= 0.007 rad). The G1 zero arm pose holds the forearms forward at table-top
# height (palms ~0.78 m); this pose keeps the palms at ~0.62 m, below an 80 cm table, as for backend A.
START_ARMS = np.array(
    [-0.046, 0.293, -0.628, 1.027, -0.162, -0.119, 0.099, -0.054, -0.267, 0.663, 1.128, -0.162, 0.251, -0.165]
)  # REF_NAMES[:14] order (motor 15..28)


def make_wbc(waist_location: str):
    from decoupled_wbc.control.policy.g1_decoupled_whole_body_policy import G1DecoupledWholeBodyPolicy
    from decoupled_wbc.control.policy.g1_gear_wbc_policy import G1GearWbcPolicy
    from decoupled_wbc.control.policy.identity_policy import IdentityPolicy
    from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model

    cfg = yaml.safe_load(WBC_YAML.read_text())
    robot_model = instantiate_g1_robot_model(waist_location=waist_location)
    lower = G1GearWbcPolicy(
        robot_model=robot_model, config=str(UPSTREAM.parent / cfg["GEAR_WBC_CONFIG"]), model_path=MODEL_PATH
    )
    if (lower.config["num_obs"], lower.config["num_actions"]) != (516, 15):
        raise RuntimeError(
            f"unexpected lower-body config {lower.config['num_obs']} obs / {lower.config['num_actions']}"
        )
    wbc = G1DecoupledWholeBodyPolicy(
        robot_model=robot_model, upper_body_policy=IdentityPolicy(), lower_body_policy=lower
    )
    return wbc, robot_model, np.asarray(cfg["MOTOR_KP"], float), np.asarray(cfg["MOTOR_KD"], float)


class Plant:
    """PD targets by joint name, written into gear_sonic's bridge command slots (its sim_step computes the torques)."""

    def __init__(self, scene: Scene, kp: np.ndarray, kd: np.ndarray):
        env, m = scene.env, scene.m
        names = lambda idx: [m.joint(int(j)).name for j in idx]  # noqa: E731 - bridge slots hold 1-based joint ids
        self.body_slots = names(env.body_joint_index)
        self.left_slots, self.right_slots = names(env.left_hand_index), names(env.right_hand_index)
        if sorted(self.body_slots) != sorted(BODY) or sorted(self.left_slots + self.right_slots) != sorted(
            HANDS
        ):
            raise RuntimeError(
                f"bridge slots {self.body_slots} / {self.left_slots + self.right_slots} do not match the scene"
            )

        def cmd(n):
            return SimpleNamespace(
                motor_cmd=[SimpleNamespace(q=0.0, dq=0.0, kp=0.0, kd=0.0, tau=0.0) for _ in range(n)]
            )

        self.br = env.unitree_bridge
        self.br.low_cmd, self.br.left_hand_cmd, self.br.right_hand_cmd = (
            cmd(len(self.body_slots)),
            cmd(7),
            cmd(7),
        )
        for i, n in enumerate(self.body_slots):
            c = self.br.low_cmd.motor_cmd[i]
            c.kp, c.kd = kp[BODY.index(n)], kd[BODY.index(n)]
        for c in self.br.left_hand_cmd.motor_cmd + self.br.right_hand_cmd.motor_cmd:
            c.kp, c.kd = HAND_KP, HAND_KD

    def set(self, targets: dict) -> None:
        for i, n in enumerate(self.body_slots):
            self.br.low_cmd.motor_cmd[i].q = float(targets[n])
        for slots, cmd in (
            (self.left_slots, self.br.left_hand_cmd),
            (self.right_slots, self.br.right_hand_cmd),
        ):
            for i, n in enumerate(slots):
                cmd.motor_cmd[i].q = float(targets[n])


class Controller:
    """50 Hz decoupled-WBC step: MuJoCo state -> upstream observation, complete goal, PD targets by joint name."""

    def __init__(self, scene: Scene, waist_location: str):
        self.scene = scene
        assert abs(scene.dt * DECIMATION - 0.02) < 1e-9, (
            f"need physics 200 Hz / control 50 Hz, got dt {scene.dt} x {DECIMATION}"
        )
        self.wbc, self.rm, kp, kd = make_wbc(waist_location)
        self.lower = self.wbc.lower_body_policy
        m = scene.m
        self.rm_names = list(self.rm.joint_names)
        missing = [n for n in self.rm_names if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) < 0]
        if missing:
            raise RuntimeError(f"robot-model joints missing in the MuJoCo scene: {missing}")
        self.rm_qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in self.rm_names])
        self.rm_vadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in self.rm_names])
        self.upper_names = [self.rm_names[i] for i in self.rm.get_joint_group_indices("upper_body")]
        self.plant = Plant(scene, kp, kd)
        # start from the measured pose (arms down after the reset); the gear-WBC default arm pose reaches table-top height
        # (the arms are then set to START_ARMS; the hands and waist keep the measured pose)
        self.ref = {
            n: float(scene.d.qpos[m.jnt_qposadr[m.joint(n).id]])
            for n in set(self.upper_names) | set(REF_NAMES)
        }
        self.ref.update(
            zip(REF_NAMES[:14], START_ARMS.tolist(), strict=True)
        )  # arms start at START_ARMS; hands, waist: measured
        self.ranges = {n: m.jnt_range[m.joint(n).id].copy() for n in REF_NAMES}
        self.clipped, self.held, self.seq = np.zeros(len(REF_NAMES), bool), False, -1
        self.counts = {"messages": 0, "stale": 0, "nonfinite": 0, "seq_gap": 0, "clipped_values": 0}
        self.last_msg_wall = None

    def apply_message(self, msg: dict) -> None:
        self.held = False
        if time.time() - msg["t_wall"] > MAX_MSG_AGE_S:
            self.counts["stale"] += 1
            self.held = True
            return
        vals = np.r_[msg["q_body"][15:29], msg["q_hand"], msg["q_body"][12:15]]
        if not np.isfinite(vals).all():
            self.counts["nonfinite"] += 1
            self.held = True
            return
        if self.seq >= 0:
            self.counts["seq_gap"] += max(0, int(msg["seq"]) - self.seq - 1)
        self.seq, self.last_msg_wall = int(msg["seq"]), time.time()
        self.counts["messages"] += 1
        for i, n in enumerate(REF_NAMES):
            lo, hi = self.ranges[n]
            v = float(np.clip(vals[i], lo, hi))
            self.clipped[i] = v != vals[i]
            self.ref[n] = v
        self.counts["clipped_values"] += int(self.clipped.sum())

    def applied_ref(self) -> np.ndarray:
        return np.array([self.ref[n] for n in REF_NAMES], np.float32)

    def observation(self) -> dict:
        d = self.scene.d
        return {"q": d.qpos[self.rm_qadr].copy(), "dq": d.qvel[self.rm_vadr].copy(),
                "floating_base_pose": d.qpos[:7].copy(), "floating_base_vel": d.qvel[:6].copy()}  # fmt: skip

    def goal(self) -> dict:
        return {"target_upper_body_pose": np.array([self.ref[n] for n in self.upper_names]), **GOAL_CONST}

    def step(self, t: float) -> None:
        self.wbc.set_observation(self.observation())
        self.wbc.set_goal(self.goal())  # always complete: IdentityPolicy returns exactly this goal
        q = self.wbc.get_action(time=t)["q"]
        targets = dict(zip(self.rm_names, q, strict=True))
        targets.update({n: self.ref[n] for n in HANDS})  # hands straight from the joint message
        with self.scene.lock:
            self.plant.set(targets)

    def activate(self) -> None:
        self.lower.handle_keyboard_button("]")  # use_policy_action = True, on the lower-body policy only

    def flags(self) -> tuple[bool, bool]:
        return bool(self.lower.use_policy_action), bool(self.lower.use_teleop_policy_cmd)


def state_message(scene: Scene) -> bytes:
    d = scene.d
    return b"g1_debug" + msgpack.packb(
        {
            "body_q": d.qpos[scene.qadr].tolist(),
            "left_hand_q": d.qpos[scene.hadr[:7]].tolist(),
            "right_hand_q": d.qpos[scene.hadr[7:]].tolist(),
        }  # fmt: skip
    )


def startup(scene: Scene, ctl: Controller, steps: dict, t: float) -> None:
    """Band hang -> RL on -> band released -> reset onto the ground (shared by the run and the stand self-check)."""
    if "active" not in steps and t >= T_ACTIVATE:
        steps["active"] = t
        ctl.activate()
        scene.mark("lower-body RL output on")
        scene.view["phase"] = "RL on, hung"
    if "band" not in steps and t >= T_BAND:
        steps["band"] = t
        scene.sim.handle_keyboard_button("9")
        scene.mark("sim key '9': band released")
        scene.view["phase"] = "band released"
    if "reset" not in steps and t >= T_RESET:
        steps["reset"] = t
        scene.sim.handle_keyboard_button("backspace")
        scene.mark("sim key 'Backspace': reset onto ground")
        scene.view["phase"] = "settling on ground"


def run(a) -> None:
    try:
        scene = Scene(a.out, os.environ.get("TABLE_GAP_CM"))
    except Exception:
        write_termination(a.out, "error", "startup", traceback.format_exc(limit=4))
        raise
    steps, n, t_done, episode_end = {}, 0, None, None
    reason, detail = "error", ""
    ctl = None
    wall0 = time.monotonic()
    try:
        ctl = Controller(scene, a.waist_location)
        for f in MODEL_PATH.split(","):
            scene.mark(f"onnx {f} sha256 {hashlib.sha256((POLICY_DIR / f).read_bytes()).hexdigest()}")
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.SUBSCRIBE, b"joints")
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.connect(f"tcp://localhost:{a.action_port}")
        pub = ctx.socket(zmq.PUB)
        pub.bind(f"tcp://*:{a.state_port}")
        scene.mark(f"decoupled WBC host started (waist_location {a.waist_location})")
        wall0 = time.monotonic()
        while True:
            scene.step()
            n += 1
            t = scene.t
            if n % DECIMATION == 0:
                # held stays set until a valid message replaces the rejected target (apply_message sets it either
                # way); clipped describes this tick's message only
                ctl.clipped[:] = False
                with contextlib.suppress(zmq.Again):
                    ctl.apply_message(unpack_joint_message(sub.recv(zmq.NOBLOCK)))
                ctl.step(t)
                pub.send(state_message(scene))
                startup(scene, ctl, steps, t)
                if "settled" not in steps and t >= T_SETTLED:
                    if ctl.flags() != (True, True):
                        raise RuntimeError(
                            f"policy flags (use_policy_action, use_teleop_policy_cmd) = {ctl.flags()}"
                        )
                    steps["settled"] = t
                    scene.place_table()
                    scene.fall.armed = True
                    (a.gate / "deploy_ready").touch()
                    (a.gate / "settled").touch()
                    scene.mark("settled -> deploy_ready + settled")
                    scene.view["phase"] = "streaming joint targets"
                if "settled" in steps:
                    scene.record(applied_ref=ctl.applied_ref(), clipped=ctl.clipped.copy(), held=ctl.held,
                                 seq=ctl.seq)  # fmt: skip
                    fell = scene.check_fall()
                    if fell:
                        reason, detail = "fell", fell
                        break
                if t_done is None and (a.gate / "done").exists():
                    t_done, episode_end = t, read_done(a.gate)
                    scene.mark(f"streamer done: {episode_end}")
                if t_done is not None and t >= t_done + 3:
                    reason, detail = (
                        ("completed", "")
                        if episode_end == "completed"
                        else ("aborted", f"streamer: {episode_end}")
                    )
                    break
                if (
                    t_done is None
                    and ctl.last_msg_wall is not None
                    and time.time() - ctl.last_msg_wall > SILENCE_S
                ):
                    reason, detail = "aborted", f"no joint message for {SILENCE_S:g} s and no GATE/done"
                    break
                if "settled" in steps and ctl.last_msg_wall is None and t >= steps["settled"] + FIRST_MSG_S:
                    reason, detail = "aborted", f"no joint message within {FIRST_MSG_S:g} s of settled"
                    break
                if t > 1800:
                    reason, detail = "timeout", "1800 s of sim time"
                    break
            scene.pace(wall0)
    except Exception:
        detail = traceback.format_exc(limit=4)
        raise
    finally:
        counts = ctl.counts if ctl else {}
        try:
            scene.close(reason, detail, counts=counts, waist_location=a.waist_location)
        except Exception as e:
            write_termination(
                a.out, reason, "close", detail + repr(e), counts=counts, waist_location=a.waist_location
            )


def selfcheck(a) -> None:
    scene = Scene(a.out, None)
    m, d = scene.m, scene.d
    if a.selfcheck == "contract":
        from decoupled_wbc.control.utils.gear_wbc_utils import get_gravity_orientation

        ctl = Controller(scene, a.waist_location)
        for f in MODEL_PATH.split(","):
            print(f, hashlib.sha256((POLICY_DIR / f).read_bytes()).hexdigest())
        print("upper_body", ctl.upper_names)
        print("lower_body", [ctl.rm_names[i] for i in ctl.rm.get_joint_group_indices("lower_body")])
        assert set(REF_NAMES[:14]) <= set(ctl.upper_names), "arms must be upper-body joints"
        assert not set(WAIST_NAMES) <= set(ctl.upper_names), "waist must stay a lower-body joint"
        d.qpos[3:7] = [
            np.cos(0.1),
            0,
            np.sin(0.1),
            0,
        ]  # 11.5 deg pitch: quaternion order must be MuJoCo's (w,x,y,z)
        mujoco.mj_forward(m, d)
        expect = d.xmat[scene.pelvis].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        got = np.asarray(get_gravity_orientation(d.qpos[3:7].copy()), float)
        assert np.allclose(got, expect, atol=1e-6), (got, expect)
    elif a.selfcheck == "state":
        names = list(BODY) + list(HANDS)
        vals = {}
        for i, n in enumerate(names):
            lo, hi = m.jnt_range[m.joint(n).id]
            vals[n] = lo + (hi - lo) * (i + 1) / (len(names) + 1)
            d.qpos[m.jnt_qposadr[m.joint(n).id]] = vals[n]
        msg = msgpack.unpackb(state_message(scene)[len(b"g1_debug") :], raw=False)
        assert np.allclose(msg["body_q"], [vals[n] for n in BODY])
        assert np.allclose(msg["left_hand_q"], [vals[n] for n in HANDS[:7]])
        assert np.allclose(msg["right_hand_q"], [vals[n] for n in HANDS[7:]])
    elif a.selfcheck == "command":
        ctl = Controller(scene, "lower_body")
        ref = np.array(
            [lo + (hi - lo) * (i + 1) / 32 for i, (lo, hi) in enumerate(ctl.ranges[n] for n in REF_NAMES)],
            np.float32,
        )
        assert len(set(np.round(ref[:28], 6))) == 28, "test values must be distinct"
        ctl.apply_message(unpack_joint_message(pack_joint_message(0, "selfcheck", -1, ref)))
        ctl.step(0.0)  # also proves a complete goal: get_action runs without KeyError (spec test 9)
        br = scene.env.unitree_bridge
        for i, n in enumerate(ctl.plant.body_slots):
            if n in REF_NAMES[:14]:
                assert abs(br.low_cmd.motor_cmd[i].q - ref[REF_NAMES.index(n)]) < 1e-6, n
        for slots, cmd in (
            (ctl.plant.left_slots, br.left_hand_cmd),
            (ctl.plant.right_slots, br.right_hand_cmd),
        ):
            for i, n in enumerate(slots):
                assert abs(cmd.motor_cmd[i].q - ref[REF_NAMES.index(n)]) < 1e-6, n
        print("right-hand bridge slots:", ctl.plant.right_slots)
    elif a.selfcheck == "stand":
        ctl = Controller(scene, a.waist_location)
        steps, n, tilts = {}, 0, []
        scene.fall.armed = False
        while scene.t < T_SETTLED + 30:
            scene.step()
            n += 1
            if n % DECIMATION == 0:
                ctl.step(scene.t)
                startup(scene, ctl, steps, scene.t)
                if scene.t >= T_SETTLED:
                    scene.fall.armed = True
                    tilts.append(tilt_deg(d.qpos[3:7]))
                    fell = scene.check_fall()
                    assert fell is None, fell
        assert max(tilts) < 3.0, f"max tilt {max(tilts):.2f} deg"
        print(f"stand: max tilt {max(tilts):.2f} deg over 30 s")
    scene.view["stop"] = True
    print(f"selfcheck {a.selfcheck}: ok")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path)
    p.add_argument("gate", type=Path)
    p.add_argument("--waist-location", choices=["lower_body"], default="lower_body")
    p.add_argument("--action-port", type=int, default=5556)
    p.add_argument("--state-port", type=int, default=5557)
    p.add_argument("--selfcheck", choices=["contract", "state", "command", "stand"])
    a = p.parse_args()
    a.gate.mkdir(parents=True, exist_ok=True)
    selfcheck(a) if a.selfcheck else run(a)


if __name__ == "__main__":
    main()
