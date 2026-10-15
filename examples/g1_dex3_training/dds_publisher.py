"""DDS side of the Isaac scene, in its own process: gear_sonic's UnitreeSdk2Bridge publishes the robot state the
scene writes to shared memory and copies the deploy's latest commands back into it.

Why a process: the bridge serialises its state messages in pure Python (cyclonedds IDL), about 2 ms per 5 ms physics
step, a third of each step in the scene (test 0b profile); a thread would still hold the GIL. A script, not
multiprocessing: a spawned child re-imports the host script, which starts the deploy.

Shared memory, two float64 blocks, each [seq, stop, payload] with a seqlock (seq odd while writing):
  state: STATE fields in order (joints in BODY + DEX3_MOTORS order, quaternions w x y z, velocities lin 3 + ang 3)
  cmd:   q, dq, kp, kd, tau per motor in the same joint order (seq 0 until the deploy has sent a command)
Usage (spawned by isaac_scene.Scene): python dds_publisher.py STATE_SHM CMD_SHM CONFIG_JSON PARENT_PID
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

NB, NH = 29, 7  # body motors, motors per Dex3 hand (isaac_scene checks these against BODY / DEX3_MOTORS)
N = NB + 2 * NH
STATE = (("q", N), ("dq", N), ("ddq", N), ("tau", N), ("root_pose", 7), ("root_vel", 6), ("root_acc", 6),
         ("torso_quat", 4), ("torso_vel", 6), ("time", 1))  # fmt: skip
CMD = ("q", "dq", "kp", "kd", "tau")
HEAD = 2  # seq, stop


class _Block:
    def __init__(self, size: int, name: str | None):
        nbytes = (HEAD + size) * 8
        self.shm = shared_memory.SharedMemory(name=name, create=name is None, size=nbytes)
        self.a = np.ndarray(HEAD + size, dtype=np.float64, buffer=self.shm.buf)
        if name is None:
            self.a[:] = 0.0

    def write(self, payload: np.ndarray) -> None:
        self.a[0] += 1  # odd: writing
        self.a[HEAD:] = payload
        self.a[0] += 1

    def read(self, after: float) -> tuple[float, np.ndarray] | None:
        while True:
            s1 = self.a[0]
            if s1 == after or s1 == 0:
                return None
            if int(s1) % 2:
                continue
            data = self.a[HEAD:].copy()
            if self.a[0] == s1:
                return s1, data


class Exchange:
    """Create the two blocks (names=None, the scene) or attach to them by name (the publisher process)."""

    def __init__(self, names: tuple[str, str] | None = None):
        self.owner = names is None
        self.state = _Block(sum(n for _, n in STATE), None if names is None else names[0])
        self.cmd = _Block(len(CMD) * N, None if names is None else names[1])

    @property
    def names(self) -> tuple[str, str]:
        return self.state.shm.name, self.cmd.shm.name

    def write_state(self, s: dict) -> None:
        self.state.write(np.concatenate([np.atleast_1d(np.asarray(s[k], dtype=np.float64)) for k, _ in STATE]))

    def read_state(self, after: float) -> tuple[float, dict] | None:
        got = self.state.read(after)
        if got is None:
            return None
        seq, flat, out, i = *got, {}, 0
        for k, n in STATE:
            out[k], i = flat[i : i + n], i + n
        out["time"] = float(out["time"][0])
        return seq, out

    def write_cmd(self, c: dict) -> None:
        self.cmd.write(np.concatenate([c[k] for k in CMD]))

    def read_cmd(self) -> dict | None:
        got = self.cmd.read(0.0)
        return None if got is None else dict(zip(CMD, got[1].reshape(len(CMD), N), strict=True))

    def set_stop(self) -> None:
        self.state.a[1] = 1.0

    def stopped(self) -> bool:
        return bool(self.state.a[1])

    def close(self, unlink: bool = False) -> None:
        del self.state.a, self.cmd.a  # release the numpy views before closing the buffers
        for b in (self.state, self.cmd):
            b.shm.close()
            if unlink:
                b.shm.unlink()


def make_obs(s: dict) -> dict:
    """UnitreeSdk2Bridge.PublishLowState input from the STATE fields."""
    obs = {
        "body_q": s["q"][:NB], "body_dq": s["dq"][:NB], "body_ddq": s["ddq"][:NB], "body_tau_est": s["tau"][:NB],
        "floating_base_pose": s["root_pose"], "floating_base_vel": s["root_vel"], "floating_base_acc": s["root_acc"],
        "secondary_imu_quat": s["torso_quat"], "secondary_imu_vel": s["torso_vel"], "time": s["time"],
    }  # fmt: skip
    for side, sl in (("left", slice(NB, NB + NH)), ("right", slice(NB + NH, N))):
        obs[f"{side}_hand_q"], obs[f"{side}_hand_dq"] = s["q"][sl], s["dq"][sl]
    return obs


def pump(ex: Exchange, bridge, last_seq: float) -> float:
    """One round: publish the newest state if there is one, then copy the deploy's latest commands to `ex`."""
    got = ex.read_state(last_seq)
    if got is not None:
        last_seq, s = got
        bridge.PublishLowState(make_obs(s))
    if bridge.cmd_received():
        motors = [bridge.low_cmd.motor_cmd[i] for i in range(NB)]
        motors += [bridge.left_hand_cmd.motor_cmd[i] for i in range(NH)]
        motors += [bridge.right_hand_cmd.motor_cmd[i] for i in range(NH)]
        ex.write_cmd({k: np.array([getattr(m, k) for m in motors], dtype=np.float64) for k in CMD})
    return last_seq


def main(argv: list[str]) -> None:
    state_name, cmd_name, config_path, parent = argv[1], argv[2], Path(argv[3]), int(argv[4])
    config = json.loads(config_path.read_text())
    import ctypes

    ctypes.CDLL(str(Path(config["CYCLONEDDS_LIB"]) / "libddsc.so.0"), mode=ctypes.RTLD_GLOBAL)  # not the system one
    sys.path.insert(0, config["GEAR_ROOT"])
    from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    ChannelFactoryInitialize(config["DOMAIN_ID"], config["INTERFACE"])
    bridge = UnitreeSdk2Bridge(config)
    ex = Exchange((state_name, cmd_name))
    print(f"[dds_publisher] up: domain {config['DOMAIN_ID']} on {config['INTERFACE']}", flush=True)
    seq = 0.0
    while not ex.stopped() and os.getppid() == parent:
        new = pump(ex, bridge, seq)
        if new == seq:
            time.sleep(0.0002)  # nothing new: the scene steps every 5 ms
        seq = new
    ex.close()


if __name__ == "__main__":
    main(sys.argv)
