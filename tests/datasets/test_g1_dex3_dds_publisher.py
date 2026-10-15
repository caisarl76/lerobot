"""dds_publisher.py: the shared-memory exchange between the Isaac scene and its DDS publisher process (no DDS here:
a fake bridge stands in for gear_sonic's UnitreeSdk2Bridge)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
import dds_publisher as dp


def state(t):
    s = {name: np.full(n, float(i) + t) for i, (name, n) in enumerate(dp.STATE)}
    s["time"] = t
    return s


class FakeBridge:
    """The parts of UnitreeSdk2Bridge the publisher uses."""

    def __init__(self):
        self.published = []
        self.received = False

        def motors(n, base):
            return [SimpleNamespace(q=base + i, dq=0.1, kp=1.5, kd=0.2, tau=0.0) for i in range(n)]

        self.low_cmd = SimpleNamespace(motor_cmd=motors(35, 0.0))  # the hg LowCmd has 35 slots, 29 used
        self.left_hand_cmd = SimpleNamespace(motor_cmd=motors(7, 100.0))
        self.right_hand_cmd = SimpleNamespace(motor_cmd=motors(7, 200.0))

    def cmd_received(self):
        return self.received

    def PublishLowState(self, obs):  # noqa: N802 - gear_sonic's name
        self.published.append(obs)


def test_state_roundtrip_and_no_repeat():
    ex = dp.Exchange()
    try:
        assert ex.read_state(0) is None
        ex.write_state(state(1.5))
        seq, got = ex.read_state(0)
        for name, _ in dp.STATE:
            np.testing.assert_array_equal(np.atleast_1d(got[name]), np.atleast_1d(state(1.5)[name]))
        assert ex.read_state(seq) is None  # nothing new since
    finally:
        ex.close(unlink=True)


def test_attach_by_name_sees_the_same_memory():
    ex = dp.Exchange()
    try:
        other = dp.Exchange(ex.names)
        ex.write_state(state(2.0))
        assert other.read_state(0)[1]["time"] == 2.0
        other.set_stop()
        assert ex.stopped()
        other.close()
    finally:
        ex.close(unlink=True)


def test_pump_publishes_new_state_once_and_returns_commands_in_scene_order():
    ex, br = dp.Exchange(), FakeBridge()
    try:
        assert ex.read_cmd() is None
        ex.write_state(state(0.005))
        seq = dp.pump(ex, br, 0)
        assert len(br.published) == 1 and br.published[0]["time"] == 0.005
        assert br.published[0]["body_q"].shape == (dp.NB,)
        assert dp.pump(ex, br, seq) == seq and len(br.published) == 1  # no new state: no publish
        assert ex.read_cmd() is None  # the deploy has not sent anything yet
        br.received = True
        dp.pump(ex, br, seq)
        cmd = ex.read_cmd()
        np.testing.assert_array_equal(cmd["q"][: dp.NB], np.arange(dp.NB))  # body 29 of the 35 slots
        assert cmd["q"][dp.NB] == 100.0 and cmd["q"][dp.NB + dp.NH] == 200.0  # left hand, then right hand
        assert cmd["kp"][0] == 1.5 and cmd["kd"][-1] == 0.2
    finally:
        ex.close(unlink=True)


def test_make_obs_splits_hands_after_the_body():
    s = state(1.0)
    s["q"] = np.arange(dp.N, dtype=float)
    obs = dp.make_obs(s)
    assert obs["left_hand_q"][0] == dp.NB and obs["right_hand_q"][0] == dp.NB + dp.NH
    assert obs["floating_base_pose"].shape == (7,)
