"""handover_success.Judge on synthetic trajectories (spec test 1)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
import eval_session as es
import handover_success as hs

CFG = {**hs.DEFAULTS, "table_top_z": 0.80, "bottle_radius": 0.033, "bottle_height": 0.20,
       "zone_center_pelvis": [0.35, -0.20, 0.25], "zone_radius": 0.08, "grasp_hand": 1}  # fmt: skip
UP = np.array([1.0, 0.0, 0.0, 0.0])
PELVIS = np.array([0.0, 0.0, 0.79])
BOTTLE0 = np.array([0.45, -0.15, 0.80 + 0.10])  # centre; bottom on the table
FAR = np.array([[0.2, 0.3, 0.9], [0.2, -0.3, 0.9]])


def reason(name):
    return es.DEFAULT_FAILURE_REASONS.index(name) + 1


def run(frames, episode_end="completed"):
    j = hs.Judge(CFG)
    for t, (bottle, palms, quat, fallen) in enumerate(frames):
        j.feed(t * 0.02, bottle, quat, palms, PELVIS, UP, fallen)
        if (v := j.early()) is not None:
            return v
    return j.final(episode_end)


def palms_at(right):
    return np.array([[0.2, 0.3, 0.9], right])


def zone_world():
    return PELVIS + np.array(CFG["zone_center_pelvis"])


def lift_and_carry(n_hold):
    """reach (palm 4 cm from the axis), lift 15 cm, carry the bottle + palm into the zone, hold n_hold frames."""
    out = []
    grip = BOTTLE0 + [0.0, -0.04, 0.0]
    out += [(BOTTLE0, palms_at(grip), UP, None)] * 5
    lifted = BOTTLE0 + [0, 0, 0.15]
    out += [(lifted, palms_at(lifted + [0, -0.04, 0]), UP, None)] * 5
    zp = zone_world()
    out += [(zp + [0, 0.04, 0], palms_at(zp), UP, None)] * n_hold
    return out


def test_success_after_one_second_in_zone():
    assert run(lift_and_carry(51)) == {"verdict": "success"}


def test_success_is_latched_even_if_dropped_after():
    frames = lift_and_carry(51) + [(np.array([0.5, 0.0, 0.03]), FAR, UP, None)] * 5
    assert run(frames) == {"verdict": "success"}


def test_palm_in_zone_but_bottle_left_on_table_is_not_success():
    zp = zone_world()
    frames = [(BOTTLE0, palms_at(zp), UP, None)] * 80
    assert run(frames)["reason"] == reason("did not reach the object")


def test_never_near_is_reason_1():
    assert run([(BOTTLE0, FAR, UP, None)] * 20)["reason"] == reason("did not reach the object")


def test_reached_never_lifted_is_reason_2():
    frames = [(BOTTLE0, palms_at(BOTTLE0 + [0, -0.04, 0]), UP, None)] * 20
    assert run(frames)["reason"] == reason("grasp failed / missed")


def test_lift_then_floor_is_early_reason_3():
    frames = lift_and_carry(5) + [(np.array([0.5, 0.0, 0.03]), FAR, UP, None)]
    v = run(frames, episode_end="should not be used")
    assert v["reason"] == reason("dropped the object")


def test_lift_then_back_on_table_is_final_reason_3():
    frames = lift_and_carry(5) + [(BOTTLE0 + [0, 0, 0.01], FAR, UP, None)] * 3
    assert run(frames)["reason"] == reason("dropped the object")


def test_tipped_before_lift_is_reason_5():
    tipped = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])  # 90 deg about x
    assert run([(BOTTLE0, FAR, tipped, None)])["reason"] == reason("collision (table, knocked over)")


def test_off_the_table_before_lift_is_reason_5():
    assert run([(np.array([0.5, 0.0, 0.03]), FAR, UP, None)])["reason"] == reason("collision (table, knocked over)")


def test_lifted_held_outside_zone_is_reason_4():
    lifted = BOTTLE0 + [0, 0, 0.15]
    frames = [(lifted, palms_at(lifted + [0, -0.04, 0]), UP, None)] * 80
    v = run(frames)
    assert v["reason"] == reason("wrong object / target") and "zone" in v["note"]


def test_lifted_then_dipped_is_reason_6():
    lifted, dipped = BOTTLE0 + [0, 0, 0.15], BOTTLE0 + [0, 0, 0.05]
    frames = [(lifted, palms_at(lifted), UP, None)] * 3 + [(dipped, palms_at(dipped), UP, None)] * 3
    assert run(frames)["reason"] == reason("stalled / froze")


def test_watchdog_and_errors():
    frames = [(BOTTLE0, FAR, UP, None)] * 3
    assert run(frames, "watchdog: arm speed")["reason"] == reason("safety stop (watchdog)")
    assert run(frames, "no valid chunk for 2 s") is None
    assert run(frames, "robot state 0.50 s old") is None


def test_fall_is_other_with_note():
    v = run([(BOTTLE0, FAR, UP, "tilt 40 deg")])
    assert v["reason"] == reason("other") and v["note"] == "fell"


def test_to_gate():
    assert hs.Judge.to_gate({"verdict": "success"}) == {"verdict": "success", "reason": "", "note": ""}
    assert hs.Judge.to_gate(None) == {"verdict": "redo", "reason": "", "note": ""}
    assert hs.Judge.to_gate({"verdict": "fail", "reason": 3, "note": "x"}) == {"verdict": "fail", "reason": "3", "note": "x"}
