"""derive_handover_task.py: grasp / release frames and the bottle position in front of the palm."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
import derive_handover_task as dt


def test_events_right_hand_close_then_reopen():
    lo, hi = np.zeros(14), np.ones(14)
    h = np.zeros((100, 14))
    h[30:70, 7:] = 1.0  # right hand closes at 30, reopens at 70
    h[50:60, :7] = 1.0  # left closes later
    assert dt.events(h, lo, hi) == (1, 30, 70)


def test_bottle_from_palm_offsets_along_palm_normal():
    r = np.eye(3)
    p = dt.bottle_from_palm(np.array([0.4, -0.2, 0.9]), r, 0.05, hand=1)
    assert np.isclose(np.linalg.norm(p - [0.4, -0.2, 0.9]), 0.05)


def test_episode_rows_keep_only_the_streamed_episode():
    # the deploy holds the Dex3 hands closed in planner mode: grasp events count only inside the episode phase
    sim_wall = np.arange(10.0)
    phase = np.array(["planner token", "episode", "episode", "episode", "blend out"])
    wall = np.array([1.0, 3.0, 4.0, 6.0, 7.0])
    assert dt.episode_rows(sim_wall, wall, phase).tolist() == [3, 4, 5, 6]


def test_events_ignore_a_hand_still_closed_from_the_planner_at_the_start():
    lo, hi = np.zeros(14), np.ones(14)
    h = np.zeros((100, 14))
    h[:10, :7] = 1.0  # left still closed from planner mode, opens at 10
    h[:10, 7:] = 1.0
    h[40:80, 7:] = 1.0  # right grasps at 40, releases at 80
    assert dt.events(h, lo, hi) == (1, 40, 80)


def test_events_take_the_hand_that_closes_and_reopens():
    # a handover releases the bottle: a hand clenched for the rest of the episode (idle left fist) is not the grasp
    lo, hi = np.zeros(14), np.ones(14)
    h = np.zeros((100, 14))
    h[5:, :7] = 1.0
    h[40:80, 7:] = 1.0
    assert dt.events(h, lo, hi) == (1, 40, 80)


def test_events_skip_a_flicker_through_the_threshold():
    lo, hi = np.zeros(14), np.ones(14)
    h = np.zeros((200, 14))
    h[5:11, :7] = 1.0  # left crosses 50 % for 6 records (0.12 s at 50 Hz) as it leaves the planner fist
    h[100:150, 7:] = 1.0
    assert dt.events(h, lo, hi) == (1, 100, 150)
