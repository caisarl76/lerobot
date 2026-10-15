"""Automatic verdict for handover_bottle trials in simulation (spec: Judge rules). Pure numpy.

Fed once per physics step. early() ends a trial at once (success, knocked over before a lift, dropped on the floor,
fall); the first early verdict is latched. final(episode_end) classifies the trial when the streamer reports its
end (ended.i). Reasons are 1-based indices into eval_session.DEFAULT_FAILURE_REASONS; final() returns None for redo.
"""

from __future__ import annotations

import numpy as np

try:
    from .eval_session import DEFAULT_FAILURE_REASONS
except ImportError:
    from eval_session import DEFAULT_FAILURE_REASONS

DEFAULTS = {
    "lift_m": 0.10,  # bottle bottom above the table top that counts as lifted
    "hold_s": 1.0,  # time in the zone for success
    "reach_margin_m": 0.04,  # palm-to-axis distance <= radius + margin counts as reached / carrying
    "down_m": 0.02,  # bottom within this of the table top after a lift: back on the table
    "floor_z_m": 0.10,  # bottle centre below this: on the floor
    "tip_deg": 60.0,  # bottle axis tilt that counts as knocked over (before a lift)
    "table_half_xy": [0.4, 0.8],  # table top half sizes (sim_scene table)
}
REDO_PREFIXES = ("no valid chunk", "robot state")


def reason(name: str) -> int:
    return DEFAULT_FAILURE_REASONS.index(name) + 1


def _rot(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


class Judge:
    def __init__(self, cfg: dict):
        self.c = {**DEFAULTS, **cfg}
        self.reached = self.lifted = False
        self.zone_since: float | None = None
        self.latched: dict | None = None
        self.table_xy: np.ndarray | None = None  # table centre xy, set by the host when the table is placed
        self.last: dict = {}

    def _axis_distance(self, palm, bottle, quat) -> float:
        axis = _rot(quat)[:, 2]
        h = self.c["bottle_height"] / 2
        s = np.clip(np.dot(palm - bottle, axis), -h, h)
        return float(np.linalg.norm(palm - (bottle + s * axis)))

    def feed(self, t, bottle_pos, bottle_quat_wxyz, palms, pelvis_pos, pelvis_quat_wxyz, fallen) -> None:
        if self.latched is not None:
            return
        c = self.c
        bottle, quat = np.asarray(bottle_pos, float), np.asarray(bottle_quat_wxyz, float)
        palms = np.asarray(palms, float)
        bottom = bottle[2] - c["bottle_height"] / 2
        reach = c["bottle_radius"] + c["reach_margin_m"]
        near = [self._axis_distance(p, bottle, quat) <= reach for p in palms]
        self.reached |= any(near)
        lifted_now = bottom >= c["table_top_z"] + c["lift_m"]
        self.lifted |= lifted_now
        tilt = np.degrees(np.arccos(np.clip(_rot(quat)[2, 2], -1, 1)))
        self.last = {"bottom": bottom, "lifted_now": lifted_now}
        if fallen:
            self.latched = {"verdict": "fail", "reason": reason("other"), "note": "fell"}
            return
        if bottle[2] < c["floor_z_m"]:
            name = "dropped the object" if self.lifted else "collision (table, knocked over)"
            self.latched = {"verdict": "fail", "reason": reason(name), "note": "on the floor"}
            return
        off_table = self.table_xy is not None and np.any(
            np.abs(bottle[:2] - self.table_xy) > np.asarray(c["table_half_xy"])
        )
        if not self.lifted and (tilt > c["tip_deg"] or off_table):
            self.latched = {"verdict": "fail", "reason": reason("collision (table, knocked over)"), "note": ""}
            return
        palm = palms[c["grasp_hand"]]
        zone = np.asarray(pelvis_pos, float) + _rot(pelvis_quat_wxyz) @ np.asarray(c["zone_center_pelvis"], float)
        in_zone = lifted_now and near[c["grasp_hand"]] and np.linalg.norm(palm - zone) <= c["zone_radius"]
        if not in_zone:
            self.zone_since = None
        elif self.zone_since is None:
            self.zone_since = t
        elif t - self.zone_since >= c["hold_s"] - 1e-9:
            self.latched = {"verdict": "success"}

    def early(self) -> dict | None:
        return self.latched

    def final(self, episode_end: str) -> dict | None:
        if self.latched is not None:
            return self.latched
        if episode_end.startswith("watchdog"):
            return {"verdict": "fail", "reason": reason("safety stop (watchdog)"), "note": episode_end}
        if episode_end.startswith(REDO_PREFIXES):
            return None
        if not self.reached:
            return {"verdict": "fail", "reason": reason("did not reach the object"), "note": ""}
        if not self.lifted:
            return {"verdict": "fail", "reason": reason("grasp failed / missed"), "note": ""}
        if self.last["bottom"] <= self.c["table_top_z"] + self.c["down_m"]:
            return {"verdict": "fail", "reason": reason("dropped the object"), "note": "back on the table"}
        if self.last["lifted_now"]:
            return {"verdict": "fail", "reason": reason("wrong object / target"), "note": "not in handover zone"}
        return {"verdict": "fail", "reason": reason("stalled / froze"), "note": "lifted, never held"}

    @staticmethod
    def to_gate(v: dict | None) -> dict:
        """verdict.i JSON for eval_session.gate_verdict (None = redo)."""
        if v is None:
            return {"verdict": "redo", "reason": "", "note": ""}
        if v["verdict"] == "success":
            return {"verdict": "success", "reason": "", "note": ""}
        return {"verdict": "fail", "reason": str(v["reason"]), "note": v.get("note", "")}
