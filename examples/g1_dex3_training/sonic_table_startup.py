"""Table-safe startup for tabletop evaluation: planner stance (arms down) -> fixed initial pose, hands never in the table.

NVIDIA's standing token (LATENT_INITIAL_MOTION_TOKEN) raises the hands to ~0.8 m, 0.28-0.36 m in front of the
pelvis: into an 80 cm table whose near edge is 5-12 cm from the torso. At the table the streamer skips it and
plays this precomputed arm trajectory instead (legs stay at SONIC's nominal standing):

  plan:  search arm waypoints (tuck behind the table edge, spread and raise at the sides, optionally raise over the top) so every arm
         collision geom stays >= --margin from the table and clear of the robot's own body along the joint-space
         path stance -> tuck -> side raise -> raise -> initial pose; smoothstep segments capped at --max-speed; encode to SONIC tokens
         with the dataset converter (mode 0, no slew limit). Writes startup .npz (30 Hz arms, hands, tokens).
  check: MuJoCo closed loop with the v1.1 decoder (offline harness, validated against NVIDIA's deploy) and a
         physical table: min reached clearance, table contacts, tilt, foot contacts, per table-edge distance.

Fixed initial pose = per-joint median of the Unitree joint28 episodes' first action. Stance = the arm pose the
official deploy reaches in planner mode (measured in sim, streamer logs 2026-09-24).
Usage (audit container, PYTHONPATH=/audit/pylib:/code, /gear_sonic mounted, /run-output read-only):
  python sonic_table_startup.py plan --out /audit/table_startup/startup.npz
  python sonic_table_startup.py check --startup /audit/table_startup/startup.npz
"""

from __future__ import annotations

import argparse
import glob
import heapq
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq
from sonic_roundtrip_audit import (
    CONVERTER_XML,
    DEFAULT,
    MODEL,
    SCENE,
    Decoder,
    Robot,
    generic_encoder_inputs,
    gravity,
    startup_hang,
)
from sonic_targets import SonicEncoder, build_encoder_inputs, load_joint_limits

FPS = 30
TORSO_FRONT_X = 0.08  # torso mesh max x, pelvis frame (G1 43dof scene)
TABLE_TOP_Z = 0.80
# Planner-mode arm pose in the official deploy (motor order, left 7 then right 7), measured in MuJoCo.
STANCE_ARMS = np.array(
    [-0.028, 0.296, -0.648, 0.985, -0.149, -0.117, 0.117, -0.033, -0.299, 0.634, 1.019, 0.3, -0.035, -0.269]
)


def with_table(robot: Robot, gap: float) -> int:
    """Add an 80 cm table (3 cm top, no apron) whose near edge is `gap` in front of the torso; return its geom id."""
    spec = mujoco.MjSpec.from_file(SCENE)
    edge = robot.d.xpos[robot.pelvis][0] + TORSO_FRONT_X + gap
    body = spec.worldbody.add_body(name="table", pos=[edge + 0.4, 0, TABLE_TOP_Z - 0.015])
    body.add_geom(
        name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.4, 0.8, 0.015], rgba=[0.6, 0.45, 0.3, 1]
    )
    qpos, qvel, ctrl = robot.d.qpos.copy(), robot.d.qvel.copy(), robot.d.ctrl.copy()
    robot.m = spec.compile()
    robot.m.opt.timestep = 0.005
    robot.d, robot.kin = mujoco.MjData(robot.m), mujoco.MjData(robot.m)
    robot.d.qpos[:], robot.d.qvel[:], robot.d.ctrl[:] = qpos, qvel, ctrl
    mujoco.mj_forward(robot.m, robot.d)
    return mujoco.mj_name2id(robot.m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")


def collision_geoms(m, keys):
    return [
        g
        for g in range(m.ngeom)
        if (m.geom_contype[g] or m.geom_conaffinity[g])
        and any(k in m.body(m.geom_bodyid[g]).name for k in keys)
    ]


class Clearance:
    """Kinematic clearance of one arm's collision geoms to the table and to the torso/pelvis/legs."""

    def __init__(self, robot: Robot, table: int, side: str):
        self.r, self.table = robot, table
        m = robot.m
        self.arm = collision_geoms(m, [f"{side}_{k}" for k in ("elbow", "wrist", "hand", "shoulder_yaw")])
        self.body = collision_geoms(m, ("torso", "pelvis", "hip", "knee"))
        self.base = robot.d.qpos.copy()
        self.col = slice(15, 22) if side == "left" else slice(22, 29)

    def __call__(self, arm7: np.ndarray, table_only: bool = False) -> tuple[float, float]:
        k = self.r.kin
        k.qpos[:] = self.base
        k.qpos[self.r.qadr[self.col]] = arm7
        mujoco.mj_kinematics(self.r.m, k)
        return self.distances(k, table_only)

    def distances(self, data, table_only: bool = False) -> tuple[float, float]:
        ft = np.zeros(6)
        table = min(mujoco.mj_geomDistance(self.r.m, data, g, self.table, 0.5, ft) for g in self.arm)
        if table_only:
            return table, 1.0
        body = min(mujoco.mj_geomDistance(self.r.m, data, g, b, 0.5, ft) for g in self.arm for b in self.body)
        return table, body


def segment(a: np.ndarray, b: np.ndarray, max_speed: float) -> np.ndarray:
    """Smoothstep from a to b at FPS; duration so the peak joint speed (1.5x mean) stays <= max_speed."""
    t = max(1.5 * float(np.abs(b - a).max()) / max_speed, 0.5)
    s = np.linspace(0, 1, int(np.ceil(t * FPS)) + 1)[1:, None]
    return a + (b - a) * (3 * s**2 - 2 * s**3)


def path_ok(clear, pts, margin, body_min, final_min, start_min=None) -> bool:
    """Every other sample (<= 1 cm hand travel) clears the table by margin (start_min over the first 10%, final_min
    over the last 10%) and the robot's own body by body_min."""
    for i in range(0, len(pts), 2):
        need = start_min if start_min is not None and i < 0.1 * len(pts) else margin
        need = final_min if i >= 0.9 * len(pts) else need
        table, _ = clear(pts[i], table_only=True)
        if table < need or clear(pts[i])[1] < body_min:
            return False
    return True


def plan_arm(clear, start, goal, margin, max_speed, transit_margin):
    """Shortest safe waypoint path for one arm through layers: tuck (hands back behind the edge) -> side raise (arms
    spread, hands still behind the edge) -> raise (optional, hands above the table) -> goal. Dijkstra over the
    layers with lazy segment checks; cost = sum of each segment's largest joint change (= duration at the cap).
    Returns 3 intermediate waypoints (a skipped raise repeats the side raise)."""
    sign = 1.0 if clear.col.start == 15 else -1.0  # roll/yaw mirror for the right arm
    final_table, _ = clear(goal)
    body_min = min(0.01, clear(start)[1])
    final_min = min(margin, final_table - 0.005)
    grid = itertools.product
    layers = [
        [start],
        [
            np.r_[sp, sign * r, 0.0, el, start[4:]]
            for sp, r, el in grid((0.0, 0.3, 0.6), (0.15, 0.35), (0.0, 0.3, 0.6))
        ],
        [
            np.r_[sp, sign * r, sign * yaw, el, start[4:]]
            for sp, r, yaw, el in grid(
                (-0.3, -0.15, 0.0, 0.15, 0.3),
                np.linspace(0.6, 1.8, 7),
                (-0.6, -0.3, 0.0, 0.3, 0.6),
                np.linspace(0, 1.8, 7),
            )
        ],
        [
            np.r_[sp, sign * r, sign * yaw, el, (start[4:] + goal[4:]) / 2]
            for sp, r, yaw, el in grid(
                np.linspace(-1.2, 0.6, 7),
                np.linspace(0.3, 1.8, 6),
                (-0.8, -0.4, 0.0, 0.4, 0.8),
                np.linspace(0, 2.0, 6),
            )
        ],
        [goal],
    ]
    # leaving the stance may not get closer than the stance already is (its wrists sit near the table's underside
    # edge) and must reach the margin by the end of that move
    first_margin = min(margin, clear(start)[0] - 0.005)

    def safe(la, a, lb, b):
        """Leaving the stance: no closer than the stance, margin by the end. Spread/raise transit: transit_margin
        in the middle (covers SONIC tracking drift), margin at the ends. Settling: margin, then the goal's own."""
        pts = segment(layers[la][a], layers[lb][b], max_speed)
        if la == 0:
            return path_ok(clear, pts, first_margin, body_min, margin)
        if lb == 4:
            return path_ok(clear, pts, margin, body_min, final_min)
        return path_ok(clear, pts, transit_margin, body_min, margin, start_min=margin)

    def nexts(layer):
        return (2,) if layer == 1 else (3, 4) if layer == 2 else (4,) if layer == 3 else (1,)

    heap, settled, checked = [(0.0, 0, 0, None)], {}, 0
    while heap:
        cost, layer, idx, parent = heapq.heappop(heap)
        if (layer, idx) in settled:
            continue
        if parent is not None:
            checked += 1
            if not safe(parent[0], parent[1], layer, idx):
                continue
        settled[layer, idx] = parent
        if layer == 4:
            break
        for nl in nexts(layer):
            here = layers[layer][idx]
            for k, w in enumerate(layers[nl]):
                if (nl, k) not in settled:
                    heapq.heappush(heap, (cost + float(np.abs(w - here).max()), nl, k, (layer, idx)))
    else:
        raise RuntimeError(f"no safe path found for arm at column {clear.col.start}")
    path, node = [], (4, 0)
    while node is not None:
        path.append(node)
        node = settled[node]
    path = path[::-1]  # (0,0) start ... (4,0) goal
    ws = [layers[la][k] for la, k in path[1:-1]]
    if len(ws) == 2:  # raise skipped
        ws.append(ws[1])
    print(f"[plan] column {clear.col.start}: cost {cost:.2f} rad, {checked} segments checked", flush=True)
    info = {
        "start_table_clearance_m": first_margin + 0.005,
        "final_table_clearance_m": final_table,
        "body_min_m": body_min,
        "path_cost_rad": cost,
        "raise_skipped": len(path) == 4,
    }
    return tuple(ws), info


def initial_pose(joint28_root: str) -> tuple[np.ndarray, int]:
    first = []
    for f in sorted(glob.glob(f"{joint28_root}/data/*/*.parquet")):
        t = pq.read_table(f, columns=["frame_index", "action"]).to_pandas()
        first += [np.asarray(a) for a in t[t.frame_index == 0]["action"]]
    return np.median(np.stack(first), 0), len(first)


def plan(a):
    goal28, n_ep = initial_pose(a.joint28)
    robot = Robot()
    robot.reset(DEFAULT)
    table = with_table(robot, a.gap)
    waypoints, info = {}, {}
    for side, cols in (("left", slice(0, 7)), ("right", slice(7, 14))):
        clear = Clearance(robot, table, side)
        mids, info[side] = plan_arm(
            clear, STANCE_ARMS[cols], goal28[cols], a.margin, a.max_speed, a.transit_margin
        )
        waypoints[side] = (STANCE_ARMS[cols], *mids, goal28[cols])
    # both arms share segment timing: each segment lasts as long as the slower arm needs
    arms = [STANCE_ARMS[None]]
    names = ("tuck: hands back behind the edge", "spread: arms up at the sides", "raise: over the table top",
             "settle: into the initial pose")  # fmt: skip
    segments, frame = [], 1
    for k in range(4):
        a0 = np.r_[waypoints["left"][k], waypoints["right"][k]]
        a1 = np.r_[waypoints["left"][k + 1], waypoints["right"][k + 1]]
        if np.abs(a1 - a0).max() > 1e-9:  # a skipped raise on both arms adds no segment
            arms.append(segment(a0, a1, a.max_speed))
            segments.append({"name": names[k], "start_s": frame / FPS, "duration_s": len(arms[-1]) / FPS,
                             "max_joint_change_rad": round(float(np.abs(a1 - a0).max()), 3)})  # fmt: skip
            frame += len(arms[-1])
    arms.append(np.repeat(goal28[None, :14], int(a.hold_s * FPS), 0))
    segments.append({"name": "hold at the initial pose", "start_s": frame / FPS, "duration_s": a.hold_s})
    arms = np.concatenate(arms)
    hands = np.linspace(np.zeros(14), goal28[14:], len(arms))
    limits = load_joint_limits(Path(a.robot_xml))
    inputs, enc_hands, report = build_encoder_inputs(
        np.c_[arms, hands], limits, arm_speed_limit=None, hand_speed_limit=None, fps=FPS
    )
    tokens = SonicEncoder(MODEL / "model_encoder.onnx", MODEL / "observation_config.yaml").encode(inputs)
    speed = np.abs(np.diff(arms, axis=0)).max() * FPS
    meta = {
        "fps": FPS,
        "duration_s": len(arms) / FPS,
        "max_joint_speed_rad_s": float(speed),
        "initial_pose": f"median first action over {n_ep} Unitree joint28 episodes",
        "table": {
            "top_z_m": TABLE_TOP_Z,
            "planned_gap_from_torso_m": a.gap,
            "margin_m": a.margin,
            "transit_margin_m": a.transit_margin,
        },
        "segments": segments,
        "waypoints": {s: [np.round(w, 3).tolist() for w in ws] for s, ws in waypoints.items()},
        "per_arm": info,
        "encoder": report,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, arms=arms, hands=enc_hands, tokens=tokens, meta=json.dumps(meta))
    print(json.dumps(meta, indent=1, default=float))


def check(a):
    z = np.load(a.startup)
    meta = json.loads(str(z["meta"]))
    enc = SonicEncoder(MODEL / "model_encoder.onnx", MODEL / "observation_config.yaml")
    results = {}
    for gap in a.gaps:
        robot, dec = Robot(), Decoder()
        stand = enc.encode(generic_encoder_inputs(DEFAULT[None], 50, robot.limits)[0])[0]
        body0 = DEFAULT.copy()
        body0[15:29] = STANCE_ARMS
        stance = enc.encode(generic_encoder_inputs(body0[None], 50, robot.limits)[0])[0]
        info = startup_hang(robot, dec, stand)
        for i in range(100):  # 2 s token blend to the planner-stance arms (stands in for the planner handoff)
            w = (i + 1) / 100
            robot.step(dec((1 - w) * stand + w * stance, *robot.state()))
        for _ in range(50):
            robot.step(dec(stance, *robot.state()))
        table = with_table(robot, gap)
        clears = {s: Clearance(robot, table, s) for s in ("left", "right")}
        dense = np.arange(0, len(z["tokens"]) - 1, FPS / 50)  # 50 Hz ticks, look-ahead linear interpolation
        min_clear, contacts, tilt, feet = 1.0, 0, 0.0, 8
        rec = {"t": [], "qpos": [], "clear": []}
        for t in dense:
            i = int(t)
            tok = z["tokens"][i] + (t - i) * (z["tokens"][i + 1] - z["tokens"][i])
            robot.step(dec(tok, *robot.state()))
            min_clear = min(min_clear, *(c.distances(robot.d, table_only=True)[0] for c in clears.values()))
            contacts += any(
                table in (robot.d.contact[j].geom1, robot.d.contact[j].geom2) for j in range(robot.d.ncon)
            )
            tilt = max(tilt, float(np.degrees(np.arccos(np.clip(-gravity(robot.state()[2])[2], -1, 1)))))
            feet = min(feet, robot.floor_contacts())
            rec["t"].append(t / FPS)
            rec["qpos"].append(robot.d.qpos.copy())
            rec["clear"].append(min(c.distances(robot.d, table_only=True)[0] for c in clears.values()))
        if a.record:
            edge = float(robot.m.body("table").pos[0] - 0.4)
            np.savez(f"{a.record}/gap_{gap * 100:.0f}cm.npz", **{k: np.asarray(v) for k, v in rec.items()},
                     table_edge_x=edge, gap=gap, segments=json.dumps(meta.get("segments", [])))  # fmt: skip
        reached = robot.state()[0][15:29]
        results[f"gap_{gap * 100:.0f}cm"] = {
            "min_reached_table_clearance_cm": round(100 * min_clear, 1),
            "table_contact_ticks": contacts,
            "tilt_max_deg": round(tilt, 2),
            "floor_contacts_min": feet,
            "final_arm_error_rad_max": round(float(np.abs(reached - z["arms"][-1]).max()), 3),
            "settle": {k: v for k, v in info.items() if k in ("fell", "settle_tilt_max_deg")},
        }
    print(json.dumps({"startup_duration_s": meta["duration_s"], "results": results}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("plan")
    s.add_argument("--out", required=True)
    s.add_argument("--joint28", default="/run-output/datasets/joint28")
    s.add_argument("--robot-xml", default=str(CONVERTER_XML))
    s.add_argument(
        "--gap", type=float, default=0.05, help="table edge distance from the torso front (worst case)"
    )
    s.add_argument("--margin", type=float, default=0.03)
    s.add_argument(
        "--transit-margin", type=float, default=0.05, help="mid spread/raise clearance (tracking drift)"
    )
    s.add_argument("--max-speed", type=float, default=0.5)
    s.add_argument("--hold-s", type=float, default=1.0)
    s = sub.add_parser("check")
    s.add_argument("--startup", required=True)
    s.add_argument("--gaps", type=float, nargs="+", default=[0.05, 0.085, 0.12])
    s.add_argument("--record", help="directory: save reached qpos per 50 Hz tick for rendering")
    a = p.parse_args()
    plan(a) if a.cmd == "plan" else check(a)
