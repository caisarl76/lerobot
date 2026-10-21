"""Copy a sonic78 dataset with the 46-D state of NVIDIA's official UNITREE_G1_SONIC GR00T config.

State layout (= stream_backends.sonic46_state on feat/g1-groot-official46, what the robot streamer sends):
  [0:12] legs left 6 + right 6, [12:15] waist yaw/roll/pitch, [15:29] left + right arm 7 (motor order),
  [29:36] left hand, [36:43] right hand, both index 0/1, middle 0/1, thumb 0/1/2 (= Dex3 hand_q[[3,4,5,6,0,1,2]]),
  [43:46] projected gravity (world [0, 0, -1] in the pelvis frame).
Episodes, frames, action, tasks and videos (symlinked) are unchanged; observation.state statistics are recomputed
(exact quantiles), and meta/state46_provenance.json records the sources and checks.

  handover: state from the raw GR00T-WBC recording through <src>.segments.json: its 43-D observation.state by joint
            name, gravity = the recorded observation.projected_gravity. Checked against the 28-D state by name.
  he:       arms from the sonicstate dataset (SONIC-relabelled, kept), legs + waist and hands from raw HE
            (observation.leg_joints, observation.hand_joints); the raw hands must equal the 28-D hands under the HE
            converter's reorder. HE has no IMU: gravity from forward kinematics of the legs with both feet flat.

Usage (lerobot image, CPU only):
  python prepare_state46.py handover --src DS --raw RAW --out OUT
  python prepare_state46.py he --src DS --raw HE_SNAPSHOT --xml g1_29dof_with_hand.xml --out OUT [--workers 32]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import xml.etree.ElementTree as ET  # nosec B405 - parses the local robot XML only
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SIDES = ("left", "right")
LEG_JOINTS = [
    f"{s}_{j}_joint"
    for s in SIDES
    for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
]
ARM_JOINTS = [
    f"{s}_{j}_joint"
    for s in SIDES
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
HAND_JOINTS = [
    f"{s}_hand_{j}_joint"
    for s in SIDES
    for j in ("index_0", "index_1", "middle_0", "middle_1", "thumb_0", "thumb_1", "thumb_2")
]
STATE46_NAMES = (
    LEG_JOINTS
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + ARM_JOINTS
    + HAND_JOINTS
    + ["gravity_x", "gravity_y", "gravity_z"]
)
HAND_MODEL_FROM_RAW = np.array([3, 4, 5, 6, 0, 1, 2])  # Dex3 hand_q slots -> HAND_JOINTS order, both hands
# The HE converter (joint28 meta/provenance.json action_mapping) swapped the right hand's last two pairs; an involution.
HE_RIGHT_SWAP = np.array([0, 1, 2, 5, 6, 3, 4])
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def stats(x: np.ndarray) -> dict[str, list]:
    x = x.astype(np.float64)
    out = {"min": x.min(0), "max": x.max(0), "mean": x.mean(0), "std": x.std(0), "count": np.array([len(x)])}
    out.update({k: np.quantile(x, q, axis=0) for k, q in QUANTILES.items()})
    return {k: v.tolist() for k, v in out.items()}


# ---------------------------------------------------------------- gravity from leg forward kinematics
def _quat_mat(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _axis_rot(axis: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rodrigues, [N,3,3] for angles q [N] about a unit axis."""
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    s, c = np.sin(q)[:, None, None], np.cos(q)[:, None, None]
    return np.eye(3) + s * k + (1 - c) * (k @ k)


def leg_chains(xml: Path) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Per foot: (body rotation, joint axis) from the pelvis to <side>_ankle_roll_link, in LEG_JOINTS order."""
    root = ET.parse(xml).getroot()  # nosec B314 - trusted local robot model file
    chains = []
    for side in SIDES:
        chain, names = [], LEG_JOINTS[:6] if side == "left" else LEG_JOINTS[6:]
        for name in names:
            body = next(
                b for b in root.iter("body") if any(j.get("name") == name for j in b.findall("joint"))
            )
            joint = next(j for j in body.findall("joint") if j.get("name") == name)
            quat = [float(v) for v in body.get("quat", "1 0 0 0").split()]
            axis = np.array([float(v) for v in joint.get("axis", "0 0 1").split()])
            chain.append((_quat_mat(*np.divide(quat, np.linalg.norm(quat))), axis / np.linalg.norm(axis)))
        if body.get("name") != f"{side}_ankle_roll_link":
            raise ValueError(f"{side} chain ends at {body.get('name')}")
        chains.append(chain)
    return chains


def feet_flat_gravity(chains, legs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gravity in the pelvis frame [N,3] if both soles are level (mean of the two feet), and the two feet's
    disagreement in degrees [N]."""
    per_foot = []
    for chain, q in zip(chains, (legs[:, :6], legs[:, 6:12]), strict=True):
        rot = np.broadcast_to(np.eye(3), (len(legs), 3, 3))
        for i, (body_rot, axis) in enumerate(chain):
            rot = rot @ body_rot @ _axis_rot(axis, q[:, i])
        per_foot.append(-rot[:, :, 2])  # foot z = world up; gravity = -z expressed in the pelvis frame
    g = per_foot[0] + per_foot[1]
    cos = np.clip(np.sum(per_foot[0] * per_foot[1], axis=1), -1, 1)
    return g / np.linalg.norm(g, axis=1, keepdims=True), np.degrees(np.arccos(cos))


# ---------------------------------------------------------------- writing the dataset copy
def _episode_index(src: Path) -> dict[int, tuple[int, int]]:
    """episode -> (dataset_from_index, dataset_to_index)."""
    out = {}
    for f in sorted((src / "meta/episodes").rglob("*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "dataset_from_index", "dataset_to_index"]).to_pydict()
        out.update(
            zip(
                t["episode_index"],
                zip(t["dataset_from_index"], t["dataset_to_index"], strict=True),
                strict=True,
            )
        )
    return out


def _state_schema(table: pa.Table, values: np.ndarray) -> pa.Table:
    column = pa.FixedSizeListArray.from_arrays(pa.array(values.ravel(), type=pa.float32()), values.shape[1])
    table = table.set_column(table.schema.get_field_index("observation.state"), "observation.state", column)
    metadata = dict(table.schema.metadata or {})
    if b"huggingface" in metadata:
        hf = json.loads(metadata[b"huggingface"])
        hf["info"]["features"]["observation.state"].update(length=values.shape[1])
        metadata[b"huggingface"] = json.dumps(hf).encode()
    return table.replace_schema_metadata(metadata)


def write(src: Path, out: Path, states: dict[int, np.ndarray], provenance: dict) -> None:
    """Copy src to out with observation.state replaced by states[episode] (indexed by frame_index)."""
    tmp = out.with_name(out.name + ".incomplete")
    if out.exists() or tmp.exists():
        raise FileExistsError(f"{out} or {tmp} already exists")
    tmp.mkdir(parents=True)
    shutil.copytree(src / "meta", tmp / "meta")
    for p in (tmp / "meta").rglob("*"):
        if p.is_file():
            p.chmod(p.stat().st_mode | 0o200)
    videos = src / "videos"
    target = os.readlink(videos) if videos.is_symlink() else os.path.join("..", src.name, "videos")
    (tmp / "videos").symlink_to(target, target_is_directory=True)
    if set(states) != set(_episode_index(src)):
        raise ValueError("episode sets differ")
    for f in sorted((src / "data").rglob("*.parquet")):
        table = pq.read_table(f)
        eps, frames = table["episode_index"].to_numpy(), table["frame_index"].to_numpy()
        state = np.empty((table.num_rows, 46), np.float32)
        for ep in np.unique(eps):
            m = eps == ep
            if frames[m].max() >= len(states[ep]) or len(states[ep]) != m.sum():
                raise ValueError(f"episode {ep}: {len(states[ep])} states vs {m.sum()} rows")
            state[m] = states[ep][frames[m]]
        dest = tmp / f.relative_to(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(_state_schema(table, state), dest, compression="zstd")
    for f in sorted((tmp / "meta/episodes").rglob("*.parquet")):
        t = pq.read_table(f)
        ids = t["episode_index"].to_pylist()
        ep_stats = {e: stats(states[e]) for e in ids}
        for key in ep_stats[ids[0]]:
            name = f"stats/observation.state/{key}"
            if name in t.column_names:
                t = t.set_column(
                    t.schema.get_field_index(name), name, pa.array([ep_stats[e][key] for e in ids])
                )
        pq.write_table(t, f)
    all_states = np.concatenate([states[e] for e in sorted(states)])
    meta = tmp / "meta"
    st = json.loads((meta / "stats.json").read_text())
    st["observation.state"] = stats(all_states)
    (meta / "stats.json").write_text(json.dumps(st, indent=2))
    info = json.loads((meta / "info.json").read_text())
    info["features"]["observation.state"].update(shape=[46], names=STATE46_NAMES)
    (meta / "info.json").write_text(json.dumps(info, indent=4))
    provenance.update(
        source_dataset=str(src), episodes=len(states), frames=len(all_states), names=STATE46_NAMES
    )
    (meta / "state46_provenance.json").write_text(json.dumps(provenance, indent=2))
    tmp.rename(out)


# ---------------------------------------------------------------- sources
def _v3_states(src: Path, names: list[str]) -> dict[int, np.ndarray]:
    """The 28-D observation.state per episode (frame order), columns selected by name."""
    all_names = json.loads((src / "meta/info.json").read_text())["features"]["observation.state"]["names"]
    idx = [all_names.index(n) for n in names]
    parts: dict[int, list] = {}
    for f in sorted((src / "data").rglob("*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"])
        eps, frames = t["episode_index"].to_numpy(), t["frame_index"].to_numpy()
        state = np.stack(t["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float32)
        for ep in np.unique(eps):
            m = eps == ep
            parts.setdefault(int(ep), []).append((frames[m], state[m][:, idx]))
    out = {}
    for ep, p in parts.items():
        frames = np.concatenate([f for f, _ in p])
        out[ep] = np.concatenate([s for _, s in p])[np.argsort(frames, kind="stable")]
        if not np.array_equal(np.sort(frames), np.arange(len(frames))):
            raise ValueError(f"episode {ep}: frames are not 0..n-1")
    return out


def handover(a) -> None:
    src, raw = Path(a.src), Path(a.raw)
    segments = json.loads(Path(f"{src}.segments.json").read_text())
    info = json.loads((raw / "meta/info.json").read_text())
    raw_names = info["features"]["observation.state"]["names"]
    idx = [raw_names.index(n) for n in STATE46_NAMES[:43]]
    names28 = json.loads((src / "meta/info.json").read_text())["features"]["observation.state"]["names"]
    state28 = _v3_states(src, names28)
    states, check = {}, 0.0
    for ep, seg in enumerate(segments):
        if Path(seg["source"]).resolve() != raw.resolve():
            raise ValueError(f"episode {ep} comes from {seg['source']}")
        f = raw / info["data_path"].format(episode_chunk=seg["episode"] // 1000, episode_index=seg["episode"])
        t = pq.read_table(f, columns=["observation.state", "observation.projected_gravity"])
        s = np.stack(t["observation.state"].to_numpy(zero_copy_only=False))[
            seg["start_frame"] : seg["end_frame"]
        ]
        g = np.stack(t["observation.projected_gravity"].to_numpy(zero_copy_only=False))
        states[ep] = np.concatenate([s[:, idx], g[seg["start_frame"] : seg["end_frame"]]], axis=1).astype(
            np.float32
        )
        by_name = states[ep][:, [STATE46_NAMES.index(n) for n in names28]]
        check = max(check, float(np.abs(by_name - state28[ep]).max()))
    if check > 0:
        raise ValueError(f"46-D state differs from the 28-D state by name: max {check}")
    write(
        src,
        Path(a.out),
        states,
        {
            "raw_source": str(raw),
            "state": "raw observation.state (43, by joint name) + observation.projected_gravity (recorded, IMU)",
            "check_vs_28d_by_name_max_abs": check,
        },
    )


def _read_raw(f: str) -> tuple[int, np.ndarray, np.ndarray]:
    t = pq.read_table(
        f, columns=["episode_index", "frame_index", "observation.leg_joints", "observation.hand_joints"]
    )
    ep = t["episode_index"].to_numpy()
    if len(set(ep)) != 1 or not np.array_equal(t["frame_index"].to_numpy(), np.arange(t.num_rows)):
        raise ValueError(f"{f}: not one episode with frames 0..n-1")
    legs = np.asarray(t["observation.leg_joints"].to_pylist(), np.float32)
    hands = np.asarray(t["observation.hand_joints"].to_pylist(), np.float32)
    return int(ep[0]), legs, hands


def he(a) -> None:
    src, raw, xml = Path(a.src), Path(a.raw), Path(a.xml)
    names28 = json.loads((src / "meta/info.json").read_text())["features"]["observation.state"]["names"]
    state28 = _v3_states(src, names28)
    # joint28 order (Unitree names in the HE copy): arms 14 in motor order, then left and right hand
    if names28[0] not in ("kLeftShoulderPitch", ARM_JOINTS[0]) or names28[13] not in (
        "kRightWristYaw",
        ARM_JOINTS[13],
    ):
        raise ValueError(f"unexpected 28-D names {names28}")
    files = sorted(str(p) for p in (raw / "data").rglob("*.parquet"))
    with Pool(a.workers) as pool:
        raw_eps = {
            ep: (legs, hands) for ep, legs, hands in pool.imap_unordered(_read_raw, files, chunksize=8)
        }
    if set(raw_eps) != set(state28):
        raise ValueError(f"{len(raw_eps)} raw episodes vs {len(state28)}")
    chains = leg_chains(xml)
    states, disagree = {}, []
    for ep, s28 in state28.items():
        legs, hands = raw_eps[ep]
        if len(legs) != len(s28) or legs.shape[1] != 15 or hands.shape[1] != 14:
            raise ValueError(f"episode {ep}: raw {legs.shape} {hands.shape} vs {s28.shape}")
        # frame alignment and hand order: the 28-D hands are the raw hands, the right hand reordered
        if not (
            np.array_equal(s28[:, 14:21], hands[:, :7])
            and np.array_equal(s28[:, 21:28], hands[:, 7:][:, HE_RIGHT_SWAP])
        ):
            raise ValueError(f"episode {ep}: 28-D hands are not the raw hands")
        g, dis = feet_flat_gravity(chains, legs[:, :12].astype(np.float64))
        disagree.append(dis)
        states[ep] = np.concatenate(
            [
                legs,
                s28[:, :14],
                hands[:, :7][:, HAND_MODEL_FROM_RAW],
                hands[:, 7:][:, HAND_MODEL_FROM_RAW],
                g,
            ],
            axis=1,
        ).astype(np.float32)
    disagree = np.concatenate(disagree)
    tilt = np.degrees(np.arccos(-np.concatenate([s[:, 45] for s in states.values()])))
    write(
        src,
        Path(a.out),
        states,
        {
            "raw_source": str(raw),
            "state": "legs + waist: raw observation.leg_joints (motor order 0..14); arms: source 28-D state (SONIC-"
            "relabelled); hands: raw observation.hand_joints, Dex3 slots [3,4,5,6,0,1,2] per hand (equal to the 28-D "
            "hands, whose right hand is raw slots [0,1,2,5,6,3,4], checked exactly on every frame)",
            "gravity": "no IMU in HE: world [0,0,-1] in the pelvis frame from forward kinematics of the 12 leg joints "
            "with both soles level, mean of the two feet. On the handover recording (real IMU, SONIC stance) this "
            "method is off by p50 0.78 / p95 1.24 / max 2.7 deg (constant [0,0,-1]: 3.5 / 7.6 / 10.3 deg).",
            "fk_xml": str(xml),
            "fk_xml_sha256": hashlib.sha256(xml.read_bytes()).hexdigest(),
            "pelvis_tilt_deg": {
                k: float(np.percentile(tilt, q)) for k, q in (("p01", 1), ("p50", 50), ("p99", 99))
            },
            "feet_disagreement_deg": {
                k: float(np.percentile(disagree, q)) for k, q in (("p50", 50), ("p95", 95), ("p99", 99))
            },
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("handover")
    h.add_argument("--src", required=True)
    h.add_argument("--raw", required=True)
    h.add_argument("--out", required=True)
    e = sub.add_parser("he")
    e.add_argument("--src", required=True)
    e.add_argument("--raw", required=True)
    e.add_argument("--xml", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--workers", type=int, default=32)
    args = p.parse_args()
    {"handover": handover, "he": he}[args.cmd](args)
