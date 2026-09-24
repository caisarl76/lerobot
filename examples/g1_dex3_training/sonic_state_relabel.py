"""Relabel observation.state arms with the arm pose SONIC actually reaches (user decision 2026-09-24).

A SONIC-token policy copies its state input; in closed loop the robot's SONIC-reached arm posture differs from the
teleop-recorded state, so the policy drifts. This makes training state match deployment state:

  simulate: for each episode, decode the stored sonic78_nolimit tokens closed loop in MuJoCo (offline harness,
            validated against NVIDIA's deploy), tokens look-ahead interpolated to 50 Hz like sonic_policy_streamer,
            starting from the episode's first pose; save the reached 14 arm joints at every 30 Hz frame
            (one .npz per episode, resumable, parallel).
  build:    write a new dataset: observation.state[:, :14] = reached arms, Dex3 hands (14:) kept as recorded;
            action, videos, tasks unchanged; per-episode and global observation.state statistics recomputed
            (min/max/mean/std/count, q01/q99). The source dataset is not modified.

Usage (lerobot image, PYTHONPATH with mujoco, /code = this directory):
  python sonic_state_relabel.py simulate --src /run-output/datasets/sonic78_nolimit --joint28 /run-output/datasets/joint28 --work /audit/relabel --workers 32
  python sonic_state_relabel.py build --src /run-output/datasets/sonic78_nolimit --work /audit/relabel --out /run-output/datasets/sonic78_nolimit_sonicstate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from .sonic_roundtrip_audit import MODEL, WARMUP_TICKS, Decoder, Robot, v3_episode, v3_meta
    from .sonic_targets import NOMINAL_BODY
except ImportError:
    from sonic_roundtrip_audit import MODEL, WARMUP_TICKS, Decoder, Robot, v3_episode, v3_meta
    from sonic_targets import NOMINAL_BODY

_worker = {}


def _init():
    _worker["robot"], _worker["dec"] = Robot(), Decoder()


def reached_arms(
    robot, dec, tokens: np.ndarray, first_arms: np.ndarray, fps: float
) -> tuple[np.ndarray, float]:
    """Closed-loop decode; returns reached arm joints [N,14] at each source frame and the minimum pelvis height."""
    n = len(tokens)
    rows = np.rint(np.arange(n) / fps * 50).astype(int)
    robot.reset(np.r_[NOMINAL_BODY[:15], first_arms])
    dec.reset()
    for _ in range(
        WARMUP_TICKS
    ):  # standing in the first pose, holding the first token (as after the ease-in)
        robot.step(dec(tokens[0], *robot.state()))
    out, k_next, z_min = np.empty((n, 14), np.float32), 0, np.inf
    for j in range(rows[-1] + 1):
        x = j * fps / 50
        k = min(int(np.floor(x + 1e-9)), n - 1)
        w = x - k
        tok = (1 - w) * tokens[k] + w * tokens[min(k + 1, n - 1)]
        q, dq, quat, gyro = robot.state()
        while k_next < n and rows[k_next] == j:
            out[k_next] = q[15:]
            k_next += 1
        z_min = min(z_min, float(robot.d.qpos[2]))
        robot.step(dec(tok, q, dq, quat, gyro))
    return out, z_min


def _simulate_one(job):
    ep_row, src, joint28, info, work = job
    path = Path(work) / f"{ep_row['episode_index']:05d}.npz"
    if path.exists():
        return None
    act = v3_episode(src, ep_row, info, ["frame_index", "action"])["action"].astype(np.float32)
    first = v3_episode(joint28, ep_row, info, ["frame_index", "action"])["action"][0, :14].astype(np.float64)
    arms, z_min = reached_arms(_worker["robot"], _worker["dec"], act[:, :64], first, float(info["fps"]))
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, arms=arms, pelvis_z_min=z_min)
    os.replace(tmp, path)
    return ep_row["episode_index"], z_min


def simulate(a):
    rows, info = v3_meta(a.src)
    Path(a.work).mkdir(parents=True, exist_ok=True)
    jobs = [(r, a.src, a.joint28, info, a.work) for r in rows]
    done = 0
    with Pool(a.workers, initializer=_init) as pool:
        for res in pool.imap_unordered(_simulate_one, jobs, chunksize=4):
            done += 1
            if res is not None and res[1] < 0.55:
                print(f"episode {res[0]}: pelvis fell to {res[1]:.3f} m", flush=True)
            if done % 100 == 0:
                print(f"{done}/{len(jobs)} episodes", flush=True)
    print(f"simulated {len(jobs)} episodes into {a.work}", flush=True)


def build(a):
    from lerobot.datasets.compute_stats import get_feature_stats

    src, out, work = Path(a.src), Path(a.out), Path(a.work)
    tmp = out.with_name(out.name + ".incomplete")
    if out.exists() or tmp.exists():
        raise FileExistsError(f"{out} or {tmp} already exists")
    rows, info = v3_meta(str(src))
    arms = {r["episode_index"]: np.load(work / f"{r['episode_index']:05d}.npz")["arms"] for r in rows}
    falls = [ep for ep in arms if float(np.load(work / f"{ep:05d}.npz")["pelvis_z_min"]) < 0.55]
    tmp.mkdir(parents=True)
    shutil.copytree(src / "meta", tmp / "meta")
    for p in (tmp / "meta").rglob("*"):
        if p.is_file():
            p.chmod(p.stat().st_mode | 0o200)
    (tmp / "videos").symlink_to(os.readlink(src / "videos"), target_is_directory=True)
    ep_stats, all_states = {}, []
    for f in sorted((src / "data").rglob("*.parquet")):
        table = pq.read_table(f)
        state = np.asarray(table["observation.state"].to_pylist(), np.float32)
        eps, frames = table["episode_index"].to_numpy(), table["frame_index"].to_numpy()
        for ep in np.unique(eps):
            m = eps == ep
            if len(arms[ep]) != m.sum():
                raise ValueError(f"episode {ep}: {len(arms[ep])} relabelled frames vs {m.sum()} rows")
            state[m, :14] = arms[ep][frames[m]]
            ep_stats[int(ep)] = get_feature_stats(state[m].astype(np.float64), axis=0, keepdims=False)
        all_states.append(state)
        field = table.schema.field("observation.state")
        col = pa.array(state.tolist(), type=field.type)  # keep the source column type exactly
        new = table.set_column(table.schema.get_field_index("observation.state"), field, col)
        new = new.replace_schema_metadata(table.schema.metadata)
        dest = tmp / f.relative_to(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(new, dest, compression="zstd")
        if pq.ParquetFile(dest).metadata.num_rows != table.num_rows:
            raise ValueError(f"row count changed in {dest}")
    for f in sorted((tmp / "meta/episodes").rglob("*.parquet")):
        t = pq.read_table(f)
        ids = t["episode_index"].to_pylist()
        for key in ep_stats[ids[0]]:  # min/max/mean/std/count and the q01..q99 quantiles
            name = f"stats/observation.state/{key}"
            if name in t.column_names:
                vals = pa.array([np.asarray(ep_stats[e][key]).tolist() for e in ids])
                t = t.set_column(t.schema.get_field_index(name), name, vals)
        pq.write_table(t, f)
    s = np.concatenate(all_states).astype(np.float64)
    stats = json.loads((tmp / "meta/stats.json").read_text())
    stats["observation.state"] = {
        k: np.asarray(v).tolist() for k, v in get_feature_stats(s, axis=0, keepdims=False).items()
    }
    (tmp / "meta/stats.json").write_text(json.dumps(stats, indent=2))
    prov = {
        "source_dataset": str(src),
        "relabelled": "observation.state[:, :14] = SONIC-reached arm joints; [:, 14:] Dex3 hands as recorded",
        "harness": "sonic_roundtrip_audit offline MuJoCo closed loop, v1.1 decoder, look-ahead token interpolation",
        "decoder_sha256": hashlib.sha256((MODEL / "model_decoder.onnx").read_bytes()).hexdigest(),
        "episodes": len(arms),
        "frames": int(len(s)),
        "episodes_pelvis_below_0.55m": falls,
    }
    (tmp / "meta/relabel_provenance.json").write_text(json.dumps(prov, indent=2))
    tmp.rename(out)
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--src", required=True)
    s.add_argument("--joint28", required=True)
    s.add_argument("--work", required=True)
    s.add_argument("--workers", type=int, default=16)
    b = sub.add_parser("build")
    b.add_argument("--src", required=True)
    b.add_argument("--work", required=True)
    b.add_argument("--out", required=True)
    args = p.parse_args()
    {"simulate": simulate, "build": build}[args.cmd](args)
