"""Copy a v3.0 sonic78 dataset (e.g. HE at 30 fps) at 50 Hz, the rate of our robot recordings and deploy.

observation.state and action (64 SONIC tokens + 14 hand joints) are linearly interpolated in time per episode: the same
look-ahead interpolation the streamer's ChunkResampler applies to 30 Hz token chunks (re-encoding the joints at 50 Hz
gave no gain, docs/research/2026-09-23-sonic-roundtrip-audit.md). Episode k spans the same time as before: frames
i / 50 for i = 0 .. (n - 1) * 50 // fps. Videos are shared unchanged (symlink, still 30 fps); LeRobot's decoder takes
the nearest frame (round(t * video fps)), so train with tolerance_s >= 1 / (2 * video fps), e.g. 0.02 for 30 fps.
Statistics of every non-video feature are recomputed (exact quantiles), image statistics are kept.

Usage (lerobot image, CPU only): python resample_50hz.py --src DS --out OUT [--fps 50] [--workers 16]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from prepare_state46 import stats, video_link

INTERP = ("observation.state", "action")
FRAME_COLS = ("timestamp", "frame_index", "episode_index", "index", "task_index")


def n_out(n: int, fps_in: int, fps_out: int) -> int:
    """Frames at fps_out within an episode of n frames at fps_in (last frame at or before (n - 1) / fps_in)."""
    return int((n - 1) * fps_out // fps_in) + 1


def resample_file(job: tuple) -> tuple[str, dict[int, dict[str, np.ndarray]]]:
    src_file, dest_file, fps_in, fps_out, starts = job
    g = math.gcd(fps_in, fps_out)
    step_in, step_out = (
        fps_in // g,
        fps_out // g,
    )  # frames i * step_in (in) and i * step_out (out) share a time
    table = pq.read_table(src_file)
    eps = table["episode_index"].to_numpy()
    cols = {k: np.stack(table[k].to_numpy(zero_copy_only=False)).astype(np.float64) for k in INTERP}
    frames, tasks = table["frame_index"].to_numpy(), table["task_index"].to_numpy()
    out: dict[int, dict[str, np.ndarray]] = {}
    for ep in np.unique(eps):
        m = np.flatnonzero(eps == ep)
        m = m[np.argsort(frames[m], kind="stable")]
        n = len(m)
        if not np.array_equal(frames[m], np.arange(n)) or len(set(tasks[m])) != 1:
            raise ValueError(f"{src_file} episode {ep}: frames not 0..n-1 or several tasks")
        k = n_out(n, fps_in, fps_out)
        t_in, t_out = np.arange(n) / fps_in, np.arange(k) / fps_out
        e = {
            c: np.column_stack([np.interp(t_out, t_in, cols[c][m, j]) for j in range(cols[c].shape[1])])
            for c in INTERP
        }
        for c in INTERP:  # exact on the shared time grid
            shared = e[c][::step_out]
            if not np.allclose(shared, cols[c][m][::step_in][: len(shared)], atol=1e-6):
                raise ValueError(f"episode {ep}: {c} does not hit the source frames")
        e["timestamp"] = t_out
        e["frame_index"] = np.arange(k)
        e["episode_index"] = np.full(k, ep)
        e["index"] = starts[int(ep)] + np.arange(k)
        e["task_index"] = np.full(k, tasks[m[0]])
        out[int(ep)] = e
    order = sorted(out, key=lambda ep: starts[ep])
    arrays = []
    for field in table.schema:
        v = np.concatenate([out[ep][field.name] for ep in order])
        if pa.types.is_fixed_size_list(field.type):
            arrays.append(
                pa.FixedSizeListArray.from_arrays(pa.array(v.ravel(), field.type.value_type), v.shape[1])
            )
        else:
            arrays.append(pa.array(v, field.type))
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays(arrays, schema=table.schema), dest_file, compression="zstd")
    return str(dest_file), {
        ep: {c: out[ep][c] for c in INTERP + ("timestamp", "frame_index", "index")} for ep in out
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    src, out = a.src, a.out
    info = json.loads((src / "meta/info.json").read_text())
    fps_in, fps_out = int(info["fps"]), a.fps
    if fps_in != info["fps"]:
        raise ValueError(f"non-integer source fps {info['fps']}")
    tmp = out.with_name(out.name + ".incomplete")
    if out.exists() or tmp.exists():
        raise FileExistsError(f"{out} or {tmp} already exists")

    ep_files = sorted((src / "meta/episodes").rglob("*.parquet"))
    ep_tables = {f: pq.read_table(f) for f in ep_files}
    lengths = {}
    for t in ep_tables.values():
        d = t.select(["episode_index", "length", "dataset_from_index"]).to_pydict()
        for ep, n, start in zip(d["episode_index"], d["length"], d["dataset_from_index"], strict=True):
            lengths[ep] = (start, n_out(n, fps_in, fps_out), n)
    starts, total = {}, 0
    for ep in sorted(lengths, key=lambda e: lengths[e][0]):
        starts[ep], total = total, total + lengths[ep][1]

    tmp.mkdir(parents=True)
    shutil.copytree(src / "meta", tmp / "meta")
    for f in (tmp / "meta").rglob("*"):
        if f.is_file():
            f.chmod(f.stat().st_mode | 0o200)
    (tmp / "videos").symlink_to(video_link(src, out), target_is_directory=True)
    jobs = [
        (f, tmp / f.relative_to(src), fps_in, fps_out, starts)
        for f in sorted((src / "data").rglob("*.parquet"))
    ]
    per_ep: dict[int, dict[str, np.ndarray]] = {}
    with Pool(a.workers) as pool:
        for _, eps in pool.imap_unordered(resample_file, jobs):
            per_ep.update(eps)
    if set(per_ep) != set(lengths):
        raise ValueError(f"{len(per_ep)} episodes written vs {len(lengths)} in meta")

    for f, t in ep_tables.items():
        ids = t["episode_index"].to_pylist()
        n = [len(per_ep[e]["index"]) for e in ids]
        t = t.set_column(
            t.schema.get_field_index("length"), "length", pa.array(n, t.schema.field("length").type)
        )
        for name, vals in (
            ("dataset_from_index", [starts[e] for e in ids]),
            ("dataset_to_index", [starts[e] + k for e, k in zip(ids, n, strict=True)]),
        ):
            t = t.set_column(t.schema.get_field_index(name), name, pa.array(vals, t.schema.field(name).type))
        for feat in INTERP + ("timestamp", "frame_index", "index"):
            ep_stats = {
                e: stats(np.asarray(per_ep[e][feat]).reshape(len(per_ep[e]["index"]), -1)) for e in ids
            }
            for key in ep_stats[ids[0]]:
                name = f"stats/{feat}/{key}"
                if name in t.column_names:
                    col = pa.array([ep_stats[e][key] for e in ids], t.schema.field(name).type)
                    t = t.set_column(t.schema.get_field_index(name), name, col)
        pq.write_table(t, tmp / f.relative_to(src))

    st = json.loads((tmp / "meta/stats.json").read_text())
    for feat in INTERP + ("timestamp", "frame_index", "index"):
        if feat in st:
            allv = np.concatenate(
                [np.asarray(per_ep[e][feat]).reshape(len(per_ep[e]["index"]), -1) for e in sorted(per_ep)]
            )
            st[feat] = stats(allv)
    (tmp / "meta/stats.json").write_text(json.dumps(st, indent=2))
    info["fps"] = a.fps
    info["total_frames"] = total
    for v in info["features"].values():
        if v.get("dtype") != "video" and "fps" in v:
            v["fps"] = float(a.fps)
    (tmp / "meta/info.json").write_text(json.dumps(info, indent=4))
    (tmp / "meta/resample_provenance.json").write_text(
        json.dumps(
            {
                "source_dataset": str(src),
                "fps": [fps_in, fps_out],
                "interpolated": list(INTERP),
                "method": "np.interp per column and episode; exact on the shared time grid (checked)",
                "videos": "shared, unchanged (video fps kept); train with tolerance_s >= 1/(2 video fps)",
                "frames": [sum(v[2] for v in lengths.values()), total],
            },
            indent=2,
        )
    )
    tmp.rename(out)
    print(f"{out}: {len(per_ep)} episodes, {total} frames at {fps_out} Hz (from {fps_in})")


if __name__ == "__main__":
    main()
