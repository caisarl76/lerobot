"""Repair corrupt `observation.state` frames in the Unitree G1 Dex3 datasets and recompute the state statistics.

The Unitree state (identical in `joint28`, `sonic78_nolimit` and `sonic78_nolimit_sonicstate`; the latter only
relabels dims 0:14) has isolated frames with right-hand values up to ~3000 rad (dims 24-27, right index/middle),
266 frames in 39 episodes (found 2026-09-30). Every policy's state normalization includes them.

Method, per episode and per state dimension: a value is corrupt when it is non-finite or |value| > --threshold
(3.2 rad, beyond every G1 arm and Dex3 joint range). Corrupt values are replaced by linear interpolation in
frame_index between the nearest valid values of the same dimension (np.interp; at an episode edge the nearest valid
value is held). Valid values, other dimensions, other columns, actions and videos are not changed.

Then, for `observation.state` only:
- `meta/episodes/*.parquet` per-episode stats of the affected episodes are recomputed from the repaired frames;
- `meta/stats.json` is recomputed over all frames (every statistic already present: min, max, mean, std, count,
  q01, q10, q50, q90, q99).
The recomputation method is first checked against the stored values of dimensions / episodes that are not changed.

Every changed file is backed up next to itself (`<file><suffix>`, default `.bak-corruptstate-20260930`) before an
atomic replace. Without --apply nothing is written (dry run). Run in a training container:

  python fix_unitree_corrupt_state.py --root /run-output/datasets/joint28 --root ... --report out.json [--apply]
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STATE = "observation.state"


def state_array(table: pa.Table) -> np.ndarray:
    return np.stack(table.column(STATE).to_numpy(zero_copy_only=False)).astype(np.float32)


def state_column(values: np.ndarray, like: pa.DataType) -> pa.Array:
    flat = pa.array(values.reshape(-1), type=like.value_type)
    if pa.types.is_fixed_size_list(like):
        return pa.FixedSizeListArray.from_arrays(flat, values.shape[1])
    offsets = pa.array(np.arange(0, values.size + 1, values.shape[1], dtype=np.int32))
    return pa.ListArray.from_arrays(offsets, flat).cast(like)


def repair(state: np.ndarray, episodes: np.ndarray, frames: np.ndarray, threshold: float):
    """Return repaired copy, bad mask, and per-episode notes."""
    bad = ~np.isfinite(state) | (np.abs(state) > threshold)
    fixed = state.copy()
    notes = {}
    for episode in np.unique(episodes[bad.any(axis=1)]):
        rows = np.flatnonzero(episodes == episode)
        rows = rows[np.argsort(frames[rows])]
        for dim in np.flatnonzero(bad[rows].any(axis=0)):
            ok = ~bad[rows, dim]
            if not ok.any():
                raise ValueError(f"episode {episode} dim {dim}: no valid value to interpolate from")
            fixed[rows[~ok], dim] = np.interp(frames[rows[~ok]], frames[rows[ok]], state[rows[ok], dim])
        notes[int(episode)] = {
            "frames": int(bad[rows].any(axis=1).sum()),
            "dims": np.flatnonzero(bad[rows].any(axis=0)).tolist(),
            "max_abs_before": float(np.abs(state[rows][bad[rows]]).max()),
        }
    return fixed, bad, notes


def feature_stats(values: np.ndarray, keys) -> dict:
    out = {}
    for key in keys:
        if key == "min":
            out[key] = values.min(0)
        elif key == "max":
            out[key] = values.max(0)
        elif key == "mean":
            out[key] = values.mean(0, dtype=np.float64)
        elif key == "std":
            out[key] = values.std(0, dtype=np.float64)
        elif key == "count":
            out[key] = np.array([len(values)])
        elif key.startswith("q") and key[1:].isdigit():
            out[key] = np.quantile(values, int(key[1:]) / 100.0, axis=0)
        else:
            raise ValueError(f"unknown statistic {key}")
    return out


def backup_and_replace(path: Path, write, suffix: str) -> None:
    backup = path.with_name(path.name + suffix)
    if not backup.exists():
        shutil.copy2(path, backup)
    tmp = path.with_name(path.name + ".tmp-corruptstate")
    write(tmp)
    os.replace(tmp, path)


def process(root: Path, threshold: float, apply: bool, suffix: str) -> dict:
    report: dict = {"root": str(root), "files": {}, "episodes": {}}
    files = sorted((root / "data").rglob("*.parquet"))
    all_states, all_fixed = [], []
    affected_episodes: dict[int, np.ndarray] = {}
    for path in files:
        table = pq.read_table(path)
        state = state_array(table)
        episodes = table.column("episode_index").to_numpy()
        frames = table.column("frame_index").to_numpy()
        fixed, bad, notes = repair(state, episodes, frames, threshold)
        all_states.append(state)
        all_fixed.append(fixed)
        for episode in notes:
            part = fixed[episodes == episode]
            prev = affected_episodes.get(episode)
            affected_episodes[episode] = part if prev is None else np.concatenate([prev, part])
        if not bad.any():
            continue
        report["files"][str(path.relative_to(root))] = {
            "values": int(bad.sum()),
            "frames": int(bad.any(1).sum()),
        }
        report["episodes"].update(notes)
        if apply:
            field_index = table.schema.get_field_index(STATE)
            new_table = table.set_column(
                field_index,
                table.schema.field(field_index),
                state_column(fixed, table.schema.field(STATE).type),
            )
            compression = pq.ParquetFile(path).metadata.row_group(0).column(0).compression.lower()
            backup_and_replace(
                path, lambda tmp, t=new_table, c=compression: pq.write_table(t, tmp, compression=c), suffix
            )
            check = pq.read_table(path)
            assert check.schema.equals(table.schema, check_metadata=True), f"{path}: schema changed"
            for name in table.column_names:
                if name != STATE:
                    assert check.column(name).equals(table.column(name)), f"{path}: column {name} changed"
            assert np.array_equal(state_array(check), fixed), f"{path}: state not written as repaired"
    state, fixed = np.concatenate(all_states), np.concatenate(all_fixed)
    changed_dims = np.flatnonzero((state != fixed).any(0))
    same_dims = np.setdiff1d(np.arange(state.shape[1]), changed_dims)

    # Global stats.json: verify the method on unchanged dims, then recompute.
    stats_path = root / "meta/stats.json"
    stats = json.loads(stats_path.read_text())
    keys = list(stats[STATE])
    before = {k: np.asarray(v, dtype=np.float64) for k, v in stats[STATE].items()}
    ours_old = feature_stats(state, keys)
    ours_new = feature_stats(fixed, keys)
    report["stats_method_check_unchanged_dims"] = {
        k: float(np.abs(ours_old[k][same_dims] - before[k][same_dims]).max()) if k != "count" else None
        for k in keys
    }
    report["stats_method_check_changed_dims_before_fix"] = {
        k: float(np.abs(ours_old[k][changed_dims] - before[k][changed_dims]).max()) if k != "count" else None
        for k in keys
    }
    report["changed_dims"] = changed_dims.tolist()
    report["stats_before"] = {k: before[k][changed_dims].round(4).tolist() for k in keys if k != "count"}
    report["stats_after"] = {
        k: np.asarray(ours_new[k])[changed_dims].round(4).tolist() for k in keys if k != "count"
    }
    report["max_abs_state_after"] = float(np.abs(fixed).max())

    # Per-episode stats in meta/episodes.
    ep_files = sorted((root / "meta/episodes").rglob("*.parquet"))
    prefix = f"stats/{STATE}/"
    ep_report = {"files": [], "method_check_max_abs_diff": {}}
    checked = False
    for path in ep_files:
        table = pq.read_table(path)
        cols = [c for c in table.column_names if c.startswith(prefix)]
        ep_index = table.column("episode_index").to_numpy()
        if not checked:
            # Method check on the first unaffected episode of this file, re-read from the data files.
            for row, episode in enumerate(ep_index):
                if int(episode) not in affected_episodes:
                    values = np.concatenate([s[e == episode] for s, e in _episode_iter(files)])
                    ours = feature_stats(values, [c.removeprefix(prefix) for c in cols])
                    for c in cols:
                        stored = np.asarray(table.column(c)[row].as_py(), dtype=np.float64)
                        ep_report["method_check_max_abs_diff"][c] = float(
                            np.abs(np.asarray(ours[c.removeprefix(prefix)], dtype=np.float64) - stored).max()
                        )
                    checked = True
                    break
        rows = [i for i, e in enumerate(ep_index) if int(e) in affected_episodes]
        if not rows:
            continue
        ep_report["files"].append(str(path.relative_to(root)))
        if apply:
            new_table = table
            for c in cols:
                values = table.column(c).to_pylist()
                for i in rows:
                    stat = feature_stats(affected_episodes[int(ep_index[i])], [c.removeprefix(prefix)])
                    values[i] = np.asarray(stat[c.removeprefix(prefix)]).astype(np.float64).tolist()
                idx = table.schema.get_field_index(c)
                new_table = new_table.set_column(
                    idx, table.schema.field(idx), pa.array(values, type=table.schema.field(c).type)
                )
            compression = pq.ParquetFile(path).metadata.row_group(0).column(0).compression.lower()
            backup_and_replace(
                path, lambda tmp, t=new_table, c=compression: pq.write_table(t, tmp, compression=c), suffix
            )
            check = pq.read_table(path)
            assert check.schema.equals(table.schema, check_metadata=True), f"{path}: schema changed"
    report["episode_stats"] = ep_report

    if apply:
        stats[STATE] = {
            k: (
                np.asarray(v).astype(int).tolist()
                if k == "count"
                else np.asarray(v, dtype=np.float64).tolist()
            )
            for k, v in ours_new.items()
        }
        backup_and_replace(stats_path, lambda tmp: tmp.write_text(json.dumps(stats, indent=4)), suffix)
        # Verify: re-read everything and rescan.
        rescan = np.concatenate([state_array(pq.read_table(p)) for p in files])
        assert np.array_equal(rescan, fixed), "data re-read differs from the repaired state"
        assert not (~np.isfinite(rescan) | (np.abs(rescan) > threshold)).any(), "corrupt values remain"
        reread = json.loads(stats_path.read_text())[STATE]
        for k in keys:
            assert np.allclose(
                np.asarray(reread[k], dtype=np.float64), np.asarray(ours_new[k], dtype=np.float64)
            ), k
        report["verified"] = True
    return report


_episode_cache: list = []


def _episode_iter(files):
    if not _episode_cache:
        for p in files:
            t = pq.read_table(p, columns=["episode_index", STATE])
            _episode_cache.append((state_array(t), t.column("episode_index").to_numpy()))
    return _episode_cache


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--threshold", type=float, default=3.2)
    parser.add_argument("--suffix", default=".bak-corruptstate-20260930")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    reports = []
    for root in args.root:
        _episode_cache.clear()
        report = process(root, args.threshold, args.apply, args.suffix)
        reports.append(report)
        print(
            f"{root}: {len(report['episodes'])} episodes, "
            f"{sum(v['frames'] for v in report['episodes'].values())} frames, {len(report['files'])} data files, "
            f"{len(report['episode_stats']['files'])} episode-meta files, dims {report['changed_dims']}, "
            f"max |state| after {report['max_abs_state_after']:.3f}, applied={args.apply}"
        )
    args.report.write_text(json.dumps(reports, indent=1) + "\n")


if __name__ == "__main__":
    main()
