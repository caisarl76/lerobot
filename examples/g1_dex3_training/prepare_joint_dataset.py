"""Prepare all 13 joint-action datasets without rewriting source data or copying videos.

Run with ``python -m examples.g1_dex3_training.prepare_joint_dataset``. Source
mounts must remain available at the same paths while training the merged dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

try:
    from .inventory import build_inventory
except ImportError:
    from inventory import build_inventory

CAMERAS = ("observation.images.cam_left_high", "observation.images.cam_right_high")
BATCH_SIZE = 8192


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _episodes(root):
    rows = []
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return sorted(rows, key=lambda row: row["episode_index"])


def _tasks(root):
    table = pq.read_table(root / "meta/tasks.parquet")
    # pandas persists its named string index as a parquet column.
    name = "task" if "task" in table.column_names else "__index_level_0__"
    rows = table.to_pylist()
    mapping = {row["task_index"]: row[name] for row in rows}
    _require(
        len(mapping) == len(rows) and sorted(mapping) == list(range(len(rows))),
        f"invalid task indices: {root}",
    )
    _require(len(set(mapping.values())) == len(rows), f"duplicate task names: {root}")
    return mapping


def _duration(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(json.loads(result.stdout)["streams"][0]["duration"])
    _require(math.isfinite(duration) and duration > 0, f"invalid video duration: {path}")
    return duration


def validate_source(root: Path):
    """Scan every numeric row and video header; do not decode video frames."""
    info = json.loads((root / "meta/info.json").read_text())
    _require(info["codebase_version"] == "v3.0", f"requires v3.0 metadata: {root}")
    _require((root / "meta/stats.json").is_file(), f"missing stats: {root}")
    episodes = _episodes(root)
    tasks = _tasks(root)
    _require(len(tasks) == info["total_tasks"], f"task count mismatch: {root}")
    _require(
        [e["episode_index"] for e in episodes] == list(range(info["total_episodes"])),
        f"episode count/indices mismatch: {root}",
    )
    recorded_starts = np.asarray([e["dataset_from_index"] for e in episodes], dtype=np.int64)
    lengths = np.asarray([e["length"] for e in episodes], dtype=np.int64)
    ends = np.asarray([e["dataset_to_index"] for e in episodes], dtype=np.int64)
    _require(
        np.all(lengths > 0)
        and np.array_equal(ends - recorded_starts, lengths)
        and int(lengths.sum()) == info["total_frames"],
        f"episode frame ranges mismatch: {root}",
    )
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    paths = sorted((root / "data").rglob("*.parquet"))
    actual_paths = {}
    actual_tasks = {}
    seen = np.zeros(len(episodes), dtype=np.int64)
    numeric_columns = {key: [] for key in ("action", "observation.state", "timestamp")}
    total = 0
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_SIZE):
            columns = batch.to_pydict()
            for key in ("action", "observation.state"):
                values = np.asarray(columns[key])
                _require(
                    values.shape == (batch.num_rows, 28) and np.isfinite(values).all(),
                    f"nonfinite or incorrect 28D {key}: {path}",
                )
            for key in ("episode_index", "index", "frame_index", "task_index"):
                values = np.asarray(columns[key])
                _require(np.issubdtype(values.dtype, np.integer), f"noninteger {key}: {path}")
            ep = np.asarray(columns["episode_index"])
            _require(np.all((ep >= 0) & (ep < len(episodes))), f"unknown episode: {path}")
            indices = np.asarray(columns["index"])
            frames = np.asarray(columns["frame_index"])
            _require(
                np.array_equal(indices, np.arange(total, total + batch.num_rows))
                and np.array_equal(frames, indices - starts[ep])
                and np.all((frames >= 0) & (frames < lengths[ep])),
                f"frame indices mismatch: {path}",
            )
            timestamps = np.asarray(columns["timestamp"])
            _require(
                np.isfinite(timestamps).all()
                and np.allclose(timestamps, frames / info["fps"], atol=1e-3, rtol=0),
                f"timestamps mismatch: {path}",
            )
            for key, pieces in numeric_columns.items():
                pieces.append(np.asarray(columns[key], dtype=np.float64).reshape(batch.num_rows, -1))
            for episode in np.unique(ep):
                _require(
                    episode not in actual_paths or actual_paths[episode] == path,
                    f"episode spans multiple parquet shards: {path}",
                )
                actual_paths[episode] = path
                task_ids = set(np.asarray(columns["task_index"])[ep == episode].tolist())
                _require(task_ids <= tasks.keys(), f"unknown task index: {path}")
                actual_tasks.setdefault(int(episode), set()).update(task_ids)
            seen += np.bincount(ep, minlength=len(episodes))
            total += batch.num_rows
    _require(
        total == info["total_frames"] and np.array_equal(seen, lengths),
        f"missing frame rows or episode count mismatch: {root}",
    )
    repairs = []
    for index, episode in enumerate(episodes):
        path = actual_paths[index]
        corrected = {
            "dataset_from_index": int(starts[index]),
            "dataset_to_index": int(starts[index] + lengths[index]),
            "data/chunk_index": int(path.parent.name.removeprefix("chunk-")),
            "data/file_index": int(path.stem.removeprefix("file-")),
            "tasks": [tasks[task_id] for task_id in sorted(actual_tasks[index])],
        }
        original = {key: episode[key] for key in corrected}
        if original != corrected:
            repairs.append({"episode_index": index, "original": original, "corrected": corrected})
    durations = {}
    cameras = [k for k, value in info["features"].items() if value["dtype"] == "video"]
    for episode in episodes:
        for camera in cameras:
            prefix = f"videos/{camera}/"
            path = root / info["video_path"].format(
                video_key=camera,
                chunk_index=episode[prefix + "chunk_index"],
                file_index=episode[prefix + "file_index"],
            )
            _require(path.is_file(), f"missing video: {path}")
            if path not in durations:
                durations[path] = _duration(path)
            start, end = episode[prefix + "from_timestamp"], episode[prefix + "to_timestamp"]
            last_frame = start + (episode["length"] - 1) / info["fps"]
            _require(
                math.isfinite(start)
                and math.isfinite(end)
                and start >= 0
                and end > start
                and end <= durations[path] + 1e-3
                and last_frame < end + 1e-3
                and last_frame < durations[path] + 1e-3,
                f"video timestamp range exceeds duration: {path}, episode {episode['episode_index']}",
            )
    # Source global summaries omit counts. Compute actual full-row normalization
    # statistics instead of assigning counts to unverified source summaries.
    numeric_stats = {}
    for key, pieces in numeric_columns.items():
        values = np.concatenate(pieces)
        numeric_stats[key] = {
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
            "count": [total],
        }
    return {
        "episodes": len(episodes),
        "frames": total,
        "tasks": tasks,
        "episode_metadata_repairs": repairs,
        "recomputed_numeric_stats": numeric_stats,
    }


def create_projection(row, target: Path):
    """Copy metadata, normalize its schema, and link the immutable payloads."""
    source = Path(row["path"]).resolve()
    target.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / "meta", target / "meta")
    (target / "data").symlink_to(source / "data", target_is_directory=True)
    (target / "videos").symlink_to(source / "videos", target_is_directory=True)
    info = json.loads((target / "meta/info.json").read_text())
    # The inventory has verified the same named 28-joint Dex3 contract in all sources.
    info["robot_type"] = "Unitree_G1_Dex3"
    dropped = row["dropped_cameras"]
    for key in dropped:
        info["features"].pop(key)
    info["features"].update(deepcopy(row["camera_features"]))
    for key, names in (
        ("action", row["action_names"]),
        ("observation.state", row["observation_state_names"]),
    ):
        info["features"][key]["names"] = names
    (target / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    stats_path = target / "meta/stats.json"
    stats = json.loads(stats_path.read_text())
    for key in dropped:
        stats.pop(key, None)
    stats.update(row.get("recomputed_numeric_stats", {}))
    stats_path.write_text(json.dumps(stats) + "\n")
    for path in (target / "meta/episodes").rglob("*.parquet"):
        table = pq.read_table(path)
        repairs = {
            item["episode_index"]: item["corrected"] for item in row.get("episode_metadata_repairs", [])
        }
        if repairs:
            records = table.to_pylist()
            for record in records:
                record.update(repairs.get(record["episode_index"], {}))
            table = type(table).from_pylist(records, schema=table.schema)
        columns = [
            name
            for name in table.column_names
            if not any(name.startswith((f"videos/{key}/", f"stats/{key}/")) for key in dropped)
        ]
        pq.write_table(table.select(columns).replace_schema_metadata(None), path)
    # Only copied metadata is chmod'd; never follow the payload symlinks.
    for path in (target / "meta").rglob("*"):
        if path.is_file():
            path.chmod(0o444)


def _verify_equivalence(source, destination, episode_offset, frame_offset, destination_tasks):
    """Compare every payload column in bounded batches, allowing only native reindexing."""
    source_tasks = _tasks(source)
    task_remap = {i: destination_tasks.index(name) for i, name in source_tasks.items()}
    output_files = iter(destination)
    for source_file in sorted((source / "data").rglob("*.parquet")):
        output_file = next(output_files)
        source_parquet, output_parquet = pq.ParquetFile(source_file), pq.ParquetFile(output_file)
        _require(
            source_parquet.metadata.num_rows == output_parquet.metadata.num_rows,
            f"merged frame count mismatch: {output_file}",
        )
        # Native writes one row group per episode; iter_batches still spans groups.
        for left, right in zip(
            source_parquet.iter_batches(batch_size=BATCH_SIZE),
            output_parquet.iter_batches(batch_size=BATCH_SIZE),
            strict=True,
        ):
            original, merged = left.to_pydict(), right.to_pydict()
            _require(original.keys() == merged.keys(), f"merged columns changed: {output_file}")
            original["episode_index"] = [x + episode_offset for x in original["episode_index"]]
            original["index"] = [x + frame_offset for x in original["index"]]
            original["task_index"] = [task_remap[x] for x in original["task_index"]]
            _require(original == merged, f"merged values changed: {output_file}")


def prepare(source_root: Path, output_root: Path, repo_id="local/g1_dex3_all_joint28"):
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    _require(not output_root.exists(), f"output already exists: {output_root}")
    inventory = build_inventory(source_root, camera_keys=CAMERAS)
    projection_root = output_root.parent / f"{output_root.name}_sources"
    _require(not projection_root.exists(), f"projection directory already exists: {projection_root}")
    # Complete validation before creating any projection or output.
    for row in inventory["datasets"]:
        print(f"Validating {row['repo_id']}", flush=True)
        validation = validate_source(Path(row["path"]))
        row["episode_metadata_repairs"] = validation["episode_metadata_repairs"]
        row["recomputed_numeric_stats"] = validation["recomputed_numeric_stats"]
    roots = []
    for row in inventory["datasets"]:
        target = projection_root / Path(row["path"]).name
        create_projection(row, target)
        roots.append(target)
    aggregate_datasets(
        inventory["repo_ids"],
        repo_id,
        roots=roots,
        aggr_root=output_root,
        concatenate_videos=False,
        concatenate_data=False,
        symlink_videos=True,
    )
    meta = LeRobotDatasetMetadata(repo_id, root=output_root)
    validation = validate_source(output_root)
    _require(not validation["episode_metadata_repairs"], "merged episode metadata does not match its payload")
    _require(
        meta.total_episodes == inventory["total_episodes"] and meta.total_frames == inventory["total_frames"],
        "native reader total counts mismatch",
    )
    destination_tasks = list(meta.tasks.index)
    expected_tasks = {name for root in roots for name in _tasks(root).values()}
    _require(set(destination_tasks) == expected_tasks, "merged task names changed")
    _require(meta.info.total_tasks == len(destination_tasks), "native reader task counts mismatch")
    files = sorted((output_root / "data").rglob("*.parquet"))
    manifest = {
        **inventory,
        "repo_id": repo_id,
        "camera_keys": list(CAMERAS),
        "total_tasks": len(destination_tasks),
        "smoke_episodes": [],
        "validation_level": "all-parquet-rows-and-video-headers",
        "note": "Video frames are not decoded here; source mounts must remain available.",
    }
    episode_offset = frame_offset = file_offset = 0
    for row in manifest["datasets"]:
        count = row["data_file_count"]
        _verify_equivalence(
            Path(row["path"]),
            files[file_offset : file_offset + count],
            episode_offset,
            frame_offset,
            destination_tasks,
        )
        row.update(
            episode_offset=episode_offset,
            frame_offset=frame_offset,
            episode_range=[episode_offset, episode_offset + row["episodes"]],
            frame_range=[frame_offset, frame_offset + row["frames"]],
            validation_level=manifest["validation_level"],
        )
        manifest["smoke_episodes"].extend(sorted({episode_offset, episode_offset + row["episodes"] - 1}))
        episode_offset += row["episodes"]
        frame_offset += row["frames"]
        file_offset += count
    _require(
        file_offset == len(files) and validation["frames"] == frame_offset,
        "unexpected merged parquet files/count",
    )
    (output_root / "meta/provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/g1_dex3_all_joint28")
    args = parser.parse_args()
    prepare(args.source_root, args.output_root, args.repo_id)


if __name__ == "__main__":
    main()
