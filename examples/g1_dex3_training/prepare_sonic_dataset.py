"""Convert a verified joint28 LeRobot v3 dataset to actual SONIC64 + Dex3 hands.

CPU-only; every episode and source frame is retained. Videos remain symlinked to
the immutable source. Failed builds remain in .incomplete for inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from .inventory import EXPECTED_JOINT_ORDER
    from .sonic_targets import (
        ACTION_NAMES,
        SonicEncoder,
        build_encoder_inputs,
        combine_tokens_and_hands,
        load_joint_limits,
    )
except ImportError:
    from inventory import EXPECTED_JOINT_ORDER
    from sonic_targets import (
        ACTION_NAMES,
        SonicEncoder,
        build_encoder_inputs,
        combine_tokens_and_hands,
        load_joint_limits,
    )


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _action_schema(table, values):
    """Replace both Arrow's type and HF's cached feature declaration."""
    column = pa.FixedSizeListArray.from_arrays(pa.array(values.ravel(), type=pa.float32()), 78)
    table = table.set_column(table.schema.get_field_index("action"), "action", column)
    metadata = dict(table.schema.metadata or {})
    if b"huggingface" in metadata:
        hf = json.loads(metadata[b"huggingface"])
        feature = hf["info"]["features"]["action"]
        feature.update(length=78, feature={"dtype": "float32", "_type": "Value"})
        metadata[b"huggingface"] = json.dumps(hf).encode()
    return table.replace_schema_metadata(metadata)


def _episode_tables(path):
    """Accumulate only the current episode, irrespective of rowgroup boundaries."""
    current, pieces = None, []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
        table = pa.Table.from_batches([batch])
        ids = table["episode_index"].to_numpy()
        boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            episode = int(ids[start])
            if current is not None and episode != current:
                yield current, pa.concat_tables(pieces)
                pieces = []
            current = episode
            pieces.append(table.slice(int(start), int(end - start)))
    if pieces:
        yield current, pa.concat_tables(pieces)


def _verify_file(source, output):
    original, converted = pq.ParquetFile(source), pq.ParquetFile(output)
    _require(original.metadata.num_rows == converted.metadata.num_rows, "converted row count changed")
    _require(converted.schema_arrow.field("action").type == pa.list_(pa.float32(), 78), "invalid action type")
    for left, right in zip(original.iter_batches(8192), converted.iter_batches(8192), strict=True):
        left, right = pa.Table.from_batches([left]), pa.Table.from_batches([right])
        _require(left.drop(["action"]).equals(right.drop(["action"])), "nonaction source columns changed")
        values = np.asarray(right["action"].to_pylist(), dtype=np.float32)
        _require(values.shape == (right.num_rows, 78), "invalid converted action dimensions")
        combine_tokens_and_hands(values[:, :64], values[:, 64:])


def prepare(source_root: Path, output_root: Path, robot_xml: Path, *, encoder):
    """Build completely, verify persisted rows, then atomically publish the dataset."""
    from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats

    started = time.monotonic()
    source_root, output_root, robot_xml = (Path(p).absolute() for p in (source_root, output_root, robot_xml))
    incomplete = output_root.with_name(output_root.name + ".incomplete")
    _require(not os.path.lexists(output_root), f"output already exists: {output_root}")
    _require(not os.path.lexists(incomplete), f"incomplete output already exists: {incomplete}")
    _require(not output_root.is_relative_to(source_root), "output must be outside source dataset")
    info = json.loads((source_root / "meta/info.json").read_text())
    _require(info["codebase_version"] == "v3.0" and info["fps"] == 30, "requires v3.0 30Hz source")
    names = info["features"]["action"]["names"]
    if len(names) == 1 and isinstance(names[0], list):
        names = names[0]
    _require(
        info["features"]["action"]["shape"] == [28] and names == list(EXPECTED_JOINT_ORDER),
        "invalid joint28 source order",
    )
    provenance_path = source_root / "meta/provenance.json"
    source_provenance = json.loads(provenance_path.read_text())
    limits = load_joint_limits(robot_xml)
    metadata_paths = sorted((source_root / "meta/episodes").rglob("*.parquet"))
    episode_rows = [row for path in metadata_paths for row in pq.read_table(path).to_pylist()]
    episodes = {row["episode_index"]: row for row in episode_rows}
    _require(
        len(episodes) == len(episode_rows) == info["total_episodes"] > 0, "source episode count mismatch"
    )
    _require(
        sum(row["length"] for row in episode_rows) == info["total_frames"], "source frame count mismatch"
    )
    _require((source_root / "videos").is_dir(), "source videos missing")
    incomplete.mkdir(parents=True)
    shutil.copytree(source_root / "meta", incomplete / "meta")
    # Copied projection metadata may have been read-only; only change destination modes.
    for path in (incomplete / "meta").rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode | 0o200)
    (incomplete / "videos").symlink_to(source_root / "videos", target_is_directory=True)
    episode_stats, reports = {}, []
    frames = 0
    for source_path in sorted((source_root / "data").rglob("*.parquet")):
        destination = incomplete / source_path.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        writer = None
        try:
            for episode, table in _episode_tables(source_path):
                episode_started = time.monotonic()
                _require(
                    episode in episodes and episode not in episode_stats,
                    f"unknown/repeated episode {episode}",
                )
                row = episodes[episode]
                _require(
                    table.num_rows == row["length"],
                    f"episode {episode} length mismatch or split across files",
                )
                _require(
                    np.array_equal(table["frame_index"].to_numpy(), np.arange(row["length"])),
                    "source frame indices changed",
                )
                _require(
                    np.array_equal(
                        table["index"].to_numpy(),
                        np.arange(row["dataset_from_index"], row["dataset_to_index"]),
                    ),
                    "source global indices mismatch",
                )
                actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
                inputs, hands, report = build_encoder_inputs(actions, limits)
                encode_started = time.monotonic()
                values = combine_tokens_and_hands(encoder.encode(inputs), hands)
                report["encoding_seconds"] = time.monotonic() - encode_started
                converted = _action_schema(table, values)
                if writer is None:
                    writer = pq.ParquetWriter(destination, converted.schema, compression="zstd")
                writer.write_table(converted, row_group_size=table.num_rows)
                episode_stats[episode] = get_feature_stats(values.astype(np.float64), axis=0, keepdims=False)
                frames += table.num_rows
                reports.append(
                    {
                        "episode_index": episode,
                        "frames": table.num_rows,
                        **report,
                        "seconds": time.monotonic() - episode_started,
                    }
                )
                if len(reports) % 50 == 0:
                    progress = {
                        "episodes": len(reports),
                        "frames": frames,
                        "seconds": time.monotonic() - started,
                    }
                    _write_json(incomplete / "meta/conversion_progress.json", progress)
                    print(json.dumps(progress), flush=True)
        finally:
            if writer is not None:
                writer.close()
        _verify_file(source_path, destination)
    _require(
        set(episode_stats) == set(episodes) and frames == info["total_frames"],
        "converted episode/frame count mismatch",
    )
    for source_path in metadata_paths:
        table = pq.read_table(source_path)
        ids = table["episode_index"].to_pylist()
        # Drop every old action statistic, including optional quantiles, before replacement.
        table = table.drop([key for key in table.column_names if key.startswith("stats/action/")])
        for key in next(iter(episode_stats.values())):
            array = pa.array([episode_stats[ep][key].tolist() for ep in ids])
            table = table.append_column(f"stats/action/{key}", array)
        pq.write_table(table.replace_schema_metadata(None), incomplete / source_path.relative_to(source_root))
    info["features"]["action"].update(
        shape=[78], names=[f"sonic_token_{i:02d}" for i in range(64)] + names[14:]
    )
    _write_json(incomplete / "meta/info.json", info)
    stats = json.loads((source_root / "meta/stats.json").read_text())
    aggregated = aggregate_stats([{"action": value} for value in episode_stats.values()])["action"]
    stats["action"] = {key: value.tolist() for key, value in aggregated.items()}
    _write_json(incomplete / "meta/stats.json", stats)
    manifest = {
        "source_root": str(source_root),
        "source_provenance": source_provenance,
        "source_provenance_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
        "source_total_episodes": info["total_episodes"],
        "source_total_frames": info["total_frames"],
        "total_episodes": len(episode_stats),
        "total_frames": frames,
        "selection": "all_episodes",
        # Unitree source and XML names describe the same positional sequence.
        "source_to_encoder_joint_mapping": [
            {"index": index, "source_name": source_name, "encoder_xml_name": xml_name}
            for index, (source_name, xml_name) in enumerate(zip(names, ACTION_NAMES, strict=True))
        ],
        "robot_xml_sha256": hashlib.sha256(robot_xml.read_bytes()).hexdigest(),
        **encoder.provenance,
        "episode_reports": reports,
        "seconds": time.monotonic() - started,
        "validation": "all persisted rows: finite quantized78, unchanged nonaction columns, counts",
    }
    _write_json(incomplete / "meta/provenance.json", manifest)
    _write_json(
        incomplete / "meta/conversion_progress.json",
        {"complete": True, "episodes": len(episode_stats), "frames": frames},
    )
    _require(not os.path.lexists(output_root), f"output appeared during conversion: {output_root}")
    incomplete.rename(output_root)
    print(f"Verified {len(episode_stats)} episodes / {frames} frames: {output_root}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "output-root", "encoder-model", "observation-config", "robot-xml"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    encoder = SonicEncoder(args.encoder_model, args.observation_config)
    prepare(args.source_root, args.output_root, args.robot_xml, encoder=encoder)


if __name__ == "__main__":
    main()
