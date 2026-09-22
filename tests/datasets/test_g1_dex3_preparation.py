"""Real parquet/video fixtures for joint dataset preparation (CPU only)."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
from inventory import EXPECTED_JOINT_ORDER, SUFFIXES, _one
from prepare_joint_dataset import CAMERAS, _verify_equivalence, create_projection, prepare, validate_source


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.video = self.base / "sample.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=s=16x16:r=30",
                "-frames:v",
                "6",
                "-c:v",
                "libx264",
                str(self.video),
            ],
            check=True,
        )

    def source(self, suffix="ObjectPlacement", *, hwc=False, extra=True):
        root = self.base / "sources" / f"G1_Dex3_{suffix}_Dataset"
        for folder in ("meta/episodes/chunk-000", "data/chunk-000"):
            (root / folder).mkdir(parents=True)
        features = {
            key: {"dtype": "float32", "shape": [28], "names": [list(EXPECTED_JOINT_ORDER)]}
            for key in ("action", "observation.state")
        }
        for key, dtype in (
            ("timestamp", "float32"),
            ("frame_index", "int64"),
            ("episode_index", "int64"),
            ("index", "int64"),
            ("task_index", "int64"),
        ):
            features[key] = {"dtype": dtype, "shape": [1], "names": None}
        camera_keys = (*CAMERAS, "observation.images.wrist") if extra else CAMERAS
        episode_rows = []
        for episode in range(2):
            row = {
                "episode_index": episode,
                "length": 3,
                "tasks": [suffix],
                "dataset_from_index": episode * 3,
                "dataset_to_index": (episode + 1) * 3,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
            for camera in camera_keys:
                row.update(
                    {
                        f"videos/{camera}/chunk_index": 0,
                        f"videos/{camera}/file_index": 0,
                        f"videos/{camera}/from_timestamp": episode / 10,
                        f"videos/{camera}/to_timestamp": (episode + 1) / 10,
                        f"stats/{camera}/count": [3],
                    }
                )
            episode_rows.append(row)
        for camera in camera_keys:
            features[camera] = {
                "dtype": "video",
                "shape": [16, 16, 3] if hwc else [3, 16, 16],
                "names": ["height", "width", "channels"] if hwc else ["channels", "height", "width"],
                "info": {"video.fps": 30},
            }
            path = root / "videos" / camera / "chunk-000/file-000.mp4"
            path.parent.mkdir(parents=True)
            shutil.copyfile(self.video, path)
        info = {
            "codebase_version": "v3.0",
            "robot_type": "g1",
            "fps": 30,
            "total_episodes": 2,
            "total_frames": 6,
            "total_tasks": 1,
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 500,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "splits": {"train": "0:2"},
            "features": features,
        }
        (root / "meta/info.json").write_text(json.dumps(info))
        stats = {
            k: {"min": [0] * 28, "max": [1] * 28, "mean": [0.5] * 28, "std": [0.2] * 28, "count": [6]}
            for k in ("action", "observation.state")
        }
        stats.update(
            {
                k: {
                    "min": [[[0]]] * 3,
                    "max": [[[1]]] * 3,
                    "mean": [[[0.5]]] * 3,
                    "std": [[[0.2]]] * 3,
                    "count": [6],
                }
                for k in camera_keys
            }
        )
        (root / "meta/stats.json").write_text(json.dumps(stats))
        pq.write_table(pa.table({"task": [suffix], "task_index": [0]}), root / "meta/tasks.parquet")
        pq.write_table(pa.Table.from_pylist(episode_rows), root / "meta/episodes/chunk-000/file-000.parquet")
        numeric = np.arange(168, dtype=np.float32).reshape(6, 28)
        table = pa.table(
            {
                "action": numeric.tolist(),
                "observation.state": (numeric + 1).tolist(),
                "episode_index": [0, 0, 0, 1, 1, 1],
                "frame_index": [0, 1, 2] * 2,
                "index": list(range(6)),
                "timestamp": [0, 1 / 30, 2 / 30] * 2,
                "task_index": [0] * 6,
            }
        )
        pq.write_table(table, root / "data/chunk-000/file-000.parquet")
        return root

    def test_projection_preserves_source_and_hand_order(self):
        root = self.source(hwc=True)
        before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
        row, _ = _one(root, "unitree/test", None, CAMERAS)
        validate_source(root)
        target = self.base / "projection"
        create_projection(row, target)
        projected_info = json.loads((target / "meta/info.json").read_text())
        self.assertEqual(projected_info["robot_type"], "Unitree_G1_Dex3")
        self.assertEqual(row["source_robot_type"], "g1")
        features = projected_info["features"]
        self.assertEqual(features[CAMERAS[0]]["shape"], [3, 16, 16])
        self.assertEqual(features["action"]["names"], list(EXPECTED_JOINT_ORDER))
        self.assertNotIn("observation.images.wrist", features)
        self.assertNotIn("observation.images.wrist", json.loads((target / "meta/stats.json").read_text()))
        metadata = pq.read_table(target / "meta/episodes/chunk-000/file-000.parquet")
        self.assertFalse(any("wrist" in c for c in metadata.column_names))
        self.assertTrue((target / "data").is_symlink())
        validate_source(target)
        self.assertEqual(
            before, {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
        )

    def test_missing_last_frame_rejected(self):
        root = self.source()
        path = root / "data/chunk-000/file-000.parquet"
        pq.write_table(pq.read_table(path).slice(0, 5), path)
        with self.assertRaisesRegex(ValueError, "frame|count|rows"):
            validate_source(root)

    def test_numeric_statistics_recomputed_from_rows_with_missing_source_counts(self):
        root = self.source()
        stats_path = root / "meta/stats.json"
        stats = json.loads(stats_path.read_text())
        for key in ("action", "observation.state"):
            stats[key].pop("count")
        stats_path.write_text(json.dumps(stats))
        original = stats_path.read_bytes()
        validation = validate_source(root)
        row, _ = _one(root, "unitree/test", None, CAMERAS)
        row.update(validation)
        target = self.base / "projection"
        create_projection(row, target)
        projected = json.loads((target / "meta/stats.json").read_text())
        table = pq.read_table(root / "data/chunk-000/file-000.parquet")
        for key in ("action", "observation.state", "timestamp"):
            values = np.asarray(table[key].to_pylist()).reshape(6, -1)
            self.assertEqual(projected[key]["count"], [6])
            np.testing.assert_allclose(projected[key]["mean"], values.mean(axis=0))
            np.testing.assert_allclose(projected[key]["std"], values.std(axis=0))
            np.testing.assert_allclose(projected[key]["q01"], np.quantile(values, 0.01, axis=0))
        self.assertEqual(stats_path.read_bytes(), original)

    def test_repairs_shard_boundary_metadata_from_verified_payload_only(self):
        root = self.source()
        first = root / "data/chunk-000/file-000.parquet"
        second = root / "data/chunk-000/file-001.parquet"
        payload = pq.read_table(first)
        pq.write_table(payload.slice(0, 3), first)
        pq.write_table(payload.slice(3, 3), second)
        meta_path = root / "meta/episodes/chunk-000/file-000.parquet"
        rows = pq.read_table(meta_path).to_pylist()
        rows[1]["dataset_from_index"] = 0
        rows[1]["dataset_to_index"] = 3
        # It also incorrectly retains the previous shard index, as in ToastedBread.
        pq.write_table(pa.Table.from_pylist(rows), meta_path)
        original = meta_path.read_bytes()
        result = validate_source(root)
        self.assertEqual(len(result["episode_metadata_repairs"]), 1)
        repair = result["episode_metadata_repairs"][0]
        self.assertEqual(repair["episode_index"], 1)
        self.assertEqual(repair["corrected"]["dataset_from_index"], 3)
        self.assertEqual(repair["corrected"]["data/file_index"], 1)
        row, _ = _one(root, "unitree/test", None, CAMERAS)
        row.update(result)
        target = self.base / "projection"
        create_projection(row, target)
        self.assertEqual(validate_source(target)["episode_metadata_repairs"], [])
        self.assertEqual(meta_path.read_bytes(), original)
        pq.write_table(payload.slice(3, 2), second)
        with self.assertRaisesRegex(ValueError, "frame|rows|count"):
            validate_source(root)

    def test_invalid_numeric_or_index_rejected(self):
        for key, replacement in (
            ("action", [[float("nan")] * 28] * 6),
            ("frame_index", [0, 1, 2, 0, 1, 1]),
            ("timestamp", [0, 0.5, 2 / 30] * 2),
            ("task_index", [5] * 6),
        ):
            with self.subTest(key=key):
                root = self.source(suffix=key)
                path = root / "data/chunk-000/file-000.parquet"
                table = pq.read_table(path)
                table = table.set_column(table.schema.get_field_index(key), key, pa.array(replacement))
                pq.write_table(table, path)
                with self.assertRaises(ValueError):
                    validate_source(root)

    def test_episode_text_repaired_from_authoritative_row_task_indices(self):
        root = self.source()
        path = root / "meta/episodes/chunk-000/file-000.parquet"
        rows = pq.read_table(path).to_pylist()
        rows[1]["tasks"] = ["Wrong copied cup instruction"]
        pq.write_table(pa.Table.from_pylist(rows), path)
        original = path.read_bytes()
        result = validate_source(root)
        self.assertEqual(result["episode_metadata_repairs"][0]["corrected"]["tasks"], ["ObjectPlacement"])
        row, _ = _one(root, "unitree/test", None, CAMERAS)
        row.update(result)
        target = self.base / "projection"
        create_projection(row, target)
        self.assertEqual(validate_source(target)["episode_metadata_repairs"], [])
        self.assertEqual(path.read_bytes(), original)

    def test_equivalence_detects_changed_hand_values_after_reindexing(self):
        root = self.source()
        original = pq.read_table(root / "data/chunk-000/file-000.parquet").to_pydict()
        original["episode_index"] = [x + 7 for x in original["episode_index"]]
        original["index"] = [x + 100 for x in original["index"]]
        original["task_index"] = [1] * 6
        output = self.base / "merged.parquet"
        pq.write_table(pa.table(original), output, row_group_size=3)
        _verify_equivalence(root, [output], 7, 100, ["another task", "ObjectPlacement"])
        original["action"][-1][21], original["action"][-1][22] = (
            original["action"][-1][22],
            original["action"][-1][21],
        )
        pq.write_table(pa.table(original), output, row_group_size=3)
        with self.assertRaisesRegex(ValueError, "values changed"):
            _verify_equivalence(root, [output], 7, 100, ["another task", "ObjectPlacement"])

    def test_video_range_rejected(self):
        root = self.source()
        path = root / "meta/episodes/chunk-000/file-000.parquet"
        table = pq.read_table(path)
        key = f"videos/{CAMERAS[0]}/to_timestamp"
        pq.write_table(table.set_column(table.schema.get_field_index(key), key, pa.array([0.1, 2.0])), path)
        with self.assertRaisesRegex(ValueError, "video"):
            validate_source(root)

    def test_real_native_merge_all_sources(self):
        try:
            import lerobot.datasets.aggregate  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"native aggregation dependencies unavailable: {exc}")
        import pandas as pd

        for i, suffix in enumerate(SUFFIXES):
            root = self.source(suffix, hwc=i >= 6, extra=i < 6)
            info_path = root / "meta/info.json"
            info = json.loads(info_path.read_text())
            info["robot_type"] = "Unitree_G1" if i < 6 else "Unitree_G1_Dex3"
            info_path.write_text(json.dumps(info))
            pd.DataFrame({"task_index": [0]}, index=pd.Index([suffix], name="task")).to_parquet(
                root / "meta/tasks.parquet"
            )
        output = self.base / "combined"
        manifest = prepare(self.base / "sources", output)
        self.assertEqual((manifest["total_episodes"], manifest["total_frames"]), (26, 78))
        self.assertEqual(manifest["total_tasks"], 13)
        self.assertEqual(manifest["datasets"][-1]["episode_offset"], 24)
        self.assertEqual(manifest["datasets"][-1]["frame_offset"], 72)
        self.assertEqual(manifest["smoke_episodes"], list(range(26)))
        self.assertEqual(json.loads((output / "meta/provenance.json").read_text()), manifest)
        videos = list((output / "videos").rglob("*.mp4"))
        self.assertTrue(all(v.is_symlink() for v in videos))
        self.assertEqual(len(videos), 26)


if __name__ == "__main__":
    unittest.main()
