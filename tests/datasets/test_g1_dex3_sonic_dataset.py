"""Real Arrow conversion tests; deterministic quantized encoder is test-only."""

import ast
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "examples/g1_dex3_training"))
sys.path.insert(0, str(ROOT / "src"))
from inventory import EXPECTED_JOINT_ORDER  # noqa: E402
from sonic_targets import ACTION_NAMES  # noqa: E402


class Encoder:
    provenance = {"provider": "test-only", "encoder_sha256": "fixture"}

    def encode(self, inputs):
        # A call must contain exactly one complete episode, even with split rowgroups.
        if inputs.shape != (3, 1751):
            raise ValueError("encoder did not receive one complete episode")
        return np.tile(np.arange(64, dtype=np.float32) / 16, (len(inputs), 1))


def numerical_stats_module():
    """Execute actual native statistics definitions without optional image/torch imports.

    The standalone NumPy/Arrow test environment lacks LeRobot dependencies. The
    full environment uses the ordinary import, including integration checks.
    """
    try:
        from lerobot.datasets import compute_stats

        return compute_stats
    except ImportError:
        tree = ast.parse((ROOT / "src/lerobot/datasets/compute_stats.py").read_text())
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        module = types.ModuleType("lerobot.datasets.compute_stats")
        module.__dict__.update(np=np, logging=__import__("logging"))
        exec(
            compile(
                tree, "compute_stats.py", "exec", flags=__import__("__future__").annotations.compiler_flag
            ),
            module.__dict__,
        )
        return module


class SonicDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / "source"
        self.output = self.base / "converted"
        for folder in ("meta/episodes/chunk-000", "data/chunk-000", "videos"):
            (self.source / folder).mkdir(parents=True)
        features = {
            key: {"dtype": "float32", "shape": [28], "names": list(EXPECTED_JOINT_ORDER)}
            for key in ("action", "observation.state")
        }
        self.info = {
            "codebase_version": "v3.0",
            "fps": 30,
            "total_episodes": 2,
            "total_frames": 6,
            "total_tasks": 1,
            "features": features,
            "splits": {"train": "0:2"},
        }
        (self.source / "meta/info.json").write_text(json.dumps(self.info))
        (self.source / "meta/provenance.json").write_text('{"datasets": [{"revision": "abc"}]}')
        stats = {"min": [0.0] * 28, "max": [1.0] * 28, "mean": [0.0] * 28, "std": [1.0] * 28, "count": [6]}
        (self.source / "meta/stats.json").write_text(
            json.dumps({"action": stats, "observation.state": stats})
        )
        pq.write_table(pa.table({"task_index": [0], "task": ["pick"]}), self.source / "meta/tasks.parquet")
        self.meta_path = Path("meta/episodes/chunk-000/file-000.parquet")
        rows = [
            {
                "episode_index": ep,
                "length": 3,
                "dataset_from_index": ep * 3,
                "dataset_to_index": ep * 3 + 3,
                "tasks": ["pick"],
                **{f"stats/action/{k}": v if k != "count" else [3] for k, v in stats.items()},
                "stats/observation.state/mean": [123.0] * 28,
            }
            for ep in range(2)
        ]
        pq.write_table(pa.Table.from_pylist(rows), self.source / self.meta_path)
        actions = np.tile(np.arange(28, dtype=np.float32) / 100, (6, 1))
        actions[3:] += 0.5
        self.data_path = Path("data/chunk-000/file-000.parquet")
        table = pa.table(
            {
                "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), 28)),
                "observation.state": actions.tolist(),
                "episode_index": [0, 0, 0, 1, 1, 1],
                "frame_index": [0, 1, 2] * 2,
                "index": list(range(6)),
                "timestamp": [0.0, 1 / 30, 2 / 30] * 2,
                "task_index": [0] * 6,
            }
        )
        hf = {
            "info": {
                "features": {
                    "action": {
                        "feature": {"dtype": "float32", "_type": "Value"},
                        "length": 28,
                        "_type": "List",
                    }
                }
            }
        }
        table = table.replace_schema_metadata({b"huggingface": json.dumps(hf).encode(), b"custom": b"keep"})
        pq.write_table(table, self.source / self.data_path, row_group_size=2)
        self.xml = self.base / "robot.xml"
        self.xml.write_text(
            '<mujoco><compiler angle="radian"/><worldbody>'
            + "".join(f'<joint name="{name}" range="-2 2"/>' for name in ACTION_NAMES)
            + "</worldbody></mujoco>"
        )
        self.stats_patch = patch.dict(
            sys.modules, {"lerobot.datasets.compute_stats": numerical_stats_module()}
        )
        self.stats_patch.start()
        self.addCleanup(self.stats_patch.stop)

    def run_conversion(self, encoder=None):
        import prepare_sonic_dataset

        return prepare_sonic_dataset.prepare(self.source, self.output, self.xml, encoder=encoder or Encoder())

    def test_preserves_all_rows_and_recomputes_native_statistics(self):
        before = {p.relative_to(self.source): p.read_bytes() for p in self.source.rglob("*") if p.is_file()}
        manifest = self.run_conversion()
        original = pq.read_table(self.source / self.data_path)
        converted = pq.read_table(self.output / self.data_path)
        self.assertEqual(converted.schema.field("action").type.list_size, 78)
        self.assertTrue(original.drop(["action"]).equals(converted.drop(["action"])))
        values = np.asarray(converted["action"].to_pylist())
        np.testing.assert_allclose(values[:, 64:], np.asarray(original["action"].to_pylist())[:, 14:])
        self.assertEqual(pq.ParquetFile(self.output / self.data_path).num_row_groups, 2)
        info = json.loads((self.output / "meta/info.json").read_text())
        self.assertEqual(info["features"]["action"]["shape"], [78])
        self.assertEqual(info["features"]["action"]["names"][64:], list(EXPECTED_JOINT_ORDER[14:]))
        self.assertEqual(info["features"]["observation.state"], self.info["features"]["observation.state"])
        hf = json.loads(converted.schema.metadata[b"huggingface"])
        self.assertEqual(hf["info"]["features"]["action"]["length"], 78)
        self.assertEqual(converted.schema.metadata[b"custom"], b"keep")
        stats = json.loads((self.output / "meta/stats.json").read_text())
        np.testing.assert_allclose(stats["action"]["mean"], values.mean(axis=0), atol=1e-6)
        np.testing.assert_allclose(stats["action"]["std"], values.std(axis=0), atol=1e-6)
        self.assertEqual(stats["action"]["count"], [6])
        self.assertEqual(len(stats["action"]["q01"]), 78)
        rows = pq.read_table(self.output / self.meta_path).to_pylist()
        originals = pq.read_table(self.source / self.meta_path).to_pylist()
        for ep, row in enumerate(rows):
            np.testing.assert_allclose(row["stats/action/mean"], values[ep * 3 : ep * 3 + 3].mean(axis=0))
            self.assertEqual(row["stats/observation.state/mean"], [123.0] * 28)
            self.assertEqual(
                {k: v for k, v in row.items() if not k.startswith("stats/action/")},
                {k: v for k, v in originals[ep].items() if not k.startswith("stats/action/")},
            )
            for key, value in row.items():
                if key.startswith("stats/action/"):
                    self.assertEqual(len(value), 1 if key.endswith("/count") else 78)
        self.assertTrue((self.output / "videos").is_symlink())
        self.assertEqual(manifest["source_total_frames"], 6)
        self.assertEqual(manifest["total_episodes"], 2)
        self.assertEqual(manifest["source_provenance"]["datasets"][0]["revision"], "abc")
        self.assertEqual(manifest["robot_xml_sha256"], hashlib.sha256(self.xml.read_bytes()).hexdigest())
        self.assertEqual(
            manifest["source_to_encoder_joint_mapping"][0],
            {
                "index": 0,
                "source_name": "kLeftShoulderPitch",
                "encoder_xml_name": "left_shoulder_pitch_joint",
            },
        )
        self.assertEqual(
            manifest["source_to_encoder_joint_mapping"][17],
            {"index": 17, "source_name": "kLeftHandMiddle0", "encoder_xml_name": "left_hand_middle_0_joint"},
        )
        self.assertEqual(len(manifest["source_to_encoder_joint_mapping"]), 28)
        self.assertEqual(
            before,
            {p.relative_to(self.source): p.read_bytes() for p in self.source.rglob("*") if p.is_file()},
        )

    def test_bad_tokens_never_publish_final_output(self):
        class InvalidEncoder(Encoder):
            def encode(self, inputs):
                return np.full((len(inputs), 64), 0.123, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "grid"):
            self.run_conversion(InvalidEncoder())
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(ValueError, "incomplete|exists"):
            self.run_conversion()

    def test_missing_rows_never_publish(self):
        path = self.source / self.data_path
        pq.write_table(pq.read_table(path).slice(0, 5), path)
        with self.assertRaisesRegex(ValueError, "length|count|rows"):
            self.run_conversion()
        self.assertFalse(self.output.exists())

    def test_nonfinite_tokens_never_publish(self):
        class InvalidEncoder(Encoder):
            def encode(self, inputs):
                values = super().encode(inputs)
                values[-1, -1] = np.nan
                return values

        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.run_conversion(InvalidEncoder())
        self.assertFalse(self.output.exists())

    def test_existing_output_is_not_overwritten(self):
        self.output.mkdir()
        marker = self.output / "marker"
        marker.write_text("preserve")
        with self.assertRaisesRegex(ValueError, "exists"):
            self.run_conversion()
        self.assertEqual(marker.read_text(), "preserve")

    def test_scrambled_source_hand_order_rejected(self):
        path = self.source / "meta/info.json"
        info = json.loads(path.read_text())
        names = info["features"]["action"]["names"]
        names[17], names[19] = names[19], names[17]
        path.write_text(json.dumps(info))
        with self.assertRaisesRegex(ValueError, "source order"):
            self.run_conversion()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
