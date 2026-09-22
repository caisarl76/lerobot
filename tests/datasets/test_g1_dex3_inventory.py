import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
from inventory import EXPECTED_JOINT_ORDER, REPO_IDS, SUFFIXES, InventoryError, build_inventory


class InventoryTests(unittest.TestCase):
    def make_root(self, base, suffix, *, fps=30, dim=28, names=None):
        root = base / f"G1_Dex3_{suffix}_Dataset"
        (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (root / "videos" / "observation.images.front" / "chunk-000").mkdir(parents=True, exist_ok=True)
        features = {
            "action": {
                "dtype": "float32",
                "shape": [dim],
                "names": [names or list(EXPECTED_JOINT_ORDER)[:dim]],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [dim],
                "names": names or list(EXPECTED_JOINT_ORDER)[:dim],
            },
            "observation.images.front": {
                "dtype": "video",
                "shape": [3, 224, 224],
                "names": ["channels", "height", "width"],
            },
        }
        (root / "meta" / "info.json").write_text(
            json.dumps(
                {
                    "codebase_version": "v3.0",
                    "fps": fps,
                    "total_episodes": 2,
                    "total_frames": 10,
                    "features": features,
                }
            )
        )
        (root / "meta" / "tasks.parquet").write_bytes(b"task-metadata-only-fixture")
        (root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").write_bytes(
            b"episode-metadata-only-fixture"
        )
        (root / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"parquet-metadata-only-fixture")
        (root / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4").write_bytes(b"video")

    def test_all13_accepted_and_summarized(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            for suffix in SUFFIXES:
                self.make_root(base, suffix)
            result = build_inventory(base)
            self.assertEqual(result["repo_ids"], list(REPO_IDS))
            self.assertEqual(result["total_episodes"], 26)
            self.assertEqual(result["total_frames"], 130)
            self.assertEqual(result["validation_level"], "metadata-only")
            self.assertEqual(result["datasets"][0]["video_file_count"], 1)
            self.assertEqual(result["datasets"][0]["data_file_count"], 1)
            self.assertEqual(result["datasets"][0]["codebase_version"], "v3.0")
            self.assertIn("observation.images.front", result["datasets"][0]["camera_features"])

    def test_missing_dataset_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            for suffix in SUFFIXES[:-1]:
                self.make_root(base, suffix)
            with self.assertRaises(InventoryError):
                build_inventory(base)

    def test_mismatched_schema_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            for suffix in SUFFIXES:
                self.make_root(base, suffix)
            self.make_root(base, SUFFIXES[1], dim=27)
            with self.assertRaises(InventoryError):
                build_inventory(base)

    def test_mismatched_joint_order_and_fps_rejected(self):
        for kwargs in ({"names": [f"x{i}" for i in range(28)]}, {"fps": 60}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as td:
                base = Path(td)
                for suffix in SUFFIXES:
                    self.make_root(base, suffix)
                self.make_root(base, SUFFIXES[1], **kwargs)
                with self.assertRaises(InventoryError):
                    build_inventory(base)

    def test_unrelated_onecam_directories_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            for suffix in SUFFIXES:
                self.make_root(base, suffix)
            (base / "G1_Dex3_PickApple_Dataset_1cam").mkdir()
            self.assertEqual(len(build_inventory(base)["datasets"]), 13)

    def test_camera_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            for suffix in SUFFIXES:
                self.make_root(base, suffix)
            info_path = base / f"G1_Dex3_{SUFFIXES[-1]}_Dataset" / "meta" / "info.json"
            info = json.loads(info_path.read_text())
            info["features"]["observation.images.front"]["shape"] = [3, 480, 640]
            info_path.write_text(json.dumps(info))
            with self.assertRaisesRegex(InventoryError, "camera schema"):
                build_inventory(base)

    def test_nonfinite_fps_rejected_even_when_first_dataset(self):
        for fps in (float("nan"), float("inf")):
            with self.subTest(fps=fps), tempfile.TemporaryDirectory() as td:
                base = Path(td)
                for suffix in SUFFIXES:
                    self.make_root(base, suffix, fps=fps)
                with self.assertRaises(InventoryError):
                    build_inventory(base)

    def test_shared_wrong_joint_order_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            for suffix in SUFFIXES:
                self.make_root(base, suffix, names=list(reversed(EXPECTED_JOINT_ORDER)))
            with self.assertRaisesRegex(InventoryError, "joint order"):
                build_inventory(base)

    def test_explicit_camera_selection_normalizes_layout_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            for suffix in SUFFIXES:
                self.make_root(base, suffix)
            path = base / f"G1_Dex3_{SUFFIXES[0]}_Dataset" / "meta" / "info.json"
            info = json.loads(path.read_text())
            info["features"]["observation.images.front"]["shape"] = [224, 224, 3]
            # The published picking data has HWC shapes but labels them CHW.
            info["features"]["observation.images.front"]["names"] = ["channels", "height", "width"]
            info["features"]["observation.images.wrist"] = dict(info["features"]["observation.images.front"])
            path.write_text(json.dumps(info))
            original = path.read_bytes()
            result = build_inventory(base, camera_keys=("observation.images.front",))
            self.assertEqual(result["total_frames"], 130)
            self.assertEqual(
                result["datasets"][0]["camera_features"]["observation.images.front"]["shape"], [3, 224, 224]
            )
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
