"""Filesystem safety tests; run with python -m unittest tests.test_g1_dex3_finalize."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


class FinalizeTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "examples/g1_dex3_training/finalize_baseline.py"
        self.assertTrue(path.exists(), "checkpoint finalizer must exist")
        spec = importlib.util.spec_from_file_location("finalize_baseline", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.allowed = self.root / "runs"
        self.run = self.allowed / "act_joint28_full"
        self.final = self.run / "checkpoints/000020"
        self.model = self.final / "pretrained_model"
        self.model.mkdir(parents=True)
        self.old = self.run / "checkpoints/000010"
        (self.old / "pretrained_model").mkdir(parents=True)
        (self.old / "pretrained_model/model.safetensors").write_bytes(b"old")
        (self.final / "training_state").mkdir()
        (self.final / "training_state/training_step.json").write_text('{"step": 20}')
        (self.run / "checkpoints/last").symlink_to("000020", target_is_directory=True)
        (self.root / "logs").mkdir()
        self.exit_path = self.root / "logs/act_joint28_full.exit"
        self.exit_path.write_text("0\n")
        for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
            (self.model / name).write_text("{}")
        self.weights = self.model / "model.safetensors"
        self.weights.write_bytes(b"final weights")
        (self.model / "policy_preprocessor_step_0.safetensors").write_bytes(b"stats")
        self.config = {"steps": 20, "output_dir": str(self.run)}
        self.write_config()
        self.report = {
            "verified": True,
            "checkpoint_path": str(self.final),
            "final_step": 20,
            "weights_sha256": hashlib.sha256(b"final weights").hexdigest(),
            "weights_size_bytes": 13,
        }

    def write_config(self):
        (self.model / "train_config.json").write_text(json.dumps(self.config))

    def prune(self):
        return self.module.prune_verified_checkpoint(self.run, self.report, allowed_root=self.allowed)

    def assert_refused_without_deletion(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.prune()
        self.assertTrue(self.old.exists())
        self.assertTrue((self.final / "training_state").exists())
        self.assertEqual(self.weights.read_bytes(), b"final weights")

    def test_failed_training_exit_does_not_delete(self):
        self.exit_path.write_text("1\n")
        self.assert_refused_without_deletion()

    def test_missing_exit_marker_does_not_delete(self):
        self.exit_path.unlink()
        self.assert_refused_without_deletion()

    def test_incomplete_training_step_does_not_delete(self):
        (self.final / "training_state/training_step.json").write_text('{"step": 19}')
        self.assert_refused_without_deletion()

    def test_configured_final_step_must_match(self):
        self.config["steps"] = 30
        self.write_config()
        self.assert_refused_without_deletion()

    def test_escaping_last_cannot_delete_source(self):
        source = self.root / "source_dataset"
        source.mkdir()
        (source / "keep").write_text("source")
        last = self.run / "checkpoints/last"
        last.unlink()
        last.symlink_to(source, target_is_directory=True)
        self.assert_refused_without_deletion()
        self.assertEqual((source / "keep").read_text(), "source")

    def test_nested_symlink_or_unexpected_entry_refuses_all_deletion(self):
        for unexpected in (self.old / "pretrained_model/link", self.run / "checkpoints/notes"):
            unexpected.symlink_to(self.root / "logs", target_is_directory=True)
            self.assert_refused_without_deletion()
            unexpected.unlink()

    def test_saved_output_directory_must_match(self):
        self.config["output_dir"] = str(self.root / "another")
        self.write_config()
        self.assert_refused_without_deletion()

    def test_changed_weights_invalidate_validation(self):
        self.report["weights_sha256"] = "0" * 64
        self.assert_refused_without_deletion()

    def test_unverified_report_cannot_authorize_pruning(self):
        self.report["verified"] = False
        self.assert_refused_without_deletion()

    def test_missing_processor_prevents_pruning(self):
        (self.model / "policy_preprocessor.json").unlink()
        self.assert_refused_without_deletion()

    def test_real_directory_named_last_is_rejected(self):
        last = self.run / "checkpoints/last"
        last.unlink()
        last.mkdir()
        self.assert_refused_without_deletion()

    def test_report_write_is_atomic_on_nonfinite_results(self):
        destination = self.run / "final_model_validation.json"
        destination.write_text('{"previous": true}\n')
        with self.assertRaises(ValueError):
            self.module._write_report(self.run, {"loss": float("nan")})
        self.assertEqual(json.loads(destination.read_text()), {"previous": True})
        self.assertEqual({p.name for p in self.run.iterdir()}, {"checkpoints", destination.name})

    def test_success_keeps_final_weights_configs_processors_and_last(self):
        before = {p.name: p.read_bytes() for p in self.model.iterdir()}
        deleted = self.prune()
        self.assertEqual(set(deleted), {str(self.old), str(self.final / "training_state")})
        self.assertFalse(self.old.exists())
        self.assertFalse((self.final / "training_state").exists())
        self.assertEqual({p.name: p.read_bytes() for p in self.model.iterdir()}, before)
        self.assertEqual((self.run / "checkpoints/last").resolve(), self.final)
        self.assertEqual(self.exit_path.read_text(), "0\n")

    def test_cli_root_cannot_be_bypassed_with_nested_run(self):
        with self.assertRaises(ValueError):
            self.module.inspect_completed_checkpoint(self.run, allowed_root=self.root)

    def test_world_model_inference_uses_current_frame_and_diffusion_keeps_history(self):
        from types import SimpleNamespace

        import torch

        state = torch.arange(2 * 9 * 28).reshape(2, 9, 28)
        images = torch.arange(2 * 9 * 3 * 4 * 4).reshape(2, 9, 3, 4, 4)
        batch = {
            "observation.state": state,
            "observation.images.head": images,
            "observation.images.head_is_pad": torch.zeros(2, 9, dtype=torch.bool),
            "action": torch.zeros(2, 32, 28),
            "action_is_pad": torch.zeros(2, 32, dtype=torch.bool),
        }
        features = {
            "observation.state": SimpleNamespace(shape=(28,)),
            "observation.images.head": SimpleNamespace(shape=(3, 4, 4)),
        }
        for policy in ("fastwam", "vla_jepa", "diffusion"):
            cfg = SimpleNamespace(type=policy, input_features=features)
            observation = self.module._inference_observation(batch, cfg)
            self.assertNotIn("action", observation)
            self.assertNotIn("action_is_pad", observation)
            if policy == "diffusion":
                self.assertIs(observation["observation.state"], state)
                self.assertIs(observation["observation.images.head"], images)
            else:
                self.assertTrue(torch.equal(observation["observation.state"], state[:, 0]))
                self.assertTrue(torch.equal(observation["observation.images.head"], images[:, 0]))
                self.assertNotIn("observation.images.head_is_pad", observation)
        self.assertEqual(batch["observation.state"].shape, (2, 9, 28))


if __name__ == "__main__":
    unittest.main()
