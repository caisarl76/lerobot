import hashlib
import json
from pathlib import Path

import pytest

from examples.g1_dex3_training.prune_smoke import POLICIES, SPACES, prune_superseded_smoke


def _fixture(tmp_path: Path, policy="act", space="joint28"):
    root = tmp_path.resolve()
    full_name = f"{policy}_{space}_full"
    full = root / "runs" / full_name
    smoke = root / "runs" / f"{policy}_{space}_smoke"
    full_model = full / "checkpoints/040000/pretrained_model"
    smoke_model = smoke / "checkpoints/000020/pretrained_model"
    full_model.mkdir(parents=True)
    smoke_model.mkdir(parents=True)
    weights = full_model / "model.safetensors"
    weights.write_bytes(b"full")
    (smoke_model / "placeholder").write_text("ok")
    (full / "checkpoints/last").symlink_to("040000")
    (smoke / "checkpoints/last").symlink_to("000020")
    (root / "logs").mkdir()
    for marker in (f"{full_name}.exit", f"{full_name}-verify.exit"):
        (root / "logs" / marker).write_text("0\n")
    (full / "final_model_validation.json").write_text(
        json.dumps(
            {
                "verified": True,
                "final_step": 40000,
                "policy_type": policy,
                "predicted_shape": [1, 4, SPACES[space]],
                "checkpoint_path": str(full / "checkpoints/040000"),
                "weights_sha256": hashlib.sha256(b"full").hexdigest(),
            }
        )
    )
    (smoke / "final_model_validation.json").write_text(
        json.dumps(
            {
                "verified": True,
                "final_step": 20,
                "policy_type": policy,
                "predicted_shape": [1, 4, SPACES[space]],
                "checkpoint_path": str(smoke / "checkpoints/000020"),
            }
        )
    )
    return root, full, smoke


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("space", SPACES)
def test_cleanup_retains_full_and_report(tmp_path, policy, space):
    root, full, smoke = _fixture(tmp_path, policy, space)
    name = f"{policy}_{space}_full"
    prune_superseded_smoke(root, name)
    assert not (smoke / "checkpoints").exists()
    assert (smoke / "final_model_validation.json").is_file()
    assert (full / "checkpoints/040000/pretrained_model/model.safetensors").is_file()
    prune_superseded_smoke(root, name)


def test_invalid_full_proof_refuses_cleanup(tmp_path):
    root, _, smoke = _fixture(tmp_path)
    report = json.loads((root / "runs/act_joint28_full/final_model_validation.json").read_text())
    report["weights_sha256"] = "bad"
    (root / "runs/act_joint28_full/final_model_validation.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Full-run"):
        prune_superseded_smoke(root, "act_joint28_full")
    assert (smoke / "checkpoints").is_dir()


def test_malformed_smoke_path_refuses_cleanup(tmp_path):
    root, _, smoke = _fixture(tmp_path)
    (smoke / "checkpoints/last").unlink()
    (smoke / "checkpoints/last").symlink_to("/tmp")
    with pytest.raises(ValueError, match="Smoke checkpoints/last"):
        prune_superseded_smoke(root, "act_joint28_full")


def test_full_weights_symlink_refuses_cleanup(tmp_path):
    root, full, smoke = _fixture(tmp_path)
    model = full / "checkpoints/040000/pretrained_model/model.safetensors"
    external = root / "external_weights"
    model.rename(external)
    model.symlink_to(external)
    with pytest.raises(ValueError, match="Full-run"):
        prune_superseded_smoke(root, "act_joint28_full")
    assert (smoke / "checkpoints").is_dir()
