# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cross-embodiment fine-tuning must not inherit the base checkpoint's gripper steps."""

import json
from copy import deepcopy

import pytest
import torch
from safetensors.torch import save_file

from conftest import make_config, make_train_batch
from lerobot.configs.types import NormalizationMode
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.vla_jepa.processor_vla_jepa import (
    BinarizeGripperProcessorStep,
    ImagePrepProcessorStep,
    PreSnapGripperProcessorStep,
    make_vla_jepa_pre_post_processors,
)
from lerobot.processor import AbsoluteActionsProcessorStep, RelativeActionsProcessorStep
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def _stats(action_dim, state_dim):
    return {
        key: {
            "min": torch.full((dim,), -2.0),
            "max": torch.full((dim,), 2.0),
            "mean": torch.full((dim,), 0.1),
            "std": torch.full((dim,), 2.0),
        }
        for key, dim in ((ACTION, action_dim), (OBS_STATE, state_dim))
    }


@pytest.fixture
def base_checkpoint(tmp_path):
    cfg = make_config(action_dim=7, state_dim=7)
    cfg.pre_snap_gripper_action = True
    cfg.binarize_gripper_action = True
    pre, post = make_vla_jepa_pre_post_processors(cfg, _stats(7, 7))
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    path = tmp_path / "policy_postprocessor.json"
    saved = json.loads(path.read_text())
    for step in saved["steps"]:
        if step["registry_name"] in ("vla_jepa_pre_snap_gripper", "vla_jepa_binarize_gripper"):
            # The published pretraining checkpoint predates gripper get_config().
            step["config"] = {}
    path.write_text(json.dumps(saved))
    return tmp_path


@pytest.mark.parametrize("action_dim, relative", [(28, False), (78, False), (28, True)])
def test_g1_finetune_rebuilds_processors_and_resume_preserves_them(
    base_checkpoint, tmp_path, action_dim, relative
):
    cfg = make_config(action_dim=action_dim, state_dim=28)
    cfg.pre_snap_gripper_action = False
    cfg.binarize_gripper_action = False
    cfg.gripper_joint_names = []
    cfg.clip_normalized_actions = False
    cfg.use_relative_actions = relative
    cfg.resize_images_to = (4, 4)
    cfg.normalization_mapping["ACTION"] = NormalizationMode.MEAN_STD
    cfg.normalization_mapping["STATE"] = NormalizationMode.MEAN_STD
    stats = _stats(action_dim, 28)
    image_key = f"{OBS_IMAGES}.laptop"
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=str(base_checkpoint),
        dataset_stats=stats,
        preprocessor_overrides={
            "rename_observations_processor": {"rename_map": {"observation.images.raw": image_key}},
            "device_processor": {"device": "cpu", "float_dtype": "float64"},
            "normalizer_processor": {
                "features": {**cfg.input_features, **cfg.output_features},
                "norm_map": cfg.normalization_mapping,
                "stats": stats,
            },
        },
        postprocessor_overrides={
            "device_processor": {"device": "cpu", "float_dtype": "float64"},
            "unnormalizer_processor": {
                "features": cfg.output_features,
                "norm_map": cfg.normalization_mapping,
                "stats": stats,
            },
        },
    )
    assert not any(
        isinstance(s, (PreSnapGripperProcessorStep, BinarizeGripperProcessorStep)) for s in post.steps
    )
    assert next(s for s in pre.steps if isinstance(s, ImagePrepProcessorStep)).resize_to == (4, 4)
    relative_step = next(s for s in pre.steps if isinstance(s, RelativeActionsProcessorStep))
    assert (
        next(s for s in post.steps if isinstance(s, AbsoluteActionsProcessorStep)).relative_step
        is relative_step
    )

    actions = torch.linspace(-0.7, 0.7, action_dim).reshape(1, 1, action_dim).expand(2, 4, action_dim).clone()
    batch = {
        "observation.images.raw": torch.ones(2, 3, 8, 8),
        OBS_STATE: torch.full((2, 28), 0.25),
        ACTION: actions,
    }
    processed = pre(deepcopy(batch))
    assert processed[image_key].shape == (2, 3, 4, 4)
    assert processed[OBS_STATE].shape == (2, 28)
    assert processed[ACTION].shape == (2, 4, action_dim)
    assert processed[ACTION].dtype == torch.float64
    expected = (actions.double() - (0.25 if relative else 0.0) - 0.1) / 2.0
    torch.testing.assert_close(processed[ACTION], expected)
    torch.testing.assert_close(post(processed[ACTION]), actions.double())

    trained = tmp_path / "trained"
    pre.save_pretrained(trained)
    post.save_pretrained(trained)
    (trained / "policy_preprocessor.json").rename(trained / "custom_pre.json")
    (trained / "policy_postprocessor.json").rename(trained / "custom_post.json")
    # Saved topology and stats win on resume even if the caller's config disagrees.
    cfg.pre_snap_gripper_action = True
    cfg.binarize_gripper_action = True
    cfg.resize_images_to = (2, 2)
    resumed_pre, resumed_post = make_pre_post_processors(
        cfg,
        pretrained_path=str(trained),
        preprocessor_config_filename="custom_pre.json",
        postprocessor_config_filename="custom_post.json",
        preprocessor_overrides={"device_processor": {"device": "cpu", "float_dtype": "float64"}},
    )
    resumed = resumed_pre(deepcopy(batch))
    torch.testing.assert_close(resumed[ACTION], processed[ACTION])
    assert resumed[image_key].shape == (2, 3, 4, 4)
    torch.testing.assert_close(resumed_post(resumed[ACTION]), actions.double())


def test_inference_preserves_legacy_gripper_steps(base_checkpoint):
    cfg = make_config(action_dim=7, state_dim=7)
    cfg.pre_snap_gripper_action = False
    cfg.binarize_gripper_action = False
    _, post = make_pre_post_processors(cfg, pretrained_path=str(base_checkpoint))
    assert any(isinstance(s, PreSnapGripperProcessorStep) and s.gripper_dim == 6 for s in post.steps)
    assert any(isinstance(s, BinarizeGripperProcessorStep) and s.gripper_dim == 6 for s in post.steps)


@pytest.mark.parametrize("action_dim", [28, 78])
def test_finetune_uses_dataset_stats_without_normalizer_overrides(base_checkpoint, action_dim):
    cfg = make_config(action_dim=action_dim, state_dim=28)
    cfg.pre_snap_gripper_action = False
    cfg.binarize_gripper_action = False
    cfg.clip_normalized_actions = False
    cfg.normalization_mapping["ACTION"] = NormalizationMode.MEAN_STD
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=str(base_checkpoint), dataset_stats=_stats(action_dim, 28)
    )
    actions = torch.full((2, 4, action_dim), 0.7)
    processed = pre({OBS_STATE: torch.zeros(2, 28), ACTION: actions})
    torch.testing.assert_close(processed[ACTION], torch.full_like(actions, 0.3))
    torch.testing.assert_close(post(processed[ACTION]), actions)


@pytest.mark.parametrize("reinitialize_head", [False, True])
def test_pretrained_loading_defaults_to_strict_backbone(tmp_path, monkeypatch, reinitialize_head):
    pytest.importorskip("transformers")
    pytest.importorskip("diffusers")
    from lerobot.policies.vla_jepa import modeling_vla_jepa

    class TinyModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.head = torch.nn.Linear(2, config.action_dim)

    monkeypatch.setattr(modeling_vla_jepa, "VLAJEPAModel", TinyModel)
    cfg = make_config(action_dim=28, state_dim=28)
    cfg.reinit_modules = ["model.head."] if reinitialize_head else []
    policy = modeling_vla_jepa.VLAJEPAPolicy(cfg)
    weights = policy.state_dict()
    if reinitialize_head:
        weights["model.head.weight"] = torch.ones(7, 2)
        weights["model.head.bias"] = torch.ones(7)
    weights.pop("model.backbone.weight")
    save_file(weights, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="model.backbone.weight"):
        modeling_vla_jepa.VLAJEPAPolicy.from_pretrained(tmp_path, config=cfg)
    # Preserve the explicit opt-out API.
    modeling_vla_jepa.VLAJEPAPolicy.from_pretrained(tmp_path, config=cfg, strict=False)

    if reinitialize_head:
        weights["model.backbone.weight"] = torch.full((2, 2), 0.75)
        save_file(weights, tmp_path / "model.safetensors")
        loaded = modeling_vla_jepa.VLAJEPAPolicy.from_pretrained(tmp_path, config=cfg)
        torch.testing.assert_close(loaded.model.backbone.weight, weights["model.backbone.weight"])
        assert loaded.model.head.weight.shape == (28, 2)

        weights["model.backbone.weight"] = torch.ones(3, 2)
        save_file(weights, tmp_path / "model.safetensors")
        with pytest.raises(ValueError, match="prefix is not in `reinit_modules`"):
            modeling_vla_jepa.VLAJEPAPolicy.from_pretrained(tmp_path, config=cfg)


@pytest.mark.parametrize("reinit_modules", [None, [], ["model.action_model."]])
def test_legacy_predictor_weights_preserve_strict_loading_and_world_loss(
    tmp_path, caplog, patch_vla_jepa_external_models, reinit_modules
):
    from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy

    cfg = make_config(action_dim=28, state_dim=28)
    cfg.enable_world_model = True
    cfg.world_model_loss_weight = 0.1
    cfg.reinit_modules = reinit_modules
    policy = VLAJEPAPolicy(cfg)
    weights = {key: value.clone() for key, value in policy.state_dict().items()}
    legacy_keys = {
        f"model.video_predictor.{encoder}.{parameter}"
        for encoder in ("state_encoder", "extrinsics_encoder")
        for parameter in ("weight", "bias")
    }
    weights.update({key: torch.zeros(1) for key in legacy_keys})
    save_file(weights, tmp_path / "model.safetensors")

    loaded = VLAJEPAPolicy.from_pretrained(tmp_path, config=cfg)
    assert "Ignoring obsolete unused VLA-JEPA video predictor weights" in caplog.text
    assert all(key in caplog.text for key in legacy_keys)
    assert not legacy_keys.intersection(loaded.state_dict())
    torch.testing.assert_close(
        loaded.model.video_predictor.action_encoder.weight,
        weights["model.video_predictor.action_encoder.weight"],
    )
    loaded.train()
    loss, logs = loaded(make_train_batch(action_dim=28, state_dim=28))
    assert torch.isfinite(loss)
    assert logs["wm_loss"] > 0
    loss.backward()
    predictor_grad = loaded.model.video_predictor.predictor_proj.weight.grad
    assert predictor_grad is not None and torch.isfinite(predictor_grad).all()
    assert predictor_grad.abs().sum() > 0

    # A similar prefix is not sufficient: only the four exact obsolete keys are ignored.
    unexpected_key = "model.video_predictor.state_encoder.unexpected"
    weights[unexpected_key] = torch.zeros(1)
    save_file(weights, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match=unexpected_key):
        VLAJEPAPolicy.from_pretrained(tmp_path, config=cfg)

    del weights[unexpected_key]
    del weights["model.video_predictor.action_encoder.weight"]
    save_file(weights, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="model.video_predictor.action_encoder.weight"):
        VLAJEPAPolicy.from_pretrained(tmp_path, config=cfg)
