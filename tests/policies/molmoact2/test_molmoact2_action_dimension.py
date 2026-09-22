"""Action projection adaptation without downloading a pretrained backbone."""

# ruff: noqa: E402

import copy
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("scipy")

from safetensors.torch import save_file

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.molmoact2 import modeling_molmoact2 as modeling
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2PackInputsProcessorStep
from lerobot.utils.constants import ACTION


class TinyModel(torch.nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.config = SimpleNamespace(
            max_action_dim=width,
            max_action_horizon=30,
            action_mode="both",
            add_action_expert=True,
            action_expert_config=SimpleNamespace(max_action_dim=width, hidden_size=8),
        )
        self.model = torch.nn.Module()
        self.model.config = copy.deepcopy(self.config)
        self.model.vision_backbone = torch.nn.Linear(4, 8)
        self.model.transformer = torch.nn.Linear(8, 8)
        expert = torch.nn.Module()
        expert.config = copy.deepcopy(self.config.action_expert_config)
        expert.hidden_size = 8
        expert.action_embed = torch.nn.Linear(width, 8)
        expert.block = torch.nn.Linear(8, 8)
        expert.final_layer = torch.nn.Module()
        expert.final_layer.linear = torch.nn.Linear(8, width)
        self.model.action_expert = expert


def make_policy(width=78, *, adapt=True):
    policy = object.__new__(modeling.MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = MolmoAct2Config(
        expected_max_action_dim=width,
        adapt_action_projections=adapt,
        train_mode_vlm="fft",
        freeze_embedding=False,
        dtype="float32",
        device="cpu",
        input_features={"observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 8, 8))},
    )
    return policy


def install_checkpoint(monkeypatch, tmp_path, source):
    save_file(source.state_dict(), tmp_path / "model.safetensors")
    monkeypatch.setattr(modeling, "_resolve_checkpoint_location", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(
        modeling,
        "HFMolmoAct2Config",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: source.config),
    )
    monkeypatch.setattr(
        modeling,
        "MolmoAct2ForConditionalGeneration",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: copy.deepcopy(source)),
    )


def test_expansion_changes_only_two_projections(monkeypatch, tmp_path):
    source = TinyModel().double()
    original = {name: value.clone() for name, value in source.state_dict().items()}
    install_checkpoint(monkeypatch, tmp_path, source)
    policy = make_policy()
    policy._load_hf_model()
    expert = policy.model.model.action_expert
    assert expert.action_embed.weight.shape == (8, 78)
    assert expert.final_layer.linear.weight.shape == (78, 8)
    assert expert.action_embed.weight.dtype == torch.float64
    assert expert.action_embed.weight.device == source.model.action_expert.action_embed.weight.device
    assert torch.count_nonzero(expert.action_embed.weight) > 0
    assert torch.count_nonzero(expert.action_embed.bias) == 0
    assert torch.count_nonzero(expert.final_layer.linear.weight) == 0
    assert torch.count_nonzero(expert.final_layer.linear.bias) == 0
    for name, value in policy.model.state_dict().items():
        if ".action_embed." not in name and ".final_layer.linear." not in name:
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    for config in (policy.model.config, policy.model.model.config):
        assert config.max_action_dim == 78
        assert config.action_expert_config.max_action_dim == 78
    assert expert.config.max_action_dim == 78
    expert.final_layer.linear(
        expert.action_embed(torch.randn(2, 30, 78, dtype=torch.float64))
    ).sum().backward()
    assert expert.final_layer.linear.weight.grad.shape == (78, 8)


@pytest.mark.parametrize(("width", "adapt"), [(32, False), (32, True), (78, True)])
def test_matching_checkpoint_preserves_trained_weights(monkeypatch, tmp_path, width, adapt):
    source = TinyModel(width)
    install_checkpoint(monkeypatch, tmp_path, source)
    policy = make_policy(width, adapt=adapt)
    policy._load_hf_model()
    for name, value in policy.model.state_dict().items():
        torch.testing.assert_close(value, source.state_dict()[name], rtol=0, atol=0)


def test_saved_78_policy_weights_load_after_base_expansion(monkeypatch, tmp_path):
    install_checkpoint(monkeypatch, tmp_path, TinyModel())
    first = make_policy()
    first._load_hf_model()
    with torch.no_grad():
        first.model.model.action_expert.action_embed.weight.fill_(0.123)
        first.model.model.action_expert.final_layer.linear.weight.fill_(0.456)
    saved_policy = tmp_path / "trained_policy"
    first.save_pretrained(saved_policy)
    reloaded = modeling.MolmoAct2Policy.from_pretrained(saved_policy, local_files_only=True, strict=True)
    for name, value in reloaded.state_dict().items():
        torch.testing.assert_close(value, first.state_dict()[name], rtol=0, atol=0)


def test_expansion_does_not_bypass_strict_checkpoint_loading(monkeypatch, tmp_path):
    source = TinyModel()
    install_checkpoint(monkeypatch, tmp_path, source)
    weights = source.state_dict()
    del weights["model.transformer.weight"]
    save_file(weights, tmp_path / "model.safetensors")
    policy = make_policy()
    with pytest.raises(RuntimeError, match="Missing key"):
        policy._load_hf_model()
    assert policy.model.model.action_expert.action_embed.in_features == 32


@pytest.mark.parametrize(
    "malformation", ["unknown_width", "output_shape", "expert_config", "backbone_config"]
)
def test_expansion_rejects_malformed_source(monkeypatch, tmp_path, malformation):
    source = TinyModel(64 if malformation == "unknown_width" else 32)
    if malformation == "output_shape":
        source.model.action_expert.final_layer.linear = torch.nn.Linear(8, 31)
    elif malformation == "expert_config":
        source.model.action_expert.config.max_action_dim = 31
    elif malformation == "backbone_config":
        source.model.config.max_action_dim = 31
    install_checkpoint(monkeypatch, tmp_path, source)
    with pytest.raises(ValueError, match="action"):
        make_policy()._load_hf_model()


def test_nonstandard_dimension_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="adapt_action_projections"):
        MolmoAct2Config(expected_max_action_dim=78)
    with pytest.raises(ValueError, match="78"):
        MolmoAct2Config(expected_max_action_dim=64, adapt_action_projections=True)


@pytest.mark.parametrize("mode", ["both", "discrete"])
def test_adaptation_requires_continuous_mode(mode):
    with pytest.raises(ValueError, match="continuous"):
        MolmoAct2Config(
            expected_max_action_dim=78,
            adapt_action_projections=True,
            action_mode=mode,
            inference_action_mode="discrete",
        )


def test_default_configuration_rejects_78_checkpoint(monkeypatch, tmp_path):
    install_checkpoint(monkeypatch, tmp_path, TinyModel(78))
    with pytest.raises(ValueError, match="max_action_dim"):
        make_policy(32, adapt=False)._load_hf_model()


def test_config_rejects_features_larger_than_action_width():
    config = make_policy().config
    config.input_features = {"observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 8, 8))}
    config.output_features = {ACTION: PolicyFeature(FeatureType.ACTION, (79,))}
    with pytest.raises(ValueError, match="action"):
        config.validate_features()


def test_processor_keeps_all_78_action_coordinates():
    step = object.__new__(MolmoAct2PackInputsProcessorStep)
    step.max_action_dim = 78
    action = torch.arange(78, dtype=torch.float32).expand(2, 30, 78)
    padded, horizon_mask, dimension_mask = step._pad_action(action)
    torch.testing.assert_close(padded, action)
    assert horizon_mask.shape == (2, 30)
    assert dimension_mask.shape == (2, 78)
    assert not dimension_mask.any()
    with pytest.raises(ValueError, match="exceeds"):
        step._pad_action(torch.zeros(2, 30, 79))
