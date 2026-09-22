# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging

import pytest
import torch
from torch import nn

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


class _TinyPolicy(nn.Module):
    """Exercise the production checkpoint loader without constructing the VLM."""

    from_pretrained = classmethod(PI05Policy.from_pretrained.__func__)
    _prepare_pretrained_state_dict = PI05Policy._prepare_pretrained_state_dict
    _fix_pytorch_state_dict_keys = PI05Policy._fix_pytorch_state_dict_keys

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = nn.Module()
        self.model.action_in_proj = nn.Linear(config.max_action_dim, 8)
        self.model.action_out_proj = nn.Linear(8, config.max_action_dim)
        self.model.backbone = nn.Linear(8, 8)
        if config.use_proprioceptive_memory:
            self.model.proprio_history_proj = nn.Linear(config.max_state_dim, 8)


def _policy(dim=78, *, adapt=False, memory=False):
    return _TinyPolicy(
        PI05Config(max_action_dim=dim, adapt_action_projections=adapt, use_proprioceptive_memory=memory)
    )


def _checkpoint(dim=32):
    policy = _policy(dim)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.fill_(7)
    return policy.state_dict()


def test_adaptation_reinitializes_both_heads_and_preserves_backbone(caplog):
    policy = _policy(adapt=True)
    fresh = {key: value.clone() for key, value in policy.state_dict().items()}
    source = _checkpoint()
    with caplog.at_level(logging.INFO):
        prepared = policy._prepare_pretrained_state_dict(source)
    policy.load_state_dict(prepared, strict=True)

    assert policy.model.action_in_proj.weight.shape == (8, 78)
    assert policy.model.action_out_proj.weight.shape == (78, 8)
    assert policy.model.action_out_proj.bias.shape == (78,)
    for key in (
        "model.action_in_proj.weight",
        "model.action_in_proj.bias",
        "model.action_out_proj.weight",
        "model.action_out_proj.bias",
    ):
        torch.testing.assert_close(policy.state_dict()[key], fresh[key])
        assert key in caplog.text
    torch.testing.assert_close(policy.model.backbone.weight, torch.full((8, 8), 7.0))
    torch.testing.assert_close(policy.model.backbone.bias, torch.full((8,), 7.0))


def test_action_dimension_adaptation_is_explicit():
    policy = _TinyPolicy(PI05Config(max_action_dim=78))
    with pytest.raises(RuntimeError, match="size mismatch"):
        policy.load_state_dict(policy._prepare_pretrained_state_dict(_checkpoint()), strict=True)


@pytest.mark.parametrize("dim", [32, 78])
def test_matching_action_dimensions_preserve_learned_heads(dim, caplog):
    policy = _policy(dim, adapt=True)
    source = _checkpoint(dim)
    with caplog.at_level(logging.INFO):
        policy.load_state_dict(policy._prepare_pretrained_state_dict(source), strict=True)
    for value in policy.state_dict().values():
        torch.testing.assert_close(value, torch.full_like(value, 7))
    assert "Reinitialized" not in caplog.text


@pytest.mark.parametrize(
    "key, shape",
    [
        ("model.action_in_proj.weight", (9, 32)),
        ("model.action_in_proj.weight", (8, 32, 1)),
        ("model.action_in_proj.bias", (9,)),
        ("model.action_out_proj.weight", (31, 8)),
        ("model.action_out_proj.weight", (32, 9)),
        ("model.action_out_proj.bias", (31,)),
    ],
)
def test_adaptation_rejects_malformed_projection_shapes(key, shape):
    source = _checkpoint()
    source[key] = torch.zeros(shape)
    with pytest.raises(ValueError, match="action projection"):
        _policy(adapt=True)._prepare_pretrained_state_dict(source)


@pytest.mark.parametrize("key", ["model.action_in_proj.bias", "model.action_out_proj.weight"])
def test_adaptation_rejects_incomplete_projection_pair(key):
    source = _checkpoint()
    del source[key]
    with pytest.raises(ValueError, match="action projection"):
        _policy(adapt=True)._prepare_pretrained_state_dict(source)


@pytest.mark.parametrize("missing", [False, True])
def test_adaptation_does_not_relax_backbone_loading(missing):
    policy = _policy(adapt=True)
    source = _checkpoint()
    if missing:
        del source["model.backbone.weight"]
    else:
        source["model.backbone.weight"] = torch.zeros(9, 8)
    with pytest.raises(RuntimeError):
        policy.load_state_dict(policy._prepare_pretrained_state_dict(source), strict=True)


def test_adaptation_retains_mem_initialization_and_learned_mem_values():
    policy = _policy(adapt=True, memory=True)
    fresh = policy.model.proprio_history_proj.weight.detach().clone()
    policy.load_state_dict(policy._prepare_pretrained_state_dict(_checkpoint()), strict=True)
    torch.testing.assert_close(policy.model.proprio_history_proj.weight, fresh)
    source = policy.state_dict()
    source["model.proprio_history_proj.weight"] = torch.full_like(fresh, 3)
    policy.load_state_dict(policy._prepare_pretrained_state_dict(source), strict=True)
    torch.testing.assert_close(policy.model.proprio_history_proj.weight, torch.full_like(fresh, 3))


def test_saved_78d_checkpoint_reloads_learned_projections(tmp_path):
    from safetensors.torch import save_file

    trained = _policy(adapt=True)
    trained.load_state_dict(trained._prepare_pretrained_state_dict(_checkpoint()), strict=True)
    with torch.no_grad():
        for parameter in trained.parameters():
            parameter.fill_(11)
    save_file(trained.state_dict(), str(tmp_path / "model.safetensors"))
    restored = _TinyPolicy.from_pretrained(tmp_path, config=trained.config, local_files_only=True)
    for value in restored.state_dict().values():
        torch.testing.assert_close(value, torch.full_like(value, 11))


def test_checkpoint_download_failure_propagates(monkeypatch):
    def fail_download(*args, **kwargs):
        raise OSError("checkpoint download failed")

    monkeypatch.setattr("transformers.utils.cached_file", fail_download)
    with pytest.raises(OSError, match="checkpoint download failed"):
        _TinyPolicy.from_pretrained("test/pi05", config=_policy().config)


def test_checkpoint_deserialization_failure_propagates(tmp_path):
    from safetensors import SafetensorError

    (tmp_path / "model.safetensors").write_text("corrupt checkpoint")
    with pytest.raises(SafetensorError):
        _TinyPolicy.from_pretrained(tmp_path, config=_policy().config, local_files_only=True)


@pytest.mark.parametrize("failure", ["missing_backbone", "backbone_shape", "action_shape"])
def test_checkpoint_incompatibility_propagates(tmp_path, failure):
    from safetensors.torch import save_file

    source = _checkpoint(78)
    if failure == "missing_backbone":
        del source["model.backbone.weight"]
    elif failure == "backbone_shape":
        source["model.backbone.weight"] = torch.zeros(9, 8)
    else:
        source = _checkpoint(32)
    save_file(source, str(tmp_path / "model.safetensors"))
    with pytest.raises(RuntimeError):
        _TinyPolicy.from_pretrained(tmp_path, config=_policy().config, local_files_only=True)


def test_checkpoint_resolution_honors_named_hub_arguments(monkeypatch, tmp_path):
    from safetensors.torch import save_file

    path = tmp_path / "model.safetensors"
    save_file(_checkpoint(78), str(path))
    received = {}

    def resolve(repo_id, filename, **kwargs):
        received.update(kwargs)
        return str(path)

    monkeypatch.setattr("transformers.utils.cached_file", resolve)
    options = {
        "revision": "training-revision",
        "cache_dir": tmp_path / "cache",
        "token": "test-token",
        "force_download": True,
        "resume_download": True,
        "proxies": {"https": "test-proxy"},
        "local_files_only": True,
    }
    restored = _TinyPolicy.from_pretrained("test/pi05", config=_policy().config, **options)
    assert received == options
    torch.testing.assert_close(restored.model.backbone.weight, torch.full((8, 8), 7.0))
