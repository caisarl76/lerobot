#!/usr/bin/env python
"""`adapt_action_projections`: load a checkpoint into a wider action space."""

import copy

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.dm05.modeling_dm05 import _ACTION_PROJECTION_KEYS, DM05Policy
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.policies.dm05.test_dm05_integration import _dm05_config, _tiny_core_config


def _config(action_dim: int, **kwargs):
    config = _dm05_config(
        core_config=_tiny_core_config().to_dict(),
        pretrained_name_or_path=".",
        dtype=torch.float32,
        vlm_gradient_checkpointing=False,
        ae_gradient_checkpointing=False,
        freeze_vlm_embedding=False,
        **kwargs,
    )
    config.max_action_dim = action_dim
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    return config


def test_wider_action_dim_reinitialises_only_the_projections(tmp_path):
    source = DM05Policy(_config(4))
    with torch.no_grad():
        source.model.model.action_in_proj.bias.fill_(1.0)  # must not survive the width change
    source.save_pretrained(tmp_path)
    source_state = source.state_dict()

    with pytest.raises(ValueError, match="must match DM05 core action_dim"):
        DM05Policy.from_pretrained(tmp_path, config=_config(6))

    config = _config(6, adapt_action_projections=True)
    target = DM05Policy.from_pretrained(tmp_path, config=copy.deepcopy(config))
    target_state = target.state_dict()
    assert target.model.config.action_dim == 6
    assert target_state["model.model.action_in_proj.weight"].shape[1] == 6
    assert target_state["model.model.action_out_proj.weight"].shape[0] == 6
    for key, value in source_state.items():
        if key not in _ACTION_PROJECTION_KEYS:
            torch.testing.assert_close(target_state[key], value, msg=key)
    assert not torch.equal(target_state["model.model.action_in_proj.bias"], torch.ones(32))

    # The adapted checkpoint reloads unchanged (resume / inference path).
    target.save_pretrained(tmp_path / "adapted")
    reloaded = DM05Policy.from_pretrained(tmp_path / "adapted", config=copy.deepcopy(config)).state_dict()
    for key, value in target_state.items():
        torch.testing.assert_close(reloaded[key], value, msg=key)
