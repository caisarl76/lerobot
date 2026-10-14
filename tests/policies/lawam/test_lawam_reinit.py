#!/usr/bin/env python

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

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

pytest.importorskip("transformers", reason="lawam requires the `lawam` extra (transformers)")
pytest.importorskip("diffusers", reason="lawam requires the `lawam` extra (diffusers)")

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.lawam.configuration_lawam import LaWAMConfig
from lerobot.policies.lawam.modeling_lawam import LaWAMPolicy
from lerobot.utils.constants import ACTION


class _TinyNative(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.action_encoder = nn.Linear(action_dim, 4)
        self.trunk = nn.Linear(4, 4)


def _policy(action_dim: int, reinit_modules=None) -> LaWAMPolicy:
    config = LaWAMConfig(
        device="cpu",
        input_features={"observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        base_vlm="dummy-qwen",
        flow_action_dim=action_dim,
        reinit_modules=reinit_modules,
    )
    return LaWAMPolicy(config, native_model=_TinyNative(action_dim))


def test_reinit_modules_skip_only_listed_mismatched_tensors(tmp_path) -> None:
    source = _policy(32)
    ckpt = tmp_path / "model.safetensors"
    save_file({k: v.contiguous() for k, v in source.state_dict().items()}, str(ckpt))

    target = _policy(78, reinit_modules=["model.action_encoder"])
    fresh_bias = target.model.action_encoder.bias.detach().clone()
    LaWAMPolicy._load_as_safetensor(target, str(ckpt), "cpu", strict=True)
    assert torch.equal(target.model.trunk.weight, source.model.trunk.weight)
    assert target.model.action_encoder.weight.shape == (4, 78)
    # Shape-compatible tensors under a reinit prefix still load from the checkpoint.
    assert torch.equal(target.model.action_encoder.bias, source.model.action_encoder.bias)
    assert not torch.equal(fresh_bias, source.model.action_encoder.bias)

    with pytest.raises(RuntimeError):
        LaWAMPolicy._load_as_safetensor(_policy(78), str(ckpt), "cpu", strict=True)
    with pytest.raises(ValueError):
        LaWAMPolicy._load_as_safetensor(_policy(78, ["model.trunk"]), str(ckpt), "cpu", strict=True)
