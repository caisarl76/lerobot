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
"""Xiaomi-Robotics-1 port: checkpoint conversion, losses, normalization and a tiny end-to-end policy."""

import json
import random

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.xiaomi_robotics.configuration_xiaomi_robotics import XiaomiRoboticsConfig  # noqa: E402
from lerobot.policies.xiaomi_robotics.modeling_xiaomi_robotics import (  # noqa: E402
    XiaomiRoboticsPolicy,
    XR1Model,
    adapt_action_width,
    convert_xr1_state_dict,
)
from tests.policies.tiny_qwen3vl import build_tiny_qwen3vl  # noqa: E402


def test_convert_state_dict_drops_lm_head_and_flattens_vlm():
    converted = convert_xr1_state_dict(
        {
            "model.vlm.model.language_model.norm.weight": torch.ones(2),
            "model.vlm.model.action_embed.weight": torch.ones(60, 2),
            "model.vlm.lm_head.weight": torch.ones(3, 2),
            "model.dit.layers.0.adaln_table": torch.ones(6, 4),
        }
    )
    assert set(converted) == {
        "vlm.language_model.norm.weight",
        "vlm.action_embed.weight",
        "dit.layers.0.adaln_table",
    }


def test_adapt_action_width_keeps_pretrained_dims_per_choice_block():
    old = {
        "action_projector.layers.0.weight": torch.arange(4 * 60.0).view(4, 60),
        "action_output_layer.layers.2.weight": torch.arange(60 * 4.0).view(60, 4),
        "action_projector_choice.1.layers.0.weight": torch.arange(5 * 60 * 2.0).view(300, 2),
    }
    own = {
        "action_projector.layers.0.weight": torch.zeros(4, 78),
        "action_output_layer.layers.2.weight": torch.zeros(78, 4),
        "action_projector_choice.1.layers.0.weight": torch.full((390, 2), -1.0),
    }
    notes = adapt_action_width(old, own, n_choices=5)
    assert len(notes) == 3
    torch.testing.assert_close(
        old["action_projector.layers.0.weight"][:, :60], torch.arange(240.0).view(4, 60)
    )
    assert (old["action_projector.layers.0.weight"][:, 60:] == 0).all()
    choice = old["action_projector_choice.1.layers.0.weight"].view(5, 78, 2)
    source = torch.arange(600.0).view(5, 60, 2)
    torch.testing.assert_close(choice[:, :60], source)
    assert (choice[:, 60:] == -1).all()


def _loss_host():
    host = XR1Model.__new__(XR1Model)
    torch.nn.Module.__init__(host)
    host.config = XiaomiRoboticsConfig(freq_excluded_dims=[17, 18, 19], device="cpu")
    return host


def test_flow_loss_with_empty_mask_is_finite():
    pred = torch.randn(4, 24, 60, requires_grad=True)
    mse, freq = _loss_host().flow_loss(
        pred, torch.randn_like(pred), torch.zeros_like(pred).bool(), torch.ones_like(pred)
    )
    assert mse.item() == 0.0 and freq.item() == 0.0
    (mse + freq).backward()
    assert pred.grad is not None


def test_flow_loss_without_full_horizon_has_zero_frequency_loss():
    pred = torch.randn(4, 30, 60, requires_grad=True)
    mask = torch.zeros_like(pred).bool()
    mask[:, :5] = True
    mse, freq = _loss_host().flow_loss(pred, torch.randn_like(pred), mask, torch.ones_like(pred))
    assert torch.isfinite(mse) and freq.item() == 0.0


def test_choice_loss_picks_best_hypothesis():
    host = _loss_host()
    target = torch.zeros(3, 4)
    pred = torch.ones(3, 5, 4)
    pred[:, 2] = 0.0  # hypothesis 2 is exact
    action, score = host.choice_loss(pred.view(3, 20), torch.zeros(1, 5), target, torch.ones(3, 4), [3])
    assert action.item() == 0.0 and score.item() == pytest.approx(0.8)


@pytest.fixture(scope="module")
def tiny_vlm(tmp_path_factory):
    return build_tiny_qwen3vl(tmp_path_factory.mktemp("xr1") / "vlm")


def _policy(vlm, tmp_path, action_dim, n_cams, weights=None):
    rel = list(range(28)) if action_dim == 28 else [-1] * 64 + list(range(14, 28))
    stats = tmp_path / f"stats_{action_dim}.json"
    stats.write_text(
        json.dumps(
            {
                "relative_action_state_indices": rel,
                "action_mean": [[0.01 * k] * action_dim for k in range(30)],
                "action_std": [[0.5] * action_dim] * 30,
                "state_q01": [-1.0] * 28,
                "state_q99": [1.0] * 28,
                "state_min": [-2.0] * 28,
                "state_max": [2.0] * 28,
            }
        )
    )
    inputs = {
        f"observation.images.cam{i}": PolicyFeature(FeatureType.VISUAL, (3, 96, 128)) for i in range(n_cams)
    }
    inputs["observation.state"] = PolicyFeature(FeatureType.STATE, (28,))
    config = XiaomiRoboticsConfig(
        input_features=inputs,
        output_features={"action": PolicyFeature(FeatureType.ACTION, (action_dim,))},
        vlm_name=str(vlm),
        dit_hidden_size=64,
        device="cpu",
        model_action_dim=60 if action_dim == 28 else 78,
        state_slots=list(range(0, 7)) + list(range(8, 15)) + list(range(16, 30)),
        relative_action_state_indices=rel,
        stats_path=str(stats),
        pretrained_weights_path=weights,
        view_names=[f"View {i}" for i in range(n_cams)],
    )
    return XiaomiRoboticsPolicy(config)


def _batch(n_cams, action_dim, bsz=3):
    batch = {f"observation.images.cam{i}": torch.rand(bsz, 3, 96, 128) for i in range(n_cams)}
    batch["observation.state"] = torch.rand(bsz, 28) * 2 - 1
    batch["action"] = torch.rand(bsz, 30, action_dim) * 2 - 1
    pad = torch.zeros(bsz, 30, dtype=torch.bool)
    pad[1, 20:] = True  # an episode end inside the chunk: fewer <a_i> tokens for that sample
    batch["action_is_pad"] = pad
    batch["task"] = ["Pick the Apple", "push the duck", "close a laptop"][:bsz]
    return batch


def test_relative_normalization_round_trip(tiny_vlm, tmp_path):
    model = _policy(tiny_vlm, tmp_path, 78, 1).model
    state = torch.rand(2, 28) * 2 - 1
    actions = torch.rand(2, 30, 78)
    normalized = model.normalize_actions(actions, state)
    assert normalized.shape == (2, 30, 78)
    torch.testing.assert_close(model.unnormalize_actions(normalized, state), actions, atol=1e-5, rtol=0)
    # Hands are relative to state 14:28, tokens absolute.
    shifted = state.clone()
    shifted[:, 14:] += 0.1
    delta = model.normalize_actions(actions, shifted) - normalized
    assert torch.allclose(delta[..., :64], torch.zeros(()))
    assert torch.allclose(delta[..., 64:], torch.full((), -0.1 / (0.5 + 1e-6)), atol=1e-5)


@pytest.mark.parametrize("prefix", [0, 4])
def test_xr1_policy_trains_with_and_without_action_prefix(tiny_vlm, tmp_path, monkeypatch, prefix):
    policy = _policy(tiny_vlm, tmp_path, 28, 1).train()
    monkeypatch.setattr(random, "random", lambda: 0.0 if prefix else 0.9)
    monkeypatch.setattr(random, "randint", lambda a, b: prefix if prefix else a)
    loss, metrics = policy.forward(_batch(1, 28))
    assert torch.isfinite(loss) and metrics["prefix_length"] == prefix
    loss.backward()
    grads = [p.grad for group in policy.get_optim_params() for p in group["params"]]
    assert sum(g is not None for g in grads) > 0.9 * len(grads)


def test_xr1_width_78_loads_60_wide_checkpoint_and_reloads(tiny_vlm, tmp_path):
    source = _policy(tiny_vlm, tmp_path, 28, 2).model
    buffers = dict(source.named_buffers())
    module = {
        "model." + ("vlm.model." + k[4:] if k.startswith("vlm.") else k): v
        for k, v in source.state_dict().items()
        if k not in buffers
    }
    weights = tmp_path / "model_states.pt"
    torch.save({"module": module}, weights)
    policy = _policy(tiny_vlm, tmp_path, 78, 2, weights=str(weights))
    torch.testing.assert_close(
        policy.model.action_projector.layers[0].weight[:, :60], source.action_projector.layers[0].weight
    )
    torch.testing.assert_close(
        policy.model.dit.layers[0].attn.qkv_proj.weight, source.dit.layers[0].attn.qkv_proj.weight
    )

    optimizer = policy.config.get_optimizer_preset().build(policy.get_optim_params())
    policy.train()
    batch = _batch(2, 78, bsz=2)
    loss, _ = policy.forward(batch)
    loss.backward()
    optimizer.step()

    torch.manual_seed(0)
    chunk = policy.predict_action_chunk(batch)
    assert chunk.shape == (2, 30, 78) and torch.isfinite(chunk).all()
    policy.save_pretrained(tmp_path / "ckpt")
    reloaded = XiaomiRoboticsPolicy.from_pretrained(tmp_path / "ckpt", strict=True)
    torch.manual_seed(0)
    torch.testing.assert_close(reloaded.predict_action_chunk(batch), chunk)
