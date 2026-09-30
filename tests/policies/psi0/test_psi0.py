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
"""Psi0 port: action header, weight loading, flow sampler, and a tiny end-to-end policy."""

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.psi0.action_header import ActionTransformerModel  # noqa: E402
from lerobot.policies.psi0.configuration_psi0 import Psi0Config  # noqa: E402
from lerobot.policies.psi0.modeling_psi0 import (  # noqa: E402
    Psi0Policy,
    flow_sigmas,
    load_action_header_weights,
)
from tests.policies.tiny_qwen3vl import build_tiny_clip, build_tiny_qwen3vl  # noqa: E402

HEADER = {"action_hidden_dim": 64, "num_attention_heads": 4, "attention_head_dim": 16, "view_feature_dim": 32}


def _header(**kwargs):
    return ActionTransformerModel(**{**HEADER, **kwargs})


def test_flow_sigmas_match_diffusers_euler_scheduler():
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(10)
    torch.testing.assert_close(flow_sigmas(10), scheduler.sigmas.float(), atol=1e-6, rtol=0)
    sample, velocity = torch.randn(2, 5, 3), torch.randn(2, 5, 3)
    expected = scheduler.step(velocity, scheduler.timesteps[0], sample).prev_sample
    sigmas = flow_sigmas(10)
    torch.testing.assert_close(sample + (sigmas[1] - sigmas[0]) * velocity, expected)


@pytest.mark.parametrize("layerwise", [False, True])
def test_header_amo_and_sonic_variants_forward_backward(layerwise):
    kwargs = dict(action_pred_horizon=30, action_dim=36 if not layerwise else 80, odim=36 if not layerwise else 45)
    if layerwise:
        kwargs.update(
            action_num_blocks=3,
            layerwise_vlm_fusion=True,
            qk_norm="rms_norm",
            combined_temb=True,
            pooled_projection_dim=8,
            state_as_action_token=True,
            state_null_token=True,
            state_drop_prob=0.5,
            dropout=0.0,
            state_feature_dropout=0.0,
        )
    else:
        kwargs.update(action_num_blocks=2)
    header = _header(**kwargs).train()
    batch, horizon, dim = 3, 30, kwargs["action_dim"]
    views = torch.randn(batch, 3 if layerwise else 1, 7, 32)
    mask = torch.ones(batch, 7)
    mask[0, 5:] = 0  # right padding of the first sample's VLM tokens
    # Per-token timesteps (training-time RTC) and per-sample ones must both work.
    for timestep in (torch.rand(batch) * 1000, torch.rand(batch, horizon) * 1000):
        out = header(
            torch.randn(batch, horizon, dim),
            timestep,
            views,
            torch.randn(batch, 1, kwargs["odim"]),
            vlm_attn_mask=mask,
            pooled_projections=torch.randn(batch, 8) if layerwise else None,
        )
        assert out.shape == (batch, horizon, dim)
        out.pow(2).mean().backward()
    assert header.action_proj_in.dec_pos.grad is not None
    if layerwise:
        assert header.state_null.grad is not None


def test_header_ignores_padded_context_tokens():
    header = _header(action_pred_horizon=4, action_dim=6, odim=5, action_num_blocks=2, dropout=0.0).eval()
    views = torch.randn(1, 1, 5, 32)
    args = (torch.randn(1, 4, 6), torch.tensor([500.0]))
    obs = torch.randn(1, 1, 5)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
    out = header(*args, views, obs, vlm_attn_mask=mask)
    changed = views.clone()
    changed[:, :, 3:] = torch.randn(1, 1, 2, 32) * 10
    torch.testing.assert_close(header(*args, changed, obs, vlm_attn_mask=mask), out)


def test_header_weights_load_fully_or_blocks_only():
    source = _header(action_pred_horizon=16, action_dim=36, odim=36, action_num_blocks=2)
    state = source.state_dict()
    same = _header(action_pred_horizon=16, action_dim=36, odim=36, action_num_blocks=2)
    missing, unexpected, skipped = load_action_header_weights(same, state, chunk_size=16, action_dim=36)
    assert not missing and not unexpected and not skipped
    torch.testing.assert_close(same.action_proj_out.linear.weight, source.action_proj_out.linear.weight)

    longer = _header(action_pred_horizon=30, action_dim=36, odim=36, action_num_blocks=2)
    before = longer.action_proj_in.ac_proj[0].weight.clone()
    missing, unexpected, skipped = load_action_header_weights(longer, state, chunk_size=30, action_dim=36)
    assert not unexpected and all(not k.startswith("transformer_blocks") for k in missing)
    torch.testing.assert_close(longer.transformer_blocks[0].attn.to_q.weight, source.transformer_blocks[0].attn.to_q.weight)
    torch.testing.assert_close(longer.action_proj_in.ac_proj[0].weight, before)  # re-initialised layer kept


def _features(n_cams, action_dim):
    inputs = {f"observation.images.cam{i}": PolicyFeature(FeatureType.VISUAL, (3, 96, 128)) for i in range(n_cams)}
    inputs["observation.state"] = PolicyFeature(FeatureType.STATE, (28,))
    return inputs, {"action": PolicyFeature(FeatureType.ACTION, (action_dim,))}


def _batch(n_cams, action_dim, bsz=2):
    batch = {f"observation.images.cam{i}": torch.rand(bsz, 3, 96, 128) for i in range(n_cams)}
    batch["observation.state"] = torch.rand(bsz, 28) * 2 - 1
    batch["action"] = torch.rand(bsz, 30, action_dim) * 2 - 1
    batch["action_is_pad"] = torch.zeros(bsz, 30, dtype=torch.bool)
    batch["task"] = ["Pick the Apple", "push the duck"][:bsz]
    return batch


@pytest.fixture(scope="module")
def tiny_assets(tmp_path_factory):
    root = tmp_path_factory.mktemp("psi0")
    return build_tiny_qwen3vl(root / "vlm"), build_tiny_clip(root / "clip")


@pytest.mark.parametrize("recipe", ["amo_joint28", "sonic78"])
def test_psi0_policy_trains_samples_and_reloads(recipe, tiny_assets, tmp_path):
    vlm, clip = tiny_assets
    common = dict(vlm_path=str(vlm), hidden_dim=64, num_heads=4, attention_head_dim=16, view_feature_dim=64)
    if recipe == "amo_joint28":
        n_cams, action_dim = 1, 28
        extra = dict(num_blocks=2, model_action_dim=36, model_state_dim=36, rtc=True, image_size=(48, 64))
    else:
        n_cams, action_dim = 2, 78
        extra = dict(
            num_blocks=2,
            model_action_dim=80,
            model_state_dim=45,
            state_slots=list(range(15, 43)),
            vlm_layer_indices=[2, 4],
            qk_norm="rms_norm",
            combined_temb=True,
            pooled_projection_dim=32,
            pooled_text_encoder_path=str(clip),
            state_as_action_token=True,
            state_null_token=True,
            state_drop_prob=0.1,
            state_noise_std=0.05,
            tune_vlm=True,
            gradient_checkpointing=True,
            view_aug=True,
            image_size=(54, 96),
        )
    inputs, outputs = _features(n_cams, action_dim)
    config = Psi0Config(input_features=inputs, output_features=outputs, device="cpu", **common, **extra)
    policy = Psi0Policy(config).train()
    batch = _batch(n_cams, action_dim)
    loss, metrics = policy.forward(batch)
    assert torch.isfinite(loss) and metrics["loss"] > 0
    loss.backward()
    groups = policy.get_optim_params()
    assert all(p.grad is not None for group in groups for p in group["params"])
    lrs = [group.get("lr") for group in groups]
    assert lrs == ([None] if recipe == "amo_joint28" else [None, 1e-6, 1e-5, 1e-4])

    torch.manual_seed(0)
    chunk = policy.predict_action_chunk(batch)
    assert chunk.shape == (2, 30, action_dim) and torch.isfinite(chunk).all()
    policy.save_pretrained(tmp_path)
    assert (tmp_path / "vlm_processor").is_dir()
    reloaded = Psi0Policy.from_pretrained(tmp_path, strict=True)
    assert reloaded.config.load_base_weights is False
    torch.manual_seed(0)
    torch.testing.assert_close(reloaded.predict_action_chunk(batch), chunk)
