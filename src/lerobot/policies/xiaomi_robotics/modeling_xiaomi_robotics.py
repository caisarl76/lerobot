# Copyright 2026 Xiaomi Corporation and The HuggingFace Inc. team. All rights reserved.
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
"""Native LeRobot port of Xiaomi-Robotics-1 (github.com/XiaomiRobotics/Xiaomi-Robotics-1, `xr1/mibot`).

Mirrors `mibot/models/VLA/XR1.py` (DiT, projectors, flow/frequency/choice losses, asynchronous prefix
training), `mibot/models/VLM/qwen3vl.py` (action/score/state token embeddings) and the JSON dataset/collate
(prompt, image resize, colour augmentation, relative actions with per-step normalization). Differences, all
needed to run on the LeRobot stack (transformers 5.x, no flash-attn): batches are right-padded instead of
flash-attn "packed", the three special embeddings are injected with a hook on the token embedding, and the
key/value cache comes from a transformers `DynamicCache`.
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.distributions import Beta

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import checkpoint_subdir, hub_load_kwargs
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor, Qwen3VLConfig
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModel,
        Qwen3VLTextRMSNorm,
        Qwen3VLTextRotaryEmbedding,
        rotate_half,
    )
else:
    AutoProcessor = None
    Qwen3VLConfig = None
    DynamicCache = None
    Qwen3VLModel = None
    Qwen3VLTextRMSNorm = None
    Qwen3VLTextRotaryEmbedding = None
    rotate_half = None

from .configuration_xiaomi_robotics import XiaomiRoboticsConfig

logger = logging.getLogger(__name__)

SCORE_TOKEN = "<score>"  # nosec B105 (a prompt token, not a password)
STATE_TOKEN = "<state>"  # nosec B105
NUM_ACTION_TOKENS = 60
ACTION_TOKENS = [f"<a_{i}>" for i in range(NUM_ACTION_TOKENS)]
IM_START_TOKEN = "<|im_start|>"  # nosec B105
ACTION_EPS = 1e-6
VLM_PROCESSOR_DIRNAME = "vlm_processor"
# Released XR-1 layout: a 60-wide action/state vector.
XR1_ACTION_DIM = 60


class Projector(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, inter_dim: int | None = None, num_layers: int = 1, bias=False
    ):
        super().__init__()
        inter_dim = output_dim if inter_dim is None else inter_dim
        if num_layers == 1:
            layers: list[nn.Module] = [nn.Linear(input_dim, output_dim, bias=bias)]
        else:
            layers = [nn.Linear(input_dim, inter_dim, bias=bias)]
            for _ in range(1, num_layers - 1):
                layers.extend([nn.GELU(approximate="tanh"), nn.Linear(inter_dim, inter_dim, bias=bias)])
            layers.extend([nn.GELU(approximate="tanh"), nn.Linear(inter_dim, output_dim, bias=bias)])
        self.layers = nn.Sequential(*layers)
        self.apply(_init_linear)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


def _init_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

    def forward(self, timestep: Tensor) -> Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timestep.device) / half
        )
        args = timestep[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(self.mlp[0].weight.dtype)
        return self.mlp(embedding)[:, None]


def _apply_rope(query: Tensor, key: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (query * cos) + (rotate_half(query) * sin), (key * cos) + (rotate_half(key) * sin)


def _repeat_batch(x: Tensor, batch_size: int) -> Tensor:
    if x.shape[0] == batch_size:
        return x
    if batch_size % x.shape[0]:
        raise ValueError(f"Cannot repeat batch of size {x.shape[0]} to {batch_size}")
    return x.repeat_interleave(batch_size // x.shape[0], dim=0)


class DiTAttention(nn.Module):
    """DiT self-attention whose keys/values are prefixed by the VLM layer's cached keys/values."""

    def __init__(self, hidden_size: int, head_dim: int, kv_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = hidden_size // head_dim
        self.kv_group = self.num_heads // kv_heads
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = Qwen3VLTextRMSNorm(head_dim)
        self.k_norm = Qwen3VLTextRMSNorm(head_dim)

    def forward(self, hidden_states, past_key_value, position_embeds, attn_mask):
        batch, seq, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states).view(batch, seq, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(2)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        cos, sin = position_embeds
        query, key = _apply_rope(query, key, cos, sin)
        cache_key, cache_value = past_key_value
        cache_key = _repeat_batch(cache_key, batch).repeat_interleave(self.kv_group, dim=1)
        cache_value = _repeat_batch(cache_value, batch).repeat_interleave(self.kv_group, dim=1)
        key = torch.cat([cache_key.to(key.dtype), key], dim=-2)
        value = torch.cat([cache_value.to(value.dtype), value], dim=-2)
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0)
        return self.o_proj(out.transpose(1, 2).reshape(batch, seq, self.hidden_size))


class DiTMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.down_proj = nn.Linear(hidden_size * 4, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int, kv_heads: int):
        super().__init__()
        self.attn = DiTAttention(hidden_size, head_dim, kv_heads)
        self.mlp = DiTMLP(hidden_size)
        self.input_layernorm = Qwen3VLTextRMSNorm(hidden_size, eps=1e-6)
        self.post_layernorm = Qwen3VLTextRMSNorm(hidden_size, eps=1e-6)
        self.adaln_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(self, hidden_states, past_key_value, position_embeds, timestep, attn_mask):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_table[None] + timestep
        ).chunk(6, dim=1)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states) * (1 + scale_attn) + shift_attn
        hidden_states = residual + gate_attn * self.attn(
            hidden_states, past_key_value, position_embeds, attn_mask
        )
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states) * (1 + scale_mlp) + shift_mlp
        return residual + gate_mlp * self.mlp(hidden_states)


class DiT(nn.Module):
    def __init__(self, layer_num: int, hidden_size: int, head_dim: int, kv_heads: int):
        super().__init__()
        self.layer_num = layer_num
        self.layers = nn.ModuleList([DecoderLayer(hidden_size, head_dim, kv_heads) for _ in range(layer_num)])
        for module in self.modules():
            _init_linear(module)
            if isinstance(module, Qwen3VLTextRMSNorm):
                module.weight.data.fill_(1.0)

    def forward(self, hidden_states, past_key_values, attn_mask, position_embeds, timestep):
        start = len(past_key_values) - self.layer_num
        if start < 0:
            raise ValueError("VLM cache has fewer layers than DiT")
        for index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states, past_key_values[start + index], position_embeds, timestep, attn_mask
            )
        return hidden_states


def convert_xr1_state_dict(module_state: dict[str, Tensor]) -> dict[str, Tensor]:
    """`model_states.pt["module"]` keys (`model.<xr1 attr>`) -> this port's keys."""
    out = {}
    for key, value in module_state.items():
        key = key.removeprefix("model.")
        if key.startswith("vlm.lm_head."):
            continue  # logits are never used; the head is tied to the (frozen) token embedding
        if key.startswith("vlm.model."):
            key = "vlm." + key.removeprefix("vlm.model.")
        out[key] = value
    return out


def _resize_linear_rows(new: Tensor, old: Tensor, blocks: int) -> Tensor:
    """Copy `old` (blocks * n_old rows) into `new` (blocks * n_new rows) block by block."""
    n_old, n_new = old.shape[0] // blocks, new.shape[0] // blocks
    n = min(n_old, n_new)
    new = new.clone()
    for b in range(blocks):
        new[b * n_new : b * n_new + n] = old[b * n_old : b * n_old + n]
    return new


def adapt_action_width(state_dict: dict[str, Tensor], own: dict[str, Tensor], n_choices: int) -> list[str]:
    """Resize the action-width layers in place (first dims keep their pretrained weights). Returns notes."""
    notes = []
    for key, new in own.items():
        if key not in state_dict or state_dict[key].shape == new.shape:
            continue
        old = state_dict[key]
        if key.startswith("action_projector.layers.0."):  # Linear(A -> 1024): columns are action dims
            n = min(old.shape[1], new.shape[1])
            value = new.clone()
            value[:, :n] = old[:, :n]
        elif key.startswith("action_output_layer.layers.2."):  # Linear(1024 -> A)
            value = _resize_linear_rows(new, old, 1)
        elif key.startswith("action_projector_choice.1.layers.0."):  # Linear(H -> A * n_choices)
            value = _resize_linear_rows(new, old, n_choices)
        else:
            raise ValueError(f"Unexpected shape mismatch for {key}: {tuple(old.shape)} vs {tuple(new.shape)}")
        state_dict[key] = value.to(old.dtype)
        notes.append(f"{key}: {tuple(old.shape)} -> {tuple(new.shape)}")
    return notes


class XR1Model(nn.Module):
    def __init__(self, config: XiaomiRoboticsConfig):
        super().__init__()
        require_package("transformers", extra="xiaomi_robotics")
        self.config = config
        if config.vlm_config is None:
            vlm_config = Qwen3VLConfig.from_pretrained(config.vlm_name)
            config.vlm_config = vlm_config.to_dict()
        else:
            vlm_config = Qwen3VLConfig(**config.vlm_config)
        vlm_config._attn_implementation = config.attn_implementation
        self.vlm = Qwen3VLModel._from_config(vlm_config, dtype=torch.bfloat16)
        text = vlm_config.text_config
        hidden = text.hidden_size
        self.vlm.action_embed = nn.Embedding(NUM_ACTION_TOKENS, hidden)
        self.vlm.score_embed = nn.Embedding(1, hidden)

        processor_path = config.processor_path or config.vlm_name
        self.processor = AutoProcessor.from_pretrained(processor_path)
        tokenizer = self.processor.tokenizer
        tokenizer.padding_side = "right"
        missing = [t for t in [SCORE_TOKEN, STATE_TOKEN, *ACTION_TOKENS] if t not in tokenizer.get_vocab()]
        if missing:
            tokenizer.add_tokens(missing, special_tokens=True)
        ids = tokenizer.convert_tokens_to_ids([SCORE_TOKEN, STATE_TOKEN, ACTION_TOKENS[0], ACTION_TOKENS[-1]])
        self.score_id, self.state_id, self.action_start_id = ids[0], ids[1], ids[2]
        if ids[3] != self.action_start_id + NUM_ACTION_TOKENS - 1 or self.state_id != self.score_id + 1:
            raise ValueError(f"XR-1 special tokens are not contiguous: {ids}")
        self.im_start_id = tokenizer.convert_tokens_to_ids(IM_START_TOKEN)
        if self.score_id >= self.vlm.get_input_embeddings().num_embeddings:
            raise ValueError("XR-1 special tokens fall outside the token embedding")

        a_dim, s_dim, dit = config.model_action_dim, config.model_state_dim, config.dit_hidden_size
        self.state_projector_choice = Projector(s_dim, hidden, num_layers=2)
        self.action_projector_choice = nn.Sequential(
            Projector(hidden, hidden, num_layers=4), Projector(hidden, a_dim * config.n_choices)
        )
        self.score_projector_choice = nn.Sequential(
            Projector(hidden, hidden, num_layers=4), Projector(hidden, config.n_choices)
        )
        self.dit = DiT(text.num_hidden_layers, dit, text.head_dim, text.num_key_value_heads)
        self.state_projector = Projector(s_dim, dit, num_layers=2)
        self.action_projector = Projector(a_dim, dit, num_layers=2)
        self.action_output_layer = Projector(dit, a_dim, inter_dim=dit, num_layers=2)
        self.t_embedder = TimestepEmbedder(dit)
        self.t_projector = Projector(dit, 6 * dit, bias=True)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=text)
        self.sink = nn.Embedding(1, dit)
        self.to(torch.bfloat16)
        self.beta = Beta(torch.tensor(1.5), torch.tensor(1.0))

        # Layout and normalization buffers (saved with the checkpoint).
        action_dim = config.action_feature.shape[0]
        state_dim = config.robot_state_feature.shape[0]
        self.register_buffer(
            "action_slots", torch.tensor(config.action_slots or list(range(action_dim))), False
        )
        self.register_buffer("state_slots", torch.tensor(config.state_slots or list(range(state_dim))), False)
        rel = config.relative_action_state_indices or [-1] * action_dim
        self.register_buffer("relative_index", torch.tensor(rel, dtype=torch.long), False)
        self.register_buffer("action_mean", torch.zeros(config.chunk_size, action_dim))
        self.register_buffer("action_std", torch.ones(config.chunk_size, action_dim))
        self.register_buffer("state_q01", torch.zeros(state_dim))
        self.register_buffer("state_q99", torch.zeros(state_dim))
        # Valid state range (the stats exclude corrupt frames); the state is clamped to it before it anchors
        # relative actions, so a corrupt reading cannot produce an unbounded target or command.
        self.register_buffer("state_min", torch.full((state_dim,), -1e4))
        self.register_buffer("state_max", torch.full((state_dim,), 1e4))

        self._state_embeds: Tensor | None = None
        self.vlm.language_model.embed_tokens.register_forward_hook(self._embedding_hook)

        if config.load_base_weights:
            if config.stats_path is None:
                raise ValueError(
                    "Training XR-1 needs `stats_path` (examples/g1_dex3_training/xr1_action_stats.py)"
                )
            self.load_stats(config.stats_path)
            if config.pretrained_weights_path:
                self.load_xr1_weights(config.pretrained_weights_path)

        # Trainability and memory settings of the released recipe.
        self.vlm.get_input_embeddings().requires_grad_(False)
        if config.vision_gradient_checkpointing:
            self.vlm.visual.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if config.ffn_gradient_checkpointing:
            for layer in self.vlm.language_model.layers:
                layer.mlp.forward = _checkpointed(layer.mlp, layer.mlp.forward)

    # ---------------------------------------------------------------- loading
    def load_stats(self, path: str) -> None:
        stats = json.loads(Path(path).read_text())
        rel = stats.get("relative_action_state_indices")
        if rel is not None and rel != self.relative_index.tolist():
            raise ValueError(
                f"{path} was computed with relative dims {rel}, config has {self.relative_index.tolist()}"
            )
        mean = torch.tensor(stats["action_mean"], dtype=torch.float32)
        std = torch.tensor(stats["action_std"], dtype=torch.float32)
        if tuple(mean.shape) != tuple(self.action_mean.shape):
            raise ValueError(f"{path}: action_mean {tuple(mean.shape)} != {tuple(self.action_mean.shape)}")
        self.action_mean.copy_(mean)
        self.action_std.copy_(std)
        self.state_q01.copy_(torch.tensor(stats["state_q01"], dtype=torch.float32))
        self.state_q99.copy_(torch.tensor(stats["state_q99"], dtype=torch.float32))
        if "state_min" in stats:
            self.state_min.copy_(torch.tensor(stats["state_min"], dtype=torch.float32))
            self.state_max.copy_(torch.tensor(stats["state_max"], dtype=torch.float32))

    def load_xr1_weights(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        state_dict = convert_xr1_state_dict(checkpoint.get("module", checkpoint))
        own = {k: v for k, v in self.state_dict().items() if k not in dict(self.named_buffers())}
        notes = adapt_action_width(state_dict, own, self.config.n_choices)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        missing = [k for k in missing if k not in dict(self.named_buffers())]
        logger.info("XR-1 weights from %s; resized: %s", path, notes)
        if missing or unexpected:
            raise ValueError(
                f"XR-1 checkpoint mismatch: missing {missing[:10]}, unexpected {unexpected[:10]}"
            )

    # ---------------------------------------------------------------- VLM embedding hook
    def _embedding_hook(self, module: nn.Module, args: tuple, output: Tensor) -> Tensor:
        ids = args[0]
        state_mask = ids == self.state_id
        if state_mask.any():
            if self._state_embeds is None:
                raise ValueError("State tokens require state embeddings")
            output = output.masked_scatter(
                state_mask[..., None].expand_as(output), self._state_embeds.to(output.dtype)
            )
        action_mask = (ids >= self.action_start_id) & (ids < self.action_start_id + NUM_ACTION_TOKENS)
        if action_mask.any():
            embeds = self.vlm.action_embed(ids[action_mask] - self.action_start_id)
            output = output.masked_scatter(action_mask[..., None].expand_as(output), embeds.to(output.dtype))
        score_mask = ids == self.score_id
        if score_mask.any():
            embeds = self.vlm.score_embed(torch.zeros_like(ids[score_mask]))
            output = output.masked_scatter(score_mask[..., None].expand_as(output), embeds.to(output.dtype))
        return output

    # ---------------------------------------------------------------- inputs
    def _resize(self, image: Tensor) -> Tensor:
        """`mibot.utils.io.resize_image`: multiples of 32, at most `image_max_pixels` pixels."""
        height, width = image.shape[-2:]
        factor, max_pixels = self.config.image_factor, self.config.image_max_pixels
        new_h = max(factor, round(height / factor) * factor)
        new_w = max(factor, round(width / factor) * factor)
        if new_h * new_w > max_pixels:
            scale = math.sqrt(height * width / max_pixels)
            new_h = math.floor(height / scale / factor) * factor
            new_w = math.floor(width / scale / factor) * factor
        if (new_h, new_w) == (height, width):
            return image
        return F.interpolate(image, size=(new_h, new_w), mode="bicubic", antialias=True, align_corners=False)

    def _augment(self, views: list[Tensor]) -> list[Tensor]:
        """`JsonDataset._augment`: brightness/contrast/saturation, each with probability 0.5, shared by the views."""
        from torchvision.transforms import functional as TF  # noqa: N812, PLC0415

        ops = [
            (TF.adjust_brightness, 1.0 + random.uniform(-32.0 / 255.0, 32.0 / 255.0)),
            (TF.adjust_contrast, random.uniform(0.5, 1.5)),
            (TF.adjust_saturation, random.uniform(0.5, 1.5)),
        ]
        flags = [random.randint(0, 1) == 0 for _ in ops]
        out = []
        for image in views:
            for use, (op, value) in zip(flags, ops, strict=True):
                if use:
                    image = op(image, value)
            out.append(image.clamp(0.0, 1.0))
        return out

    def _messages(self, views: list[Tensor], instruction: str, steps: int | None) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        prefix = "The following observations are captured from multiple views.\n"
        for name, image in zip(self.config.view_names, views, strict=True):
            content.append({"type": "text", "text": f"{prefix}# {name}\n"})
            content.append({"type": "image", "image": image})
            prefix = "\n"
        content.append(
            {"type": "text", "text": f"\nGenerate robot actions for the task:\n{instruction} /no_cot"}
        )
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]
        if steps is not None:
            actions = "".join(ACTION_TOKENS[:steps]) + SCORE_TOKEN
            messages += [
                {"role": "user", "content": [{"type": "text", "text": f"Robot state: {STATE_TOKEN}"}]},
                {"role": "assistant", "content": [{"type": "text", "text": actions}]},
            ]
        return messages

    def _vlm_inputs(self, batch: dict[str, Any], steps: list[int] | None, train: bool) -> dict[str, Tensor]:
        cams = [self._resize(batch[k].float()) for k in self.config.image_features]
        bsz = cams[0].shape[0]
        tasks = batch.get("task") or [""] * bsz
        if isinstance(tasks, str):
            tasks = [tasks] * bsz
        messages = []
        for b in range(bsz):
            views = [cam[b].clamp(0.0, 1.0) for cam in cams]
            if train and self.config.color_aug:
                views = self._augment(views)
            messages.append(self._messages(views, str(tasks[b]), None if steps is None else steps[b]))
        device = cams[0].device
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            processor_kwargs={
                "padding": True,
                "return_tensors": "pt",
                "device": device,
                "do_rescale": False,
                "do_resize": False,
            },
        )
        return {k: v.to(device) if isinstance(v, Tensor) else v for k, v in inputs.items()}

    def _normalize_state(self, state: Tensor) -> Tensor:
        """q01/q99 -> [-1, 1], clipped; dims with q99 <= q01 stay 0 (XR-1 `normalize_quantile`)."""
        valid = self.state_q99 > self.state_q01
        norm = 2.0 * (state - self.state_q01) / (self.state_q99 + ACTION_EPS - self.state_q01) - 1.0
        norm = torch.where(valid, norm, torch.zeros_like(norm)).clamp(-1.0, 1.0)
        out = torch.zeros(
            *state.shape[:-1], self.config.model_state_dim, device=state.device, dtype=state.dtype
        )
        out[..., self.state_slots] = norm
        return out

    def _relative(self, actions: Tensor, state: Tensor, sign: float) -> Tensor:
        rel = self.relative_index >= 0
        if not rel.any():
            return actions
        state = torch.maximum(torch.minimum(state, self.state_max), self.state_min)
        offset = torch.zeros_like(actions[:, :1])
        offset[..., rel] = state[:, None, self.relative_index[rel]]
        return actions + sign * offset

    def normalize_actions(self, actions: Tensor, state: Tensor) -> Tensor:
        """(B, T, D) absolute -> model layout (B, T, A): relative where configured, per-step mean/std."""
        actions = self._relative(actions, state, -1.0)
        norm = (actions - self.action_mean) / (self.action_std + ACTION_EPS)
        out = torch.zeros(
            *actions.shape[:2], self.config.model_action_dim, device=actions.device, dtype=actions.dtype
        )
        out[..., self.action_slots] = norm
        return out

    def unnormalize_actions(self, actions: Tensor, state: Tensor) -> Tensor:
        actions = actions[..., self.action_slots].float()
        actions = actions * (self.action_std + ACTION_EPS) + self.action_mean
        return self._relative(actions, state, 1.0)

    # ---------------------------------------------------------------- core
    def _vlm_forward(self, inputs: dict[str, Tensor], state_embeds: Tensor | None):
        position_ids, _ = self.vlm.get_rope_index(
            inputs["input_ids"],
            inputs["mm_token_type_ids"],
            image_grid_thw=inputs.get("image_grid_thw"),
            attention_mask=inputs["attention_mask"],
        )
        cache = DynamicCache(config=self.vlm.config.text_config)
        self._state_embeds = state_embeds
        try:
            out = self.vlm(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                position_ids=position_ids,
                pixel_values=inputs["pixel_values"].to(torch.bfloat16),
                image_grid_thw=inputs["image_grid_thw"],
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        finally:
            self._state_embeds = None
        return out.last_hidden_state, out.past_key_values, position_ids

    def _condition(self, input_ids: Tensor, attention_mask: Tensor, cache, position_ids: Tensor, train: bool):
        """Per-layer (k, v) truncated to the conditioning prefix, its mask, and the DiT position base (3, B)."""
        bsz, length = input_ids.shape
        if train:
            # End of the conditioning segment: the <|im_start|> opening the "Robot state" turn.
            state_pos = (input_ids == self.state_id).float().argmax(dim=1)
            positions = torch.arange(length, device=input_ids.device)[None]
            starts = (input_ids == self.im_start_id) & (positions < state_pos[:, None])
            cond_len = torch.where(starts, positions, torch.zeros_like(positions)).max(dim=1).values
        else:
            cond_len = attention_mask.sum(dim=1)
        max_len = int(cond_len.max())
        cache_mask = torch.arange(max_len, device=input_ids.device)[None] < cond_len[:, None]
        layers = [(layer.keys[:, :, :max_len], layer.values[:, :, :max_len]) for layer in cache.layers]
        masked_pos = torch.where(
            cache_mask[None], position_ids[:, :, :max_len], torch.zeros_like(position_ids[:, :, :max_len])
        )
        return layers, cache_mask, masked_pos.max(dim=-1).values

    def _dit_forward(
        self, noisy, timestep, action_mask, state_embed, position_embeds, cache, attn_mask, prefix_length
    ):
        temb = self.t_projector(self.t_embedder(timestep[:, 0, 0] * 1000)).view(
            -1, 6, self.config.dit_hidden_size
        )
        noisy = self.action_projector(noisy * action_mask)
        sink = self.sink.weight[None].repeat(state_embed.shape[0], 1, 1)
        hidden = torch.cat([sink, state_embed, noisy], dim=1).contiguous()
        hidden = self.dit(hidden, cache, attn_mask, position_embeds, temb)
        output = self.action_output_layer(hidden[:, -noisy.shape[1] :])
        if prefix_length:
            output = torch.cat(
                [torch.zeros_like(output[:, :prefix_length]), output[:, prefix_length:]], dim=1
            )
        return output

    @torch.no_grad()
    def _generate(self, noise: Tensor, kwargs: dict[str, Any]) -> Tensor:
        sample = noise.clone()
        dt = 1.0 / self.config.num_inference_steps
        for step in range(self.config.num_inference_steps):
            t = torch.full(
                (sample.shape[0], 1, 1), step / self.config.num_inference_steps, device=sample.device
            )
            sample = sample + self._dit_forward(sample, t.to(sample.dtype), **kwargs) * dt
        return sample

    def _dit_inputs(self, cache_mask, pos_base, state, action_length, prefix_length, train):
        bsz = state.shape[0]
        device = state.device
        state_length = state.shape[1]
        query_length = 1 + state_length + action_length
        dit_pos = (
            torch.arange(query_length, device=device).view(1, 1, -1).repeat(3, bsz, 1)
            + pos_base[..., None]
            + 1
        )
        if action_length > prefix_length:
            dit_pos[:, :, -(action_length - prefix_length) :] += 10
        cache_part = cache_mask[:, None, :].expand(-1, query_length, -1)
        causal = torch.tril(torch.ones(bsz, query_length, query_length, device=device))
        if train and prefix_length > 2:
            keep_last = 2
            action_start = 1 + state_length
            prefix_end = action_start + prefix_length - keep_last
            suffix_start = action_start + prefix_length
            drop = torch.rand(prefix_length - keep_last, device=device) < self.config.prefix_mask_prob
            causal = causal.clone()
            causal[:, suffix_start:, action_start:prefix_end] *= (~drop).float()
        attn_mask = torch.cat([cache_part.float(), causal], dim=-1)[:, None].bool()
        return dit_pos, attn_mask

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        cfg = self.config
        raw_state = batch[OBS_STATE].float()
        if raw_state.dim() == 3:
            raw_state = raw_state[:, -1]
        actions = self.normalize_actions(batch[ACTION].float(), raw_state)
        bsz, action_length, _ = actions.shape
        valid_steps = (
            ~batch["action_is_pad"] if "action_is_pad" in batch else torch.ones_like(actions[..., 0]).bool()
        )
        valid_steps[:, 0] = True
        dim_mask = torch.zeros(cfg.model_action_dim, device=actions.device, dtype=torch.bool)
        dim_mask[self.action_slots] = True
        action_mask = valid_steps[..., None] & dim_mask
        steps = valid_steps.sum(dim=1).tolist()
        state = self._normalize_state(raw_state)[:, None].to(torch.bfloat16)  # (B, 1, S)
        actions = actions.to(torch.bfloat16)

        inputs = self._vlm_inputs(batch, steps, train=self.training)
        state_embeds = self.state_projector_choice(state.flatten(0, 1))
        hidden, cache, position_ids = self._vlm_forward(inputs, state_embeds)
        input_ids = inputs["input_ids"]
        is_action_token = (input_ids >= self.action_start_id) & (
            input_ids < self.action_start_id + NUM_ACTION_TOKENS
        )
        action_choice = self.action_projector_choice(hidden[is_action_token])
        score_choice = self.score_projector_choice(hidden[input_ids == self.score_id])

        cache_layers, cache_mask, pos_base = self._condition(
            input_ids, inputs["attention_mask"], cache, position_ids, True
        )
        # Asynchronous training and the 4 noise draws per sample are training-only (XR-1 `forward`/`_repeat`).
        train = self.training
        prefix_length = random.randint(1, 6) if train and cfg.async_train and random.random() < 0.5 else 0
        dit_pos, attn_mask = self._dit_inputs(
            cache_mask, pos_base, state, action_length, prefix_length, train
        )
        repeat = cfg.training_repeat if train else 1
        dit_pos = dit_pos.repeat_interleave(repeat, dim=1)
        act = actions.repeat_interleave(repeat, dim=0)
        act_mask = action_mask.repeat_interleave(repeat, dim=0).to(act.dtype)
        state_embed = self.state_projector(state).repeat_interleave(repeat, dim=0)
        attn_mask = attn_mask.repeat_interleave(repeat, dim=0)
        position_embeds = self.rotary_emb(act, dit_pos)
        prefix = act[:, :prefix_length]
        noise = torch.randn_like(act)
        kwargs = {
            "action_mask": act_mask,
            "state_embed": state_embed,
            "position_embeds": position_embeds,
            "cache": cache_layers,
            "attn_mask": attn_mask,
            "prefix_length": prefix_length,
        }
        t = ((1 - self.beta.sample((act.shape[0],)).to(act.device)) * 0.999).to(act.dtype)[:, None, None]
        noisy = (1 - t) * noise + t * act
        target = (act - noise)[:, prefix_length:]
        pred = self._dit_forward(torch.cat([prefix, noisy[:, prefix_length:]], dim=1), t, **kwargs)[
            :, prefix_length:
        ]
        if prefix_length:
            prefix_pred = self._generate(torch.cat([prefix, noise[:, prefix_length:]], dim=1), kwargs)
            weight = (prefix_pred[:, prefix_length:] - act[:, prefix_length:]).abs()
        else:
            weight = torch.ones_like(pred)
        suffix_mask = act_mask[:, prefix_length:].bool()
        loss_mse, loss_freq = self.flow_loss(pred, target, suffix_mask, weight)

        choice_target = actions[valid_steps]  # (sum steps, A), same order as the <a_i> tokens
        choice_mask = action_mask[valid_steps]
        loss_choice, loss_score = self.choice_loss(
            action_choice, score_choice, choice_target, choice_mask, steps
        )
        loss = 0.5 * loss_mse + cfg.freq_coefficient * loss_freq + 0.5 * loss_choice + 0.5 * loss_score
        metrics = {
            "loss": loss.item(),
            "loss_mse": loss_mse.item(),
            "loss_freq": loss_freq.item(),
            "loss_choice": loss_choice.item(),
            "loss_score": loss_score.item(),
            "prefix_length": float(prefix_length),
            "vlm_tokens": float(input_ids.shape[1]),
        }
        return loss, metrics

    def flow_loss(self, pred: Tensor, target: Tensor, mask: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
        pred, target, weight = pred.float(), target.float(), weight.float()
        if not mask.any():
            zero = (pred.sum() + target.sum()) * 0.0
            return zero, zero
        with torch.no_grad():
            weight = weight.clone()
            weight[mask] /= weight[mask].mean()
            weight.clamp_(0.5, 5.0)
        loss_mse = (F.mse_loss(pred, target, reduction="none") * weight)[mask].mean()
        freq = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)).abs()
        valid_batch = mask[:, -1].any(dim=1)
        if not valid_batch.any():
            return loss_mse, freq.sum() * 0.0
        freq_mask = mask[valid_batch, : freq.shape[1]].clone()
        dims = [d for d in self.config.freq_excluded_dims if d < freq_mask.shape[-1]]
        freq_mask[:, :, dims] = False
        freq_weight = weight.mean(dim=(1, 2)).view(-1, 1, 1)
        loss_freq = (freq * freq_weight)[valid_batch][freq_mask].mean()
        return loss_mse, loss_freq

    def choice_loss(self, action_pred, score_pred, target, mask, lengths: list[int]) -> tuple[Tensor, Tensor]:
        n = self.config.n_choices
        if sum(lengths) != action_pred.shape[0] or len(lengths) != score_pred.shape[0]:
            raise ValueError("Choice token counts do not match targets")
        action_losses, score_losses = [], []
        start = 0
        for index, length in enumerate(lengths):
            end = start + length
            predictions = action_pred[start:end].float().reshape(length, n, -1).transpose(0, 1)
            sample_target = target[start:end].float().unsqueeze(0).repeat(n, 1, 1)
            sample_mask = mask[start:end].bool().unsqueeze(0).repeat(n, 1, 1)
            error = (
                F.l1_loss(predictions, sample_target, reduction="none")[sample_mask]
                .reshape(n, -1)
                .mean(dim=-1)
            )
            best = error.argmin()
            action_losses.append(error[best])
            score_losses.append(((score_pred[index].float() - error.detach()) ** 2).mean())
            start = end
        return torch.stack(action_losses).mean(), torch.stack(score_losses).mean()

    @torch.no_grad()
    def sample(self, batch: dict[str, Any], noise: Tensor | None = None) -> Tensor:
        cfg = self.config
        raw_state = batch[OBS_STATE].float()
        if raw_state.dim() == 3:
            raw_state = raw_state[:, -1]
        state = self._normalize_state(raw_state)[:, None].to(torch.bfloat16)
        inputs = self._vlm_inputs(batch, None, train=False)
        _, cache, position_ids = self._vlm_forward(inputs, None)
        cache_layers, cache_mask, pos_base = self._condition(
            inputs["input_ids"], inputs["attention_mask"], cache, position_ids, False
        )
        bsz = state.shape[0]
        dit_pos, attn_mask = self._dit_inputs(cache_mask, pos_base, state, cfg.chunk_size, 0, False)
        dim_mask = torch.zeros(cfg.model_action_dim, device=state.device, dtype=torch.bfloat16)
        dim_mask[self.action_slots] = 1
        if noise is None:
            noise = torch.randn(bsz, cfg.chunk_size, cfg.model_action_dim, device=state.device)
        noise = noise.to(torch.bfloat16)
        kwargs = {
            "action_mask": dim_mask.expand(bsz, cfg.chunk_size, -1),
            "state_embed": self.state_projector(state),
            "position_embeds": self.rotary_emb(noise, dit_pos),
            "cache": cache_layers,
            "attn_mask": attn_mask,
            "prefix_length": 0,
        }
        normalized = self._generate(noise, kwargs)
        return self.unnormalize_actions(normalized, raw_state)


def _checkpointed(module: nn.Module, forward):
    def checkpointed_forward(x):
        if module.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(forward, x, use_reentrant=False)
        return forward(x)

    return checkpointed_forward


class XiaomiRoboticsPolicy(PreTrainedPolicy):
    config_class = XiaomiRoboticsConfig
    name = "xiaomi_robotics"

    def __init__(self, config: XiaomiRoboticsConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.model = XR1Model(config)
        self.reset()

    @classmethod
    def from_pretrained(
        cls, pretrained_name_or_path, *, config: XiaomiRoboticsConfig | None = None, **kwargs
    ):
        hub = hub_load_kwargs(kwargs)
        if config is None:
            config = XiaomiRoboticsConfig.from_pretrained(pretrained_name_or_path, **hub)
        config.load_base_weights = False
        processor_dir = checkpoint_subdir(pretrained_name_or_path, VLM_PROCESSOR_DIRNAME, **hub)
        if processor_dir is not None:
            config.processor_path = str(processor_dir)
        return super().from_pretrained(pretrained_name_or_path, config=config, **kwargs)

    def _save_pretrained(self, save_directory: Path) -> None:
        super()._save_pretrained(save_directory)
        from lerobot.distributed.utils import is_main_process  # noqa: PLC0415

        if is_main_process():
            self.model.processor.save_pretrained(Path(save_directory) / VLM_PROCESSOR_DIRNAME)

    def get_optim_params(self) -> list[dict[str, Any]]:
        """XR-1 `BaseRunner.build_optimizer`: no weight decay for biases, norms, rotary and adaLN tables."""
        no_decay = ("bias", "norm", "ln", "rotary_emb", "adaln")
        decay, other = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (other if any(token in name.lower() for token in no_decay) else decay).append(param)
        return [
            {"params": decay, "weight_decay": self.config.optimizer_weight_decay},
            {"params": other, "weight_decay": 0.0},
        ]

    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque(maxlen=self.config.n_action_steps)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        return self.model(batch)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        return self.model.sample(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
