#!/usr/bin/env python

# ------------------------------------------------------------------------------
# Copyright 2025 The HuggingFace Inc. team and 2toINF (https://github.com/2toINF)
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
# ------------------------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch.optim.lr_scheduler import LambdaLR

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import (
    CosineDecayWithWarmupSchedulerConfig,
    LRSchedulerConfig,
    OptimizerConfig,
    XVLAAdamWConfig,
)
from lerobot.utils.constants import OBS_IMAGES

# Conditional import for type checking and lazy loading
from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers import Florence2Config
else:
    Florence2Config = None


def _translate_vision_config(vision_config: dict[str, Any]) -> dict[str, Any]:
    """Translate a vision config from the original Microsoft remote-code Florence-2 format
    (used by existing XVLA checkpoints) to the native ``transformers`` format.

    Configs already in the native format pass through unchanged.
    """
    vision = dict(vision_config)
    model_type = vision.pop("model_type", None)
    if model_type not in (None, "davit", "florence_vision"):
        raise ValueError(f"Unsupported Florence-2 vision backbone: {model_type!r}")
    vision.pop("enable_checkpoint", None)

    image_pos_embed = vision.pop("image_pos_embed", None)
    if image_pos_embed is not None:
        if image_pos_embed.get("type") != "learned_abs_2d":
            raise ValueError(f"Unsupported image_pos_embed type: {image_pos_embed.get('type')!r}")
        vision["max_position_embeddings"] = image_pos_embed["max_pos_embeddings"]

    visual_temporal_embedding = vision.pop("visual_temporal_embedding", None)
    if visual_temporal_embedding is not None:
        if visual_temporal_embedding.get("type") != "COSINE":
            raise ValueError(
                f"Unsupported visual_temporal_embedding type: {visual_temporal_embedding.get('type')!r}"
            )
        vision["max_temporal_embeddings"] = visual_temporal_embedding["max_temporal_embeddings"]

    image_feature_source = vision.pop("image_feature_source", None)
    if image_feature_source is not None and list(image_feature_source) != [
        "spatial_avg_pool",
        "temporal_avg_pool",
    ]:
        # the native Florence2MultiModalProjector hardcodes this feature combination
        raise ValueError(f"Unsupported image_feature_source: {image_feature_source!r}")

    if "dim_embed" in vision:
        vision["embed_dim"] = vision.pop("dim_embed")
    return vision


@PreTrainedConfig.register_subclass("xvla")
@dataclass
class XVLAConfig(PreTrainedConfig):
    """
    Configuration class for the XVLA (Extended Vision-Language-Action) policy so it can
    plug into the LeRobot training stack.

    The config mirrors the knobs exposed in the original XVLA repository but also
    declares the input/output feature contract required by LeRobot.
    """

    # Input / output structure
    n_obs_steps: int = 1
    chunk_size: int = 32
    n_action_steps: int = 32
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # Florence2 backbone and tokenizer configuration
    florence_config: dict[str, Any] = field(default_factory=dict)
    tokenizer_name: str = "facebook/bart-large"
    tokenizer_max_length: int = 64
    tokenizer_padding_side: str = "right"
    pad_language_to: str = "max_length"

    # Transformer head
    hidden_size: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_domains: int = 30
    len_soft_prompts: int = 32
    dim_time: int = 32
    max_len_seq: int = 512
    use_hetero_proj: bool = False

    # Action & proprioception
    action_mode: str = "ee6d"
    num_denoising_steps: int = 10
    use_proprio: bool = True
    max_state_dim: int = 32
    max_action_dim: int = 20  # Maximum action dimension for padding (used by "auto" action mode)
    domain_feature_key: str | None = None
    # When set, every sample uses this domain id (soft prompt + domain-aware heads), whatever the
    # preprocessor added. Needed when the processors are loaded from a checkpoint of another domain.
    domain_id: int | None = None

    # Vision preprocessing
    resize_imgs_with_padding: tuple[int, int] | None = None
    # "pad": keep the aspect ratio and pad (LeRobot default); "stretch": plain bicubic resize to
    # `resize_imgs_with_padding`, like the original X-VLA dataloader (transforms.Resize((224, 224))).
    resize_imgs_mode: str = "pad"
    num_image_views: int | None = None
    empty_cameras: int = 0

    # Freezing options for VLM components
    # By default, VLM encoders are frozen and only policy transformer + soft prompts train
    freeze_vision_encoder: bool = False  # Freeze VLM vision encoder weights
    freeze_language_encoder: bool = False  # Freeze VLM language encoder weights
    train_policy_transformer: bool = True  # Allow policy transformer to train
    train_soft_prompts: bool = True  # Allow soft prompts to train

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.99)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 10.0
    # Soft-prompt LR settings (for optional warm-up)
    optimizer_soft_prompt_lr_scale: float = 1.0  # Scale factor for soft-prompt LR
    optimizer_soft_prompt_warmup_lr_scale: float | None = None  # Start scale for warmup (e.g., 0.01)

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # Original X-VLA fine-tuning schedule (2toINF/X-VLA train.py, `--use_cosine_decay`). 0 keeps the
    # LeRobot presets above. When > 0: four param groups (vlm, transformer_core, soft_prompts,
    # action_heads); for the first `official_freeze_steps` scheduler steps only the soft prompts and
    # action heads train; then every group warms up linearly from 0 over `scheduler_warmup_steps`
    # and decays with a cosine to `scheduler_decay_lr / optimizer_lr` of its peak at the end of
    # training. Scheduler steps are micro-batches in LeRobot: multiply by the gradient accumulation.
    official_freeze_steps: int = 0
    official_lr_coef: float = 0.1  # LR multiplier of the VLM and soft prompts (`--learning_coef`)

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.chunk_size <= 0:
            raise ValueError("`chunk_size` must be strictly positive.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"`n_action_steps` ({self.n_action_steps}) must be <= `chunk_size` ({self.chunk_size})."
            )
        if self.num_image_views is not None and self.num_image_views <= 0:
            raise ValueError("`num_image_views` must be > 0 when specified.")
        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        if self.resize_imgs_mode not in ("pad", "stretch"):
            raise ValueError(f"Invalid resize_imgs_mode: {self.resize_imgs_mode}")
        self._florence_config_obj: Florence2Config | None = None

    def get_florence_config(self) -> Florence2Config:
        """
        Build (and cache) the native ``transformers`` Florence-2 config that backs the VLM.

        ``florence_config`` may be given either in the native ``transformers`` format or in the
        original Microsoft remote-code format stored by existing XVLA checkpoints (e.g. with
        ``dim_embed`` / ``image_pos_embed`` in the vision config); the latter is translated
        field-by-field to the native format.
        """
        if self._florence_config_obj is None:
            config_dict = dict(self.florence_config)
            if config_dict.get("vision_config") is None:
                raise ValueError("vision_config is required")
            if config_dict.get("text_config") is None:
                raise ValueError("text_config is required")

            vision_config = _translate_vision_config(config_dict["vision_config"])
            text_config = dict(config_dict["text_config"])
            if text_config.get("model_type", "florence2_language") == "florence2_language":
                # The MS remote-code language config is BART, field for field.
                text_config["model_type"] = "bart"

            kwargs = {
                key: config_dict[key]
                for key in (
                    "pad_token_id",
                    "bos_token_id",
                    "eos_token_id",
                    "image_token_id",
                    "is_encoder_decoder",
                    "tie_word_embeddings",
                )
                if key in config_dict
            }
            self._florence_config_obj = Florence2Config(
                vision_config=vision_config, text_config=text_config, **kwargs
            )
        return self._florence_config_obj

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("XVLA requires at least one visual feature in the inputs.")
        if self.use_proprio and self.robot_state_feature is None:
            raise ValueError("`use_proprio=True` requires a proprioceptive state feature.")
        if self.num_image_views is None:
            self.num_image_views = len(self.image_features) + self.empty_cameras
        else:
            self.num_image_views = max(self.num_image_views, len(self.image_features) + self.empty_cameras)

        if self.empty_cameras > 0:
            height, width = (480, 640)
            if self.resize_imgs_with_padding is not None:
                height, width = self.resize_imgs_with_padding
            for idx in range(self.empty_cameras):
                key = f"{OBS_IMAGES}.empty_camera_{idx}"
                if key not in self.input_features:
                    self.input_features[key] = PolicyFeature(
                        type=FeatureType.VISUAL,
                        shape=(3, height, width),
                    )

    def get_optimizer_preset(self) -> OptimizerConfig:
        """Return the XVLA-specific optimizer with differential learning rates.

        This optimizer applies:
        - 1/10 LR for VLM parameters (stable optimization)
        - Full LR for transformer/action head
        - Configurable LR for soft-prompts (with optional warm-up)
        """
        if self.official_freeze_steps > 0:
            return XVLAOfficialAdamWConfig(
                lr=self.optimizer_lr,
                betas=self.optimizer_betas,
                eps=self.optimizer_eps,
                weight_decay=self.optimizer_weight_decay,
                grad_clip_norm=self.optimizer_grad_clip_norm,
                lr_coef=self.official_lr_coef,
            )
        return XVLAAdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
            soft_prompt_lr_scale=self.optimizer_soft_prompt_lr_scale,
            soft_prompt_warmup_lr_scale=self.optimizer_soft_prompt_warmup_lr_scale,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig:
        if self.official_freeze_steps > 0:
            return XVLAOfficialSchedulerConfig(
                num_warmup_steps=self.scheduler_warmup_steps,
                freeze_steps=self.official_freeze_steps,
                min_lr_ratio=self.scheduler_decay_lr / self.optimizer_lr,
            )
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> list[int] | None:
        return None


_XVLA_GROUP_NAMES = ("vlm", "transformer_core", "soft_prompts", "action_heads")


def _xvla_group_name(param_name: str) -> str:
    if ".vlm." in f".{param_name}":
        return "vlm"
    if "soft_prompt_hub" in param_name:
        return "soft_prompts"
    if "action_encoder" in param_name or "action_decoder" in param_name:
        return "action_heads"
    return "transformer_core"


@OptimizerConfig.register_subclass("xvla-official-adamw")
@dataclass
class XVLAOfficialAdamWConfig(OptimizerConfig):
    """AdamW with the four named param groups of the original X-VLA `build_optimizer`.

    The VLM and soft prompts peak at `lr * lr_coef`, the transformer core and action heads at `lr`.
    One weight decay for all groups. The per-group schedule is `XVLAOfficialSchedulerConfig`.
    """

    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    lr_coef: float = 0.1

    def build(self, params: Any) -> torch.optim.Optimizer:
        assert isinstance(params, dict), "xvla-official-adamw needs named parameters."
        groups: dict[str, list] = {name: [] for name in _XVLA_GROUP_NAMES}
        for name, p in params.items():
            if p.requires_grad:
                groups[_xvla_group_name(name)].append(p)
        scale = {"vlm": self.lr_coef, "soft_prompts": self.lr_coef}
        param_groups = [
            {"name": name, "params": ps, "lr": self.lr * scale.get(name, 1.0)}
            for name, ps in groups.items()
            if ps
        ]
        return torch.optim.AdamW(
            param_groups, lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay
        )


@LRSchedulerConfig.register_subclass("xvla-official")
@dataclass
class XVLAOfficialSchedulerConfig(LRSchedulerConfig):
    """Per-group LR factors of the original X-VLA `update_group_lrs` with `--use_cosine_decay`.

    Steps < freeze_steps: VLM and transformer core at 0, soft prompts and action heads at peak.
    Afterwards every group: linear warmup from 0 over `num_warmup_steps`, then cosine to
    `min_lr_ratio` at `num_training_steps`.
    """

    num_warmup_steps: int = 1000
    freeze_steps: int = 1000
    min_lr_ratio: float = 0.1

    def build(self, optimizer: torch.optim.Optimizer, num_training_steps: int) -> LambdaLR:
        def after_freeze(step: int) -> float:
            progress = step - self.freeze_steps
            if progress < self.num_warmup_steps:
                return progress / max(1, self.num_warmup_steps)
            remain = max(1, num_training_steps - (self.freeze_steps + self.num_warmup_steps))
            ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - self.num_warmup_steps) / remain)))
            return self.min_lr_ratio + (1 - self.min_lr_ratio) * ratio

        def factor(trains_during_freeze: bool):
            def fn(step: int) -> float:
                if step < self.freeze_steps:
                    return 1.0 if trains_during_freeze else 0.0
                return after_freeze(step)

            return fn

        return LambdaLR(
            optimizer,
            [factor(g.get("name") in ("soft_prompts", "action_heads")) for g in optimizer.param_groups],
        )
