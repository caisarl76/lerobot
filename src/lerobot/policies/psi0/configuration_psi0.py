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
"""Configuration of the native LeRobot port of Psi0 (USC PSI Lab, arXiv 2603.12263).

Psi0 couples a Qwen3-VL-2B backbone (System-2) with a flow-matching multimodal diffusion transformer
("action header", System-1). Two released families are supported through plain config fields:

* the AMO-era real-G1 recipe (``scripts/train/psi0/finetune-real-psi0.sh``): frozen VLM
  ``pre.fast.1by1.2601091803.ckpt.ego200k.he30k``, six-block header ``postpre.1by1.pad36...``,
  36-D padded joints, training-time RTC;
* the SONIC recipe (``finetune-real-sonic-psi0-2.8B-sonic1.1-robust.sh``): VLM and 12-block header
  from ``postpre.sonic1.1.unifolm.2609181726.40k``, 80-D (64 SONIC v1.1 tokens + 14 Dex3 + 2 neck),
  tuned VLM with per-component learning rates, layer-wise VLM fusion, CLIP pooled task embedding,
  state as action token.

``write_psi0_xr1_configs.py`` in ``examples/g1_dex3_training`` fills the fields for each recipe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineAnnealingWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("psi0")
@dataclass
class Psi0Config(PreTrainedConfig):
    n_obs_steps: int = 1
    # Psi0 predicts 30-step chunks and executes all of them (action_exec_horizon = 30).
    chunk_size: int = 30
    n_action_steps: int = 30

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            # Psi0 "bounds": min/max -> [-1, 1]. The model clips the normalized state and action.
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # ---- pretrained weights -------------------------------------------------------------------
    # Directory in Hugging Face format holding the Qwen3-VL backbone (config.json, model.safetensors,
    # tokenizer and processor files), e.g. `pre.fast.1by1.2601091803.ckpt.ego200k.he30k`.
    vlm_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    # Directory holding `action_header.safetensors`; None trains the header from scratch.
    action_header_path: str | None = None
    # Load base weights in __init__. `from_pretrained` switches it off (the checkpoint holds them).
    load_base_weights: bool = True
    # Qwen3-VL architecture, filled from `vlm_path` on the first build so checkpoints are self-contained.
    vlm_config: dict | None = None
    # Tokenizer/image-processor directory; defaults to `vlm_path` (checkpoints carry a copy).
    processor_path: str | None = None

    # ---- action / state layout ----------------------------------------------------------------
    # Width of the header's action and state vectors (Psi0: `--model.action-dim`, `--model.odim`).
    model_action_dim: int = 36
    model_state_dim: int = 36
    # Slot of each dataset action/state dimension in the model vector (None = identity: 0..D-1).
    # Unused slots are zero; unused action slots are masked out of the loss (Psi0 `pad_to_len`).
    action_slots: list[int] | None = None
    state_slots: list[int] | None = None

    # ---- action header (ActionTransformerModel) ------------------------------------------------
    hidden_dim: int = 1536
    num_heads: int = 24
    attention_head_dim: int = 64
    num_blocks: int = 6
    view_feature_dim: int = 2048
    qk_norm: str | None = None
    final_layer_norm: bool = True
    # One VLM hidden-state index per header block (layer-wise fusion); None = last layer only.
    vlm_layer_indices: list[int] | None = None
    dropout: float = 0.1
    state_feature_dropout: float = 0.2
    state_drop_prob: float = 0.0
    state_as_action_token: bool = False
    state_null_token: bool = False
    combined_temb: bool = False
    pooled_projection_dim: int = 768
    # Frozen CLIP text encoder producing the pooled task embedding for `combined_temb`.
    pooled_text_encoder_path: str | None = None

    # ---- flow matching ------------------------------------------------------------------------
    train_diffusion_steps: int = 1000
    num_inference_steps: int = 10
    # Training-time real-time chunking: a random prefix of 0..max_delay-1 steps stays clean.
    rtc: bool = False
    max_delay: int = 8
    # Psi0 does not mask chunk steps past the episode end (LeRobot repeats the last action there).
    mask_padded_actions: bool = False

    # ---- inputs -------------------------------------------------------------------------------
    image_size: tuple[int, int] = (240, 320)  # (H, W) after resize + center crop
    img_aug: bool = True  # ColorJitter(0.2, (0.8, 1.2), (0.8, 1.2), 0.05), per view
    view_aug: bool = False  # RandomResizedCrop of 85-100 % of each side, aspect kept
    view_aug_min_scale: float = 0.85
    view_aug_prob: float = 1.0
    # Additive N(0, std) noise on the normalized state during training (SONIC "robust" recipe).
    state_noise_std: float = 0.0
    lowercase_instruction: bool = True

    # ---- VLM training -------------------------------------------------------------------------
    tune_vlm: bool = False
    lang_backbone_lr: float = 1e-6
    vision_tower_lr: float = 1e-5
    mm_projector_lr: float = 1e-4
    gradient_checkpointing: bool = False
    attn_implementation: str = "sdpa"

    # ---- optimizer / scheduler presets (Psi0 FinetuneTrainer) -----------------------------------
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    optimizer_grad_clip_norm: float = 1.0
    optimizer_foreach: bool | None = None
    # Counted in LeRobot scheduler steps, i.e. micro-batches: multiply by the accumulation steps.
    scheduler_warmup_steps: int = 1000

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError("`n_action_steps` must be <= `chunk_size`.")
        if self.vlm_layer_indices is not None and len(self.vlm_layer_indices) != self.num_blocks:
            raise ValueError(
                f"`vlm_layer_indices` needs one VLM layer per header block: got "
                f"{len(self.vlm_layer_indices)} indices for {self.num_blocks} blocks."
            )
        if self.combined_temb and not self.pooled_text_encoder_path:
            raise ValueError("`combined_temb` needs `pooled_text_encoder_path` (a CLIP text model).")
        if self.state_null_token and not self.state_as_action_token:
            raise ValueError("`state_null_token` requires `state_as_action_token`.")

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("Psi0 requires at least one visual input feature.")
        if self.action_feature is None:
            raise ValueError("Psi0 requires an action output feature.")
        action_dim = self.action_feature.shape[0]
        slots = self.action_slots if self.action_slots is not None else list(range(action_dim))
        if len(slots) != action_dim or len(set(slots)) != action_dim or max(slots) >= self.model_action_dim:
            raise ValueError(
                f"`action_slots` must place the {action_dim} action dims into distinct slots < "
                f"model_action_dim={self.model_action_dim}, got {slots}."
            )
        if self.robot_state_feature is not None:
            state_dim = self.robot_state_feature.shape[0]
            slots = self.state_slots if self.state_slots is not None else list(range(state_dim))
            if len(slots) != state_dim or len(set(slots)) != state_dim or max(slots) >= self.model_state_dim:
                raise ValueError(
                    f"`state_slots` must place the {state_dim} state dims into distinct slots < "
                    f"model_state_dim={self.model_state_dim}, got {slots}."
                )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
            foreach=self.optimizer_foreach,
        )

    def get_scheduler_preset(self) -> CosineAnnealingWithWarmupSchedulerConfig:
        # transformers `get_scheduler("cosine")`: linear warmup from 0, cosine to 0 at the last step.
        return CosineAnnealingWithWarmupSchedulerConfig(num_warmup_steps=self.scheduler_warmup_steps)

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
