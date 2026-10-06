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
"""Configuration of the native LeRobot port of Xiaomi-Robotics-1 (XR-1, arXiv 2607.15330).

XR-1 is a Mixture-of-Transformers VLA: Qwen3-VL-4B plus a 36-layer DiT (hidden 1024) whose layers attend to
the VLM's per-layer key/value cache, a VLM-side "choice policy" head (5 action hypotheses + scores), a
frequency-domain flow loss and asynchronous (action-prefix) training. Defaults follow the released
post-training recipe (`xr1/configs/{model/posttrain,data/load_washer,trainer/deepspeed}.yaml`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWSRConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("xiaomi_robotics")
@dataclass
class XiaomiRoboticsConfig(PreTrainedConfig):
    n_obs_steps: int = 1
    chunk_size: int = 30
    n_action_steps: int = 30

    # XR-1 normalizes inside the model (per-step mean/std of the packed relative actions, q01/q99 of the
    # state), so the LeRobot normalizer passes everything through.
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # ---- pretrained weights -------------------------------------------------------------------
    # `model_states.pt` of XiaomiRobotics/Xiaomi-Robotics-1-5B ({"module": {"model.<name>": tensor}}).
    pretrained_weights_path: str | None = None
    # Qwen3-VL-4B config, tokenizer and image processor (weights come from `pretrained_weights_path`).
    vlm_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    load_base_weights: bool = True
    vlm_config: dict | None = None
    processor_path: str | None = None
    attn_implementation: str = "sdpa"

    # ---- action / state layout ----------------------------------------------------------------
    # The released model has a (1, 60) state and (30, 60) actions. Wider actions (e.g. 78) resize the action
    # in/out layers: the first 60 dims keep their pretrained weights, the rest are initialised like XR-1.
    model_action_dim: int = 60
    model_state_dim: int = 60
    action_slots: list[int] | None = None
    state_slots: list[int] | None = None
    # For each dataset action dim, the dataset state dim it is made relative to (a - s_t), or -1 to keep
    # it absolute (XR-1 packs relative end-effector/gripper deltas; SONIC tokens have no state).
    relative_action_state_indices: list[int] | None = None
    # JSON from `examples/g1_dex3_training/xr1_action_stats.py` (per-step mean/std, state q01/q99).
    stats_path: str | None = None

    # ---- model --------------------------------------------------------------------------------
    dit_hidden_size: int = 1024
    n_choices: int = 5
    training_repeat: int = 4
    num_inference_steps: int = 5
    async_train: bool = True
    prefix_mask_prob: float = 0.5
    freq_coefficient: float = 1.0
    freq_excluded_dims: list[int] = field(default_factory=list)
    ffn_gradient_checkpointing: bool = True
    vision_gradient_checkpointing: bool = True

    # ---- inputs -------------------------------------------------------------------------------
    view_names: list[str] = field(default_factory=lambda: ["Ego View"])
    image_max_pixels: int = 160000
    image_factor: int = 32
    color_aug: bool = True

    # ---- optimizer / scheduler presets (xr1/configs/trainer/deepspeed.yaml) ---------------------
    optimizer_lr: float = 2e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.1
    optimizer_grad_clip_norm: float = 1.0
    # Moments dtype of the stochastic-rounding AdamW used for the bf16 weights ("float32" or "bfloat16").
    optimizer_state_dtype: str = "float32"
    # Counted in LeRobot scheduler steps (micro-batches): multiply by the gradient-accumulation steps.
    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 10_000
    scheduler_decay_lr: float = 5e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError("`n_action_steps` must be <= `chunk_size`.")
        if self.chunk_size > 60:
            raise ValueError("XR-1 has 60 action query tokens (<a_0>..<a_59>): chunk_size must be <= 60.")

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("XR-1 requires at least one visual input feature.")
        if len(self.view_names) != len(self.image_features):
            raise ValueError(
                f"`view_names` ({self.view_names}) needs one name per camera ({list(self.image_features)})."
            )
        if self.action_feature is None:
            raise ValueError("XR-1 requires an action output feature.")
        action_dim = self.action_feature.shape[0]
        slots = self.action_slots if self.action_slots is not None else list(range(action_dim))
        if len(slots) != action_dim or len(set(slots)) != action_dim or max(slots) >= self.model_action_dim:
            raise ValueError(
                f"`action_slots` must place {action_dim} dims below {self.model_action_dim}: {slots}"
            )
        if self.robot_state_feature is None:
            raise ValueError("XR-1 requires `observation.state`.")
        state_dim = self.robot_state_feature.shape[0]
        slots = self.state_slots if self.state_slots is not None else list(range(state_dim))
        if len(slots) != state_dim or len(set(slots)) != state_dim or max(slots) >= self.model_state_dim:
            raise ValueError(
                f"`state_slots` must place {state_dim} dims below {self.model_state_dim}: {slots}"
            )
        rel = self.relative_action_state_indices
        if rel is not None and (len(rel) != action_dim or any(i >= state_dim for i in rel)):
            raise ValueError(
                f"`relative_action_state_indices` needs {action_dim} entries < {state_dim}: {rel}"
            )

    def get_optimizer_preset(self) -> AdamWSRConfig:
        return AdamWSRConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
            state_dtype=self.optimizer_state_dtype,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
