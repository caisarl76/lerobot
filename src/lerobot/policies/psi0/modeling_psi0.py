# Copyright 2026 USC Physical Superintelligence Lab and The HuggingFace Inc. team. All rights reserved.
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
"""Native LeRobot Psi0 policy: Qwen3-VL backbone + flow-matching action header.

Follows `Psi0Model`, `FinetuneTrainer.forward_and_loss` and `Psi0ModelTransform` of
github.com/physical-superintelligence-lab/Psi0 (upstream commit of 2026-09, SONIC v1.1 release).
Differences: images are kept as tensors (no PIL round trip), normalization runs in the LeRobot
processor, and the action/state vectors are placed into the header's padded layout by `*_slots`.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import load_file
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import checkpoint_subdir, hub_load_kwargs
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor, CLIPTextModelWithProjection, CLIPTokenizer, Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
else:
    AutoProcessor = None
    CLIPTextModelWithProjection = None
    CLIPTokenizer = None
    Qwen3VLConfig = None
    Qwen3VLModel = None

from .action_header import ActionTransformerModel
from .configuration_psi0 import Psi0Config

logger = logging.getLogger(__name__)

VLM_PROCESSOR_DIRNAME = "vlm_processor"


def flow_sigmas(num_steps: int, num_train_timesteps: int = 1000) -> Tensor:
    """Sigmas of diffusers' `FlowMatchEulerDiscreteScheduler(num_train_timesteps).set_timesteps(num_steps)`
    (shift 1): linspace(1, 1/num_train_timesteps, num_steps) followed by 0."""
    sigmas = torch.linspace(1.0, 1.0 / num_train_timesteps, num_steps, dtype=torch.float64)
    return torch.cat([sigmas, torch.zeros(1, dtype=torch.float64)]).float()


def load_action_header_weights(
    header: nn.Module,
    state_dict: dict[str, Tensor],
    chunk_size: int,
    action_dim: int,
    mode: str = "official",
) -> tuple[list[str], list[str], list[str]]:
    """Load a released `action_header.safetensors`. Returns (missing, unexpected, skipped).

    - "official" (Psi0 `FinetuneTrainer.init_models`): the whole header when the chunk and action width match,
      otherwise only `transformer_blocks.*`.
    - "matching": every tensor whose shape matches; only the action-width layers (`action_proj_in.ac_proj`,
      `action_proj_out.linear`) are re-initialised when the width differs (e.g. 28D joints on the 80-D SONIC header).

    Keys whose shapes still differ are skipped (strict=False does not skip size mismatches)."""
    if mode not in ("official", "matching"):
        raise ValueError(f"unknown action header load mode {mode!r}")
    if mode == "official" and (
        state_dict["action_proj_in.dec_pos"].shape[0] != chunk_size
        or state_dict["action_proj_out.linear.weight"].shape[0] != action_dim
    ):
        state_dict = {k: v for k, v in state_dict.items() if k.startswith("transformer_blocks")}
        logger.warning("Psi0 action header size mismatch: loading only the transformer blocks.")
    own = header.state_dict()
    filtered, skipped = {}, []
    for key, value in state_dict.items():
        if key in own and own[key].shape != value.shape:
            skipped.append(f"{key}: {tuple(value.shape)} != {tuple(own[key].shape)}")
            continue
        filtered[key] = value
    missing, unexpected = header.load_state_dict(filtered, strict=False)
    return list(missing), list(unexpected), skipped


def _vlm_group(name: str) -> str:
    """Psi0 `_vlm_group_of`: patch merger(s) = projector, rest of `visual` = vision tower, else language."""
    if "merger" in name:
        return "mm_projector"
    if "visual" in name:
        return "vision_tower"
    return "lang_backbone"


class Psi0Model(nn.Module):
    def __init__(self, config: Psi0Config):
        super().__init__()
        require_package("transformers", extra="psi0")
        self.config = config
        vlm_dtype = torch.float32 if config.tune_vlm else torch.bfloat16
        if config.load_base_weights:
            self.vlm = Qwen3VLModel.from_pretrained(
                config.vlm_path, dtype=vlm_dtype, attn_implementation=config.attn_implementation
            )
            config.vlm_config = self.vlm.config.to_dict()
        else:
            if config.vlm_config is None:
                raise ValueError("Loading a Psi0 checkpoint needs `vlm_config` in its config.json.")
            vlm_config = Qwen3VLConfig(**config.vlm_config)
            vlm_config._attn_implementation = config.attn_implementation
            self.vlm = Qwen3VLModel._from_config(vlm_config, dtype=vlm_dtype)
        processor_path = config.processor_path or config.vlm_path
        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.processor.tokenizer.padding_side = "right"
        view_dim = self.vlm.config.text_config.hidden_size
        if view_dim != config.view_feature_dim:
            raise ValueError(f"VLM hidden size {view_dim} != view_feature_dim {config.view_feature_dim}")

        self.header = ActionTransformerModel(
            action_pred_horizon=config.chunk_size,
            action_dim=config.model_action_dim,
            odim=config.model_state_dim,
            view_feature_dim=config.view_feature_dim,
            action_hidden_dim=config.hidden_dim,
            num_attention_heads=config.num_heads,
            attention_head_dim=config.attention_head_dim,
            action_num_blocks=config.num_blocks,
            qk_norm=config.qk_norm,
            final_layer_norm=config.final_layer_norm,
            layerwise_vlm_fusion=config.vlm_layer_indices is not None,
            combined_temb=config.combined_temb,
            pooled_projection_dim=config.pooled_projection_dim,
            state_drop_prob=config.state_drop_prob,
            state_as_action_token=config.state_as_action_token,
            state_null_token=config.state_null_token,
            dropout=config.dropout,
            state_feature_dropout=config.state_feature_dropout,
        )
        if config.load_base_weights and config.action_header_path:
            path = Path(config.action_header_path) / "action_header.safetensors"
            missing, unexpected, skipped = load_action_header_weights(
                self.header,
                load_file(str(path)),
                config.chunk_size,
                config.model_action_dim,
                config.action_header_load,
            )
            logger.info(
                "Psi0 header from %s: %d missing, %d unexpected, %d skipped (shape): %s",
                path,
                len(missing),
                len(unexpected),
                len(skipped),
                skipped,
            )
            if unexpected:
                raise ValueError(f"Unexpected Psi0 header keys (architecture mismatch?): {unexpected[:10]}")

        # VLM trainability (Psi0 `apply_vlm_trainability`).
        self.vlm.requires_grad_(config.tune_vlm)
        if config.tune_vlm:
            # Upstream freezes the final norm ("unused"); kept for parity with the released recipe.
            self.vlm.language_model.norm.requires_grad_(False)
            if config.gradient_checkpointing:
                self.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                self.vlm.config.use_cache = False
                self.vlm.config.text_config.use_cache = False

        # Frozen CLIP text encoder for `combined_temb`; deliberately not a submodule (not saved).
        self._clip: list[Any] = []
        self._pooled_cache: dict[str, Tensor] = {}
        if config.combined_temb:
            tokenizer = CLIPTokenizer.from_pretrained(config.pooled_text_encoder_path)
            clip = CLIPTextModelWithProjection.from_pretrained(
                config.pooled_text_encoder_path, dtype=torch.bfloat16
            )
            clip.requires_grad_(False).eval()
            if clip.config.projection_dim != config.pooled_projection_dim:
                raise ValueError(
                    f"CLIP projection_dim {clip.config.projection_dim} != pooled_projection_dim "
                    f"{config.pooled_projection_dim}"
                )
            self._clip = [tokenizer, clip]

        slots = config.action_slots or list(range(config.action_feature.shape[0]))
        self.register_buffer("action_slots", torch.tensor(slots, dtype=torch.long), persistent=False)
        state_dim = config.robot_state_feature.shape[0] if config.robot_state_feature is not None else 0
        slots = config.state_slots or list(range(state_dim))
        self.register_buffer("state_slots", torch.tensor(slots, dtype=torch.long), persistent=False)

    # ---------------------------------------------------------------- inputs
    def _prepare_images(self, images: list[Tensor], train: bool) -> list[list[Tensor]]:
        """(B, C, H, W) float [0, 1] per camera -> per sample, per camera (C, H', W') after Psi0's transforms."""
        from torchvision.transforms import v2  # noqa: PLC0415

        height, width = self.config.image_size
        resized = [
            F.interpolate(img.float(), size=(height, width), mode="nearest")
            if tuple(img.shape[-2:]) != (height, width)
            else img.float()
            for img in images
        ]
        view_aug = None
        if train and self.config.view_aug:
            crop = v2.RandomResizedCrop(
                (height, width),
                scale=(self.config.view_aug_min_scale**2, 1.0),
                ratio=(width / height, width / height),
                interpolation=v2.InterpolationMode.BILINEAR,
                antialias=True,
            )
            view_aug = (
                crop
                if self.config.view_aug_prob >= 1.0
                else v2.RandomApply([crop], p=self.config.view_aug_prob)
            )
        jitter = None
        if train and self.config.img_aug:
            jitter = v2.ColorJitter(brightness=0.2, contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=0.05)
        batch = []
        for b in range(resized[0].shape[0]):
            views = []
            for cam in resized:
                img = cam[b]
                if view_aug is not None:
                    img = view_aug(img)
                if jitter is not None:
                    img = jitter(img)
                views.append(img.clamp(0.0, 1.0))
            batch.append(views)
        return batch

    def _vlm_inputs(self, images: list[list[Tensor]], instructions: list[str]) -> dict[str, Tensor]:
        messages = []
        for views, text in zip(images, instructions, strict=True):
            content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in views]
            content.append({"type": "text", "text": text})
            messages.append([{"role": "user", "content": content}])
        device = images[0][0].device
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            processor_kwargs={"padding": True, "return_tensors": "pt", "device": device, "do_rescale": False},
        )
        return {k: v.to(device) if isinstance(v, Tensor) else v for k, v in inputs.items()}

    def _instructions(self, batch: dict[str, Any], batch_size: int) -> list[str]:
        tasks = batch.get("task")
        if tasks is None:
            tasks = [""] * batch_size
        elif isinstance(tasks, str):
            tasks = [tasks] * batch_size
        tasks = [str(t) for t in tasks]
        return [t.lower() for t in tasks] if self.config.lowercase_instruction else tasks

    @torch.no_grad()
    def _pooled(self, instructions: list[str], device: torch.device) -> Tensor | None:
        if not self._clip:
            return None
        tokenizer, clip = self._clip
        if next(clip.parameters()).device != device:
            clip.to(device)
        missing = [s for s in dict.fromkeys(instructions) if s not in self._pooled_cache]
        if missing:
            toks = tokenizer(
                missing,
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).to(device)
            for s, e in zip(missing, clip(**toks).text_embeds, strict=True):
                self._pooled_cache[s] = e.clone()
        return torch.stack([self._pooled_cache[s].to(device) for s in instructions], dim=0).float()

    def _context(self, batch: dict[str, Any], train: bool) -> tuple[Tensor, Tensor, Tensor | None]:
        """VLM views (B, V, N, D) in float32, the VLM attention mask, and the pooled task embedding."""
        keys = list(self.config.image_features)
        images = self._prepare_images([batch[k] for k in keys], train)
        instructions = self._instructions(batch, len(images))
        inputs = self._vlm_inputs(images, instructions)
        grad = torch.is_grad_enabled() and self.config.tune_vlm and train
        with torch.set_grad_enabled(grad):
            out = self.vlm(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = out.hidden_states
        if self.config.vlm_layer_indices is None:
            views = hidden[-1].unsqueeze(1)
        else:
            views = torch.stack([hidden[i] for i in self.config.vlm_layer_indices], dim=1)
        pooled = self._pooled(instructions, views.device)
        return views.float(), inputs["attention_mask"], pooled

    def _state(self, batch: dict[str, Any], batch_size: int, device: torch.device, train: bool) -> Tensor:
        state = torch.zeros(batch_size, 1, self.config.model_state_dim, device=device)
        if OBS_STATE in batch and self.state_slots.numel():
            raw = batch[OBS_STATE].float()
            if raw.dim() == 3:
                raw = raw[:, -1]
            raw = raw.clamp(-1.0, 1.0)
            if train and self.config.state_noise_std > 0:
                raw = (raw + torch.randn_like(raw) * self.config.state_noise_std).clamp(-1.0, 1.0)
            state[:, 0, self.state_slots] = raw
        return state

    # ---------------------------------------------------------------- losses / sampling
    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        views, attn_mask, pooled = self._context(batch, train=self.training)
        actions = batch[ACTION].float().clamp(-1.0, 1.0)
        bsz, horizon, _ = actions.shape
        device = actions.device
        target_actions = torch.zeros(bsz, horizon, self.config.model_action_dim, device=device)
        target_actions[..., self.action_slots] = actions
        mask = torch.zeros_like(target_actions)
        mask[..., self.action_slots] = 1.0
        if self.config.mask_padded_actions and "action_is_pad" in batch:
            mask = mask * (~batch["action_is_pad"]).float()[..., None]
        states = self._state(batch, bsz, device, train=self.training)

        sigmas = torch.rand((bsz,), device=device)
        prefix_mask = None
        if self.config.rtc:
            delay = torch.randint(0, self.config.max_delay, (bsz,), device=device)
            prefix_mask = torch.arange(horizon, device=device)[None, :] < delay[:, None]
            sigmas = torch.where(prefix_mask, torch.zeros((), device=device), sigmas[:, None])
        timesteps = sigmas * self.config.train_diffusion_steps
        noise = torch.randn_like(target_actions)
        sig = sigmas
        while sig.dim() < target_actions.dim():
            sig = sig.unsqueeze(-1)
        noisy = (1 - sig) * target_actions + sig * noise
        target = noise - target_actions
        pred = self.header(
            noisy, timesteps, views, states, vlm_attn_mask=attn_mask, pooled_projections=pooled
        )
        loss = F.mse_loss(pred.float(), target.float(), reduction="none")
        if prefix_mask is not None:
            mask = mask * (~prefix_mask)[:, :, None].float()
        # Psi0: sum over time, mean over batch, weight 1/Da per dim, sum over dims.
        loss = ((loss * mask).sum(1).mean(0) / self.config.model_action_dim).sum()
        metrics = {"loss": loss.item()}
        if self.header.last_state_drop_frac is not None and self.training:
            metrics["state_drop_frac"] = self.header.last_state_drop_frac
        return loss, metrics

    @torch.no_grad()
    def sample(self, batch: dict[str, Any], noise: Tensor | None = None) -> Tensor:
        views, attn_mask, pooled = self._context(batch, train=False)
        bsz = views.shape[0]
        device = views.device
        states = self._state(batch, bsz, device, train=False)
        x = noise
        if x is None:
            x = torch.randn(bsz, self.config.chunk_size, self.config.model_action_dim, device=device)
        sigmas = flow_sigmas(self.config.num_inference_steps, self.config.train_diffusion_steps).to(device)
        for i in range(self.config.num_inference_steps):
            t = (sigmas[i] * self.config.train_diffusion_steps).expand(bsz)
            v = self.header(x, t, views, states, vlm_attn_mask=attn_mask, pooled_projections=pooled)
            x = x + (sigmas[i + 1] - sigmas[i]) * v.float()
        return x[..., self.action_slots]


class Psi0Policy(PreTrainedPolicy):
    config_class = Psi0Config
    name = "psi0"

    def __init__(self, config: Psi0Config, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.model = Psi0Model(config)
        self.reset()

    @classmethod
    def from_pretrained(
        cls, pretrained_name_or_path: str | Path, *, config: Psi0Config | None = None, **kwargs
    ):
        """Load a LeRobot Psi0 checkpoint: the base weights come from its own safetensors file."""
        hub = hub_load_kwargs(kwargs)
        if config is None:
            config = Psi0Config.from_pretrained(pretrained_name_or_path, **hub)
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
        """Psi0 `create_optimizers`: action header at the base lr, VLM components at their own lr."""
        groups: dict[str, list[nn.Parameter]] = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            group = _vlm_group(name.removeprefix("vlm.")) if name.startswith("vlm.") else "action_header"
            groups.setdefault(group, []).append(param)
        lrs = {
            "lang_backbone": self.config.lang_backbone_lr,
            "vision_tower": self.config.vision_tower_lr,
            "mm_projector": self.config.mm_projector_lr,
        }
        params = [{"params": groups.pop("action_header", [])}]
        for name, lr in lrs.items():
            if name in groups:
                params.append({"params": groups[name], "lr": lr})
        return params

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
