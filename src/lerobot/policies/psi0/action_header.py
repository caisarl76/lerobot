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
"""Psi0 action header (System-1), ported from `psi/models/psi0.py` of github.com/physical-superintelligence-lab/Psi0.

Only the paths used by the released fine-tuning recipes are kept (no ResNet trajectory conditions, no FiLM,
no perceiver fusion, no DiT variant). Module and parameter names are unchanged so the released
`action_header.safetensors` files load as they are.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from lerobot.utils.import_utils import _diffusers_available, require_package

if TYPE_CHECKING or _diffusers_available:
    from diffusers.models.attention import FeedForward
    from diffusers.models.attention_processor import Attention
    from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings
else:
    FeedForward = None
    Attention = None
    CombinedTimestepTextProjEmbeddings = object


class TimeNetwork(nn.Module):
    """Sinusoidal timestep features + MLP (`_TimeNetwork`). Accepts (B,) or per-token (B, T) timesteps."""

    def __init__(self, time_dim: int, out_dim: int):
        super().__init__()
        if time_dim % 2:
            raise ValueError("time_dim must be even")
        half_dim = time_dim // 2
        w = np.log(10000) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.w = nn.Parameter(w, requires_grad=False)
        self.out_net = nn.Sequential(nn.Linear(time_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[..., None] * self.w
        x = torch.cat((torch.cos(x), torch.sin(x)), dim=-1)
        return self.out_net(x)


class CombinedTimestepTextProjEmbeddingsPerToken(CombinedTimestepTextProjEmbeddings):
    """SD3 `combined_temb` embedding that also accepts per-token (B, T) timesteps."""

    def forward(self, timestep: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        pooled = self.text_embedder(pooled_projection)
        if timestep.dim() == 1:
            proj = self.time_proj(timestep)
            return self.timestep_embedder(proj.to(dtype=pooled_projection.dtype)) + pooled
        if timestep.dim() == 2:
            batch, horizon = timestep.shape
            proj = self.time_proj(timestep.reshape(batch * horizon))
            temb = self.timestep_embedder(proj.to(dtype=pooled_projection.dtype))
            return temb.reshape(batch, horizon, -1) + pooled.unsqueeze(1)
        raise ValueError(f"timestep must be 1-D or 2-D, got {tuple(timestep.shape)}")


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine=False, bias=True)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        if conditioning_embedding.dim() == 2:
            conditioning_embedding = conditioning_embedding.unsqueeze(1)
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class AdaLayerNormZero(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        if emb.dim() == 2:
            emb = emb.unsqueeze(1)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = nn.Parameter(pe.unsqueeze(0).transpose(0, 1), requires_grad=False)  # (max_len, 1, d)


class JointVLAAttnProcessor:
    """SD3-style joint attention over [action tokens, context tokens] with a key-padding mask on the context."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        batch_size = hidden_states.shape[0]
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        head_dim = key.shape[-1] // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        attn_mask = None
        if encoder_hidden_states is not None:
            ctx_q = (
                attn.add_q_proj(encoder_hidden_states)
                .view(batch_size, -1, attn.heads, head_dim)
                .transpose(1, 2)
            )
            ctx_k = (
                attn.add_k_proj(encoder_hidden_states)
                .view(batch_size, -1, attn.heads, head_dim)
                .transpose(1, 2)
            )
            ctx_v = (
                attn.add_v_proj(encoder_hidden_states)
                .view(batch_size, -1, attn.heads, head_dim)
                .transpose(1, 2)
            )
            if attn.norm_added_q is not None:
                ctx_q = attn.norm_added_q(ctx_q)
            if attn.norm_added_k is not None:
                ctx_k = attn.norm_added_k(ctx_k)
            query = torch.cat([query, ctx_q], dim=2)
            key = torch.cat([key, ctx_k], dim=2)
            value = torch.cat([value, ctx_v], dim=2)
            if attention_mask is not None:
                attn_mask = torch.cat(
                    [
                        torch.ones(
                            batch_size,
                            1,
                            1,
                            hidden_states.shape[1],
                            device=attention_mask.device,
                            dtype=torch.bool,
                        ),
                        attention_mask.to(torch.bool)[:, None, None, :],
                    ],
                    dim=-1,
                )

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attn_mask
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states, encoder_hidden_states


class ObservationProjection(nn.Module):
    """Projects VLM hidden states (and, unless the state is an action token, the proprio state) to context tokens."""

    def __init__(
        self,
        output_dim: int,
        odim: int,
        view_feature_dim: int,
        add_state_token: bool,
        dropout: float,
        state_feature_dropout: float,
    ):
        super().__init__()
        self.enc_pos = PositionalEncoding(d_model=output_dim)
        self.views_proj = nn.Linear(view_feature_dim, output_dim, bias=True)
        self.add_state_token = add_state_token
        # Kept even when the state is an action token: that path reuses the Linear `_obs_proc[1]`.
        self._obs_proc = nn.Sequential(nn.Dropout(p=state_feature_dropout), nn.Linear(odim, output_dim))
        self.post_proc = nn.Sequential(nn.Identity(), nn.Identity(), nn.Dropout(dropout))

    def forward(
        self,
        views: torch.Tensor,
        obs: torch.Tensor,
        vlm_attn_mask: torch.Tensor | None = None,
        obs_keep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """views (B, V, N, Dv), obs (B, 1, odim) -> tokens (B, V*N [+1], D) and the context mask."""
        view_tokens = self.views_proj(views)
        batch, n_views, n_tokens, dim = view_tokens.shape
        tokens = view_tokens.view(batch, n_views * n_tokens, dim)
        if self.add_state_token:
            obs_token = self._obs_proc(obs)
            tokens = torch.cat((tokens, obs_token), 1)
            obs_mask = torch.ones((batch, obs_token.shape[1]), device=tokens.device, dtype=torch.float32)
            if obs_keep is not None:
                obs_mask = obs_mask * obs_keep.view(batch, 1).to(obs_mask.dtype)
            if vlm_attn_mask is not None:
                vlm_attn_mask = torch.cat([vlm_attn_mask.to(obs_mask.dtype), obs_mask], 1)
            elif obs_keep is not None:
                n_ctx = tokens.shape[1] - obs_token.shape[1]
                vlm_attn_mask = torch.cat(
                    [torch.ones((batch, n_ctx), device=tokens.device, dtype=torch.float32), obs_mask], 1
                )
            tokens = self.post_proc(tokens)
            # Only the observation tokens get positional encodings (added after the context dropout,
            # as upstream `_add_obs_pos`); VLM tokens keep the VLM's own RoPE positions.
            n_obs = obs_token.shape[1]
            pos = self.enc_pos.pe[:n_obs, 0].to(tokens.dtype)  # (n_obs, D)
            return torch.cat([tokens[:, :-n_obs], tokens[:, -n_obs:] + pos[None]], dim=1), vlm_attn_mask
        if vlm_attn_mask is not None:
            vlm_attn_mask = vlm_attn_mask.to(torch.float32)
        return self.post_proc(tokens), vlm_attn_mask


class ActionProjectionIn(nn.Module):
    def __init__(self, action_pred_horizon: int, action_dim: int, output_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.ac_proj = nn.Sequential(
            nn.Linear(action_dim, action_dim), nn.GELU(approximate="tanh"), nn.Linear(action_dim, output_dim)
        )
        self.dec_pos = nn.Parameter(torch.empty(action_pred_horizon, output_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.dec_pos.data)

    def forward(self, noisy_actions: torch.Tensor) -> torch.Tensor:
        batch = noisy_actions.shape[0]
        return self.ac_proj(noisy_actions.reshape(batch, -1, self.action_dim)) + self.dec_pos.unsqueeze(0)


class ActionProjectionOut(nn.Module):
    def __init__(self, hidden_size: int, action_dim: int, final_layer_norm: bool = True):
        super().__init__()
        self.final_layer_norm = final_layer_norm
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, action_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))  # noqa: N815
        if final_layer_norm:
            nn.init.zeros_(self.adaLN_modulation[-1].weight)
            nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            t = t.unsqueeze(1)
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=-1)
        x = self.norm_final(x) * (1 + scale) + shift if self.final_layer_norm else x * scale + shift
        return self.linear(x)


class VLATransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        context_pre_only: bool = False,
        qk_norm: str | None = None,
    ):
        super().__init__()
        self.context_pre_only = context_pre_only
        self.norm1_act = AdaLayerNormZero(dim)
        self.norm1_obs = AdaLayerNormContinuous(dim, dim) if context_pre_only else AdaLayerNormZero(dim)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=context_pre_only,
            bias=True,
            processor=JointVLAAttnProcessor(),
            qk_norm=qk_norm,
            eps=1e-6,
        )
        self.norm2_act = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_act = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        if not context_pre_only:
            self.norm2_obs = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_obs = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        else:
            self.norm2_obs = None
            self.ff_obs = None

    def forward(
        self,
        action_hidden_states: torch.Tensor,
        obs_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        obs_token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        norm_act, gate_msa_act, shift_mlp_act, scale_mlp_act, gate_mlp_act = self.norm1_act(
            action_hidden_states, emb=temb
        )
        obs_temb = temb[:, -1] if temb.dim() > 2 else temb
        if self.context_pre_only:
            norm_obs = self.norm1_obs(obs_hidden_states, obs_temb)
            gate_msa_obs = shift_mlp_obs = scale_mlp_obs = gate_mlp_obs = None
        else:
            norm_obs, gate_msa_obs, shift_mlp_obs, scale_mlp_obs, gate_mlp_obs = self.norm1_obs(
                obs_hidden_states, emb=obs_temb
            )
        act_attn, obs_attn = self.attn(
            hidden_states=norm_act, encoder_hidden_states=norm_obs, attention_mask=obs_token_mask
        )
        action_hidden_states = action_hidden_states + gate_msa_act * act_attn
        norm_act = self.norm2_act(action_hidden_states) * (1 + scale_mlp_act) + shift_mlp_act
        action_hidden_states = action_hidden_states + gate_mlp_act * self.ff_act(norm_act)
        if self.context_pre_only:
            return action_hidden_states, None
        obs_hidden_states = obs_hidden_states + gate_msa_obs * obs_attn
        norm_obs = self.norm2_obs(obs_hidden_states) * (1 + scale_mlp_obs) + shift_mlp_obs
        obs_hidden_states = obs_hidden_states + gate_mlp_obs * self.ff_obs(norm_obs)
        return action_hidden_states, obs_hidden_states


class ActionTransformerModel(nn.Module):
    """Flow-matching action header. Predicts the velocity (noise - action) for a noisy (B, Tp, Da) chunk."""

    def __init__(
        self,
        action_pred_horizon: int,
        action_dim: int,
        odim: int,
        view_feature_dim: int = 2048,
        action_hidden_dim: int = 1536,
        num_attention_heads: int = 24,
        attention_head_dim: int = 64,
        action_num_blocks: int = 6,
        qk_norm: str | None = None,
        final_layer_norm: bool = True,
        layerwise_vlm_fusion: bool = False,
        combined_temb: bool = False,
        pooled_projection_dim: int = 768,
        state_drop_prob: float = 0.0,
        state_as_action_token: bool = False,
        state_null_token: bool = False,
        dropout: float = 0.1,
        state_feature_dropout: float = 0.2,
    ):
        super().__init__()
        require_package("diffusers", extra="psi0")
        inner_dim = num_attention_heads * attention_head_dim
        self.state_drop_prob = state_drop_prob
        self.state_as_action_token = state_as_action_token
        self.combined_temb = combined_temb
        self.last_state_drop_frac: float | None = None
        if combined_temb:
            self.time_ins_embed = CombinedTimestepTextProjEmbeddingsPerToken(
                embedding_dim=inner_dim, pooled_projection_dim=pooled_projection_dim
            )
        else:
            self.time_ins_embed = TimeNetwork(time_dim=256, out_dim=action_hidden_dim)
        self.obs_proj = ObservationProjection(
            output_dim=action_hidden_dim,
            odim=odim,
            view_feature_dim=view_feature_dim,
            add_state_token=not state_as_action_token,
            dropout=dropout,
            state_feature_dropout=state_feature_dropout,
        )
        if state_as_action_token:
            self.state_pos = nn.Parameter(torch.zeros(1, 1, action_hidden_dim))
        self.state_null = None
        if state_as_action_token and state_null_token:
            self.state_null = nn.Parameter(torch.randn(1, 1, action_hidden_dim) * 0.02)
        self.action_proj_in = ActionProjectionIn(action_pred_horizon, action_dim, action_hidden_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                VLATransformerBlock(
                    dim=action_hidden_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    context_pre_only=(layerwise_vlm_fusion or i == action_num_blocks - 1),
                    qk_norm=qk_norm,
                )
                for i in range(action_num_blocks)
            ]
        )
        self.action_proj_out = ActionProjectionOut(action_hidden_dim, action_dim, final_layer_norm)

    def _temb(self, timestep: torch.Tensor, pooled_projections: torch.Tensor | None) -> torch.Tensor:
        if self.combined_temb:
            return self.time_ins_embed(timestep, pooled_projections)
        return self.time_ins_embed(timestep)

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        views: torch.Tensor,
        obs: torch.Tensor,
        vlm_attn_mask: torch.Tensor | None = None,
        pooled_projections: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """noisy_action (B, Tp, Da), timestep (B,) or (B, Tp) in [0, 1000], views (B, V, N, Dv), obs (B, 1, odim)."""
        temb = self._temb(timestep, pooled_projections)
        action_hidden_states = self.action_proj_in(noisy_action)
        obs_keep = None
        if self.state_as_action_token:
            state_vec = obs[:, -1] if obs.dim() == 3 else obs
            keep = None
            if self.training and self.state_drop_prob > 0.0:
                keep = torch.rand(state_vec.shape[0], device=state_vec.device) >= self.state_drop_prob
                self.last_state_drop_frac = float((~keep).float().mean())
                if self.state_null is None:
                    state_vec = state_vec * keep.to(state_vec.dtype)[:, None]
            state_token = self.obs_proj._obs_proc[1](state_vec.to(action_hidden_states.dtype))[:, None]
            if self.training and self.state_null is not None:
                if keep is None:
                    keep = torch.ones(state_vec.shape[0], dtype=torch.bool, device=state_vec.device)
                state_token = torch.where(
                    keep[:, None, None], state_token, self.state_null.to(state_token.dtype)
                )
            state_token = state_token + self.state_pos
            action_hidden_states = torch.cat([state_token, action_hidden_states], dim=1)
            if temb.dim() == 3:
                t0 = torch.zeros(temb.shape[0], device=temb.device, dtype=timestep.dtype)
                temb = torch.cat([self._temb(t0, pooled_projections)[:, None].to(temb.dtype), temb], dim=1)
        elif self.training and self.state_drop_prob > 0.0:
            obs_keep = (torch.rand(obs.shape[0], device=obs.device) >= self.state_drop_prob).to(torch.float32)

        layerwise = views.shape[1] > 1
        if layerwise:
            if views.shape[1] != len(self.transformer_blocks):
                raise ValueError(
                    f"layer-wise fusion got {views.shape[1]} VLM layers for {len(self.transformer_blocks)} blocks"
                )
            per_block_obs = []
            obs_token_mask = None
            for i in range(views.shape[1]):
                obs_i, obs_token_mask = self.obs_proj(views[:, i : i + 1], obs, vlm_attn_mask, obs_keep)
                per_block_obs.append(obs_i)
        else:
            obs_hidden_states, obs_token_mask = self.obs_proj(views, obs, vlm_attn_mask, obs_keep)

        for index, block in enumerate(self.transformer_blocks):
            obs_in = per_block_obs[index] if layerwise else obs_hidden_states
            action_hidden_states, obs_out = block(action_hidden_states, obs_in, temb, obs_token_mask)
            if not layerwise:
                obs_hidden_states = obs_out

        if self.state_as_action_token:
            action_hidden_states = action_hidden_states[:, 1:]
            if temb.dim() == 3:
                temb = temb[:, 1:]
        return self.action_proj_out(action_hidden_states, temb)
