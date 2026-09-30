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
"""AdamW for bf16 weights with stochastic rounding.

Plain `torch.optim.AdamW` on bf16 parameters rounds every update to the nearest bf16 value. At fine-tuning
learning rates most updates are smaller than half a bf16 ULP (for a weight of 0.02 the ULP is 1.2e-4, an
Adam step at lr 2e-5 is ~2e-5), so they are silently dropped. Here the update is computed in float32 and
written back with stochastic rounding, which is unbiased: E[round(w + u)] = w + u. Moments are kept in
`state_dtype` (float32 by default; bfloat16 moments are also written with stochastic rounding). This keeps a
5B-parameter model at 2 bytes of weights + 8 (or 4) bytes of moments per trainable parameter.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch


def stochastic_round_to_bf16(x: torch.Tensor) -> torch.Tensor:
    """Round a float32 tensor to bfloat16, up or down with probability given by the truncated fraction."""
    if x.dtype != torch.float32:
        raise TypeError(f"expected float32, got {x.dtype}")
    bits = x.contiguous().view(torch.int32)
    noise = torch.randint_like(bits, 0, 1 << 16)
    rounded = (bits + noise) & -65536  # keep the upper 16 bits (0xFFFF0000)
    return rounded.view(torch.float32).to(torch.bfloat16)


def _store(target: torch.Tensor, value: torch.Tensor) -> None:
    if target.dtype == torch.bfloat16:
        target.copy_(stochastic_round_to_bf16(value))
    else:
        target.copy_(value)


class AdamWStochasticRounding(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        state_dtype: torch.dtype = torch.float32,
    ):
        if state_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"state_dtype must be float32 or bfloat16, got {state_dtype}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        self.state_dtype = state_dtype

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, weight_decay = group["lr"], group["eps"], group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("AdamWStochasticRounding does not support sparse gradients")
                grad = param.grad.float()
                state = self.state[param]
                if not state:
                    state["step"] = torch.zeros((), dtype=torch.float32)
                    state["exp_avg"] = torch.zeros_like(param, dtype=self.state_dtype)
                    state["exp_avg_sq"] = torch.zeros_like(param, dtype=self.state_dtype)
                state["step"] += 1
                step = float(state["step"])
                exp_avg = state["exp_avg"].float()
                exp_avg_sq = state["exp_avg_sq"].float()
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if self.state_dtype != torch.float32:
                    _store(state["exp_avg"], exp_avg)
                    _store(state["exp_avg_sq"], exp_avg_sq)
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                denom = (exp_avg_sq / bias_correction2).sqrt_().add_(eps)
                updated = param.float()
                updated.mul_(1 - lr * weight_decay)
                updated.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)
                _store(param, updated)
        return loss
