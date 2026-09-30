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

from lerobot.optim.adamw_sr import AdamWStochasticRounding, stochastic_round_to_bf16
from lerobot.optim.optimizers import AdamWSRConfig


def test_stochastic_rounding_is_unbiased_and_on_grid():
    torch.manual_seed(0)
    lo = torch.tensor(0.02, dtype=torch.bfloat16).float()
    hi = torch.nextafter(
        torch.tensor(0.02, dtype=torch.bfloat16), torch.tensor(1.0, dtype=torch.bfloat16)
    ).float()
    x = torch.full((200_000,), float(lo + 0.3 * (hi - lo)))
    rounded = stochastic_round_to_bf16(x).float()
    assert set(rounded.unique().tolist()) <= {lo.item(), hi.item()}
    assert rounded.mean().item() == pytest.approx(x[0].item(), rel=1e-4)
    negative = stochastic_round_to_bf16(-x).float()
    assert negative.mean().item() == pytest.approx(-x[0].item(), rel=1e-4)


def test_matches_torch_adamw_for_float32_params():
    torch.manual_seed(0)
    reference = torch.nn.Linear(8, 4)
    ours = torch.nn.Linear(8, 4)
    ours.load_state_dict(reference.state_dict())
    kwargs = {"lr": 1e-2, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.1}
    opt_ref = torch.optim.AdamW(reference.parameters(), **kwargs)
    opt = AdamWStochasticRounding(ours.parameters(), **kwargs)
    for _ in range(5):
        x = torch.randn(16, 8)
        for model, optimizer in ((reference, opt_ref), (ours, opt)):
            optimizer.zero_grad()
            model(x).pow(2).mean().backward()
            optimizer.step()
    for a, b in zip(reference.parameters(), ours.parameters(), strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("state_dtype", ["float32", "bfloat16"])
def test_bf16_updates_below_half_ulp_are_not_lost(state_dtype):
    """At lr 2e-5 a plain bf16 AdamW leaves weights of ~0.02 unchanged; stochastic rounding moves them."""
    torch.manual_seed(0)
    start = torch.full((50_000,), 0.02)
    plain = torch.nn.Parameter(start.to(torch.bfloat16))
    ours = torch.nn.Parameter(start.to(torch.bfloat16))
    opt_plain = torch.optim.AdamW([plain], lr=2e-5, weight_decay=0.0)
    opt = AdamWSRConfig(lr=2e-5, weight_decay=0.0, state_dtype=state_dtype).build([ours])
    for _ in range(20):
        for param, optimizer in ((plain, opt_plain), (ours, opt)):
            param.grad = torch.ones_like(param)
            optimizer.step()
    assert torch.equal(plain.detach(), start.to(torch.bfloat16))  # every update rounded away
    expected = 0.02 - 20 * 2e-5
    assert ours.detach().float().mean().item() == pytest.approx(expected, abs=2e-5)
