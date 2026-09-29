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
from torch import nn

from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.vla_jepa.modeling_vla_jepa import module_lr_param_groups


def _model():
    model = nn.Module()
    model.qwen, model.action_model, model.video_encoder = nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 2)
    return model


def test_module_lrs_split_params_and_scale_together():
    model = _model()
    groups = module_lr_param_groups(model, {"qwen": 1e-5, "action_model": 1e-4})
    optimizer = torch.optim.AdamW(groups, lr=3e-5)
    assert [len(g["params"]) for g in optimizer.param_groups] == [2, 2, 2]
    assert [g["lr"] for g in optimizer.param_groups] == [3e-5, 1e-5, 1e-4]
    scheduler = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=0, num_decay_steps=10, peak_lr=3e-5, decay_lr=1e-5
    ).build(optimizer, num_training_steps=10)
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    # starVLA's cosine_with_min_lr: every group ends at min_lr / base_lr = 1/3 of its own peak.
    torch.testing.assert_close([g["lr"] for g in optimizer.param_groups], [1e-5, 1e-5 / 3, 1e-4 / 3])


def test_empty_module_lrs_keeps_single_group_and_typos_fail():
    model = _model()
    assert len(list(module_lr_param_groups(model, {}))) == 6
    with pytest.raises(ValueError, match="qwne"):
        module_lr_param_groups(model, {"qwne": 1e-5})
