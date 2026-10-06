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
"""Psi0 processors: Psi0's "bounds" normalization is LeRobot MIN_MAX (or QUANTILES), both to [-1, 1].

Image transforms, VLM tokenization, padding into the header layout and clipping to [-1, 1] run in the
model (`Psi0Model`), so the pipeline itself is the standard scaffold.
"""

from __future__ import annotations

from typing import Any

import torch

from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.factory import make_default_pre_post_processors

from .configuration_psi0 import Psi0Config


def make_psi0_pre_post_processors(
    config: Psi0Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    return make_default_pre_post_processors(config, dataset_stats)
