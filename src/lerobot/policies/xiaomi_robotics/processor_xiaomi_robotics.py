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
"""XR-1 processors: identity normalization (XR-1 normalizes inside the model with per-step stats).

Relative actions, per-step mean/std, state q01/q99, image resizing and tokenization run in `XR1Model` with
buffers saved in the checkpoint, so the pipeline is the standard scaffold with IDENTITY modes.
"""

from __future__ import annotations

from typing import Any

import torch

from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.factory import make_default_pre_post_processors

from .configuration_xiaomi_robotics import XiaomiRoboticsConfig


def make_xiaomi_robotics_pre_post_processors(
    config: XiaomiRoboticsConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    return make_default_pre_post_processors(config, dataset_stats)
