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
"""Tiny random Qwen3-VL / CLIP checkpoints with the real processor files, for Psi0 and XR-1 tests.

Only the tokenizer and image-processor files are fetched (a few MB); the tests skip when they are not available.
"""

from __future__ import annotations

from pathlib import Path

import pytest

QWEN3VL_PROCESSOR_REPO = "Qwen/Qwen3-VL-2B-Instruct"
CLIP_TOKENIZER_REPO = "openai/clip-vit-large-patch14"


def tiny_qwen3vl_config(hidden: int = 64, layers: int = 4):
    from transformers import Qwen3VLConfig

    return Qwen3VLConfig(
        text_config={
            "hidden_size": hidden,
            "intermediate_size": 2 * hidden,
            "num_hidden_layers": layers,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 151936,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True},
            "rope_theta": 5000000,
            "max_position_embeddings": 4096,
            "tie_word_embeddings": True,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": hidden,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "deepstack_visual_indexes": [0],
            "num_position_embeddings": 2304,
        },
        tie_word_embeddings=True,
    )


def build_tiny_qwen3vl(out: Path, hidden: int = 64, layers: int = 4) -> Path:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    try:
        processor = AutoProcessor.from_pretrained(QWEN3VL_PROCESSOR_REPO)
    except Exception as error:  # offline without a cached processor
        pytest.skip(f"Qwen3-VL processor files unavailable: {error}")
    torch.manual_seed(0)
    out.mkdir(parents=True, exist_ok=True)
    Qwen3VLForConditionalGeneration(tiny_qwen3vl_config(hidden, layers)).save_pretrained(out)
    processor.save_pretrained(out)
    return out


def build_tiny_clip(out: Path, projection_dim: int = 32) -> Path:
    from transformers import CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizer

    try:
        tokenizer = CLIPTokenizer.from_pretrained(CLIP_TOKENIZER_REPO)
    except Exception as error:
        pytest.skip(f"CLIP tokenizer unavailable: {error}")
    out.mkdir(parents=True, exist_ok=True)
    config = CLIPTextConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, projection_dim=projection_dim
    )
    CLIPTextModelWithProjection(config).save_pretrained(out)
    tokenizer.save_pretrained(out)
    return out
