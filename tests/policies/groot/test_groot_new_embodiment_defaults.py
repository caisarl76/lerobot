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

"""Fine-tuning a new embodiment from the Hub id preprocesses like Isaac-GR00T's launch_finetune.

With ``base_model_path="nvidia/GR00T-N1.7-3B"`` no checkpoint sidecars are read, so the processor
falls back to the base checkpoint's processor_config.json: q01/q99 normalization, raw-state dropout
0.2, letterboxed albumentations geometry (shortest edge 256, crop 0.95) and train-time ColorJitter.
"""

import random

import numpy as np
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.processor_groot import (
    GrootActionUnpackUnnormalizeStep,
    GrootN17PackInputsStep,
    GrootN17VLMEncodeStep,
    make_groot_pre_post_processors,
)
from lerobot.utils.constants import ACTION, OBS_STATE

STATS = {
    OBS_STATE: {
        "min": torch.full((2,), -4.0),
        "max": torch.full((2,), 4.0),
        "q01": torch.full((2,), -1.0),
        "q99": torch.full((2,), 1.0),
    },
    ACTION: {
        "min": torch.full((3,), -4.0),
        "max": torch.full((3,), 4.0),
        "q01": torch.full((3,), -1.0),
        "q99": torch.full((3,), 1.0),
    },
}


def _processors():
    config = GrootConfig(
        base_model_path="nvidia/GR00T-N1.7-3B",
        device="cpu",
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
            "observation.images.ego": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    return make_groot_pre_post_processors(config, dataset_stats=STATS)


def _step(preprocessor, cls):
    return next(step for step in preprocessor.steps if isinstance(step, cls))


def test_hub_id_fallback_uses_base_checkpoint_processor_settings():
    pre, post = _processors()
    pack, vlm = _step(pre, GrootN17PackInputsStep), _step(pre, GrootN17VLMEncodeStep)
    unpack = _step(post, GrootActionUnpackUnnormalizeStep)

    assert pack.use_percentiles and unpack.use_percentiles
    assert pack.state_dropout_prob == 0.2
    assert pack.embodiment_mapping[pack.embodiment_tag] == 10
    assert (vlm.use_albumentations, vlm.letter_box_transform) == (True, True)
    assert (vlm.shortest_image_edge, vlm.crop_fraction) == (256, 0.95)
    assert vlm.color_jitter_params["hue"] == 0.08
    assert GrootN17VLMEncodeStep(**vlm.get_config()).color_jitter_params == vlm.color_jitter_params
    assert GrootActionUnpackUnnormalizeStep(**unpack.get_config()).use_percentiles


def test_percentile_normalization_round_trips_through_q01_q99():
    pre, post = _processors()
    pack, unpack = _step(pre, GrootN17PackInputsStep), _step(post, GrootActionUnpackUnnormalizeStep)
    action = torch.tensor([[[-1.0, 0.5, 3.0]]])  # 3.0 is beyond q99 and gets clipped
    out = pack(
        {
            TransitionKey.OBSERVATION: {OBS_STATE: torch.tensor([[0.5, -0.5]])},
            TransitionKey.ACTION: action,
            TransitionKey.COMPLEMENTARY_DATA: {"task": ["push"]},
        }
    )
    torch.testing.assert_close(out[TransitionKey.OBSERVATION]["state"][0, 0, :2], torch.tensor([0.5, -0.5]))
    normalized = out[TransitionKey.ACTION][:, :1, :3]
    torch.testing.assert_close(normalized, torch.tensor([[[-1.0, 0.5, 1.0]]]))
    decoded = unpack({TransitionKey.ACTION: normalized})[TransitionKey.ACTION]
    torch.testing.assert_close(decoded, torch.tensor([[[-1.0, 0.5, 1.0]]]))


def test_color_jitter_is_train_only_and_replayed_across_views():
    yy, xx = np.mgrid[0:480, 0:640]
    image = np.stack([xx * 255 / 640, yy * 255 / 480, (xx + yy) * 255 / 1120], -1).astype(np.uint8)
    video = np.stack([image, image])[None, None]  # (B, T, V, H, W, C)
    jitter = {"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08}

    def frames(training, color_jitter_params):
        random.seed(0)
        step = GrootN17VLMEncodeStep(
            image_target_size=[256, 256],
            shortest_image_edge=256,
            crop_fraction=0.95,
            use_albumentations=True,
            letter_box_transform=True,
            color_jitter_params=color_jitter_params,
            training=training,
        )
        return step._build_sample_images(video, batch_size=1, target_device=None)[0]

    jittered, cropped_only = frames(True, jitter), frames(True, None)
    assert jittered[0].shape == (256, 256, 3)  # letterboxed to square
    np.testing.assert_array_equal(jittered[0], jittered[1])
    assert not np.array_equal(jittered[0], cropped_only[0])
    np.testing.assert_array_equal(frames(False, jitter)[0], frames(False, None)[0])
