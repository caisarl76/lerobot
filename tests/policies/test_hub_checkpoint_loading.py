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
"""Psi0 / XR-1 checkpoints from the Hub: config and processor come from the same repo and revision as the weights."""

from types import SimpleNamespace

import pytest

import lerobot.policies.utils as policy_utils
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.psi0.modeling_psi0 import VLM_PROCESSOR_DIRNAME, Psi0Policy
from lerobot.policies.xiaomi_robotics.modeling_xiaomi_robotics import XiaomiRoboticsPolicy


@pytest.mark.parametrize("policy_cls", [Psi0Policy, XiaomiRoboticsPolicy])
def test_hub_load_uses_one_revision_and_the_saved_processor(policy_cls, tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    (snapshot / VLM_PROCESSOR_DIRNAME).mkdir(parents=True)
    calls = {}

    def fake_snapshot_download(repo_id, allow_patterns, **kwargs):
        calls["snapshot"] = (repo_id, allow_patterns, kwargs)
        return str(snapshot)

    def fake_config_load(path, **kwargs):
        calls["config"] = (path, kwargs)
        return SimpleNamespace(load_base_weights=True, processor_path="/psi-weights/training-machine-only")

    def fake_base_load(cls, path, *, config, **kwargs):
        return config

    monkeypatch.setattr(policy_utils, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(policy_cls.config_class, "from_pretrained", fake_config_load)
    monkeypatch.setattr(PreTrainedPolicy, "from_pretrained", classmethod(fake_base_load))

    config = policy_cls.from_pretrained("user/repo", revision="abc123", token="t", strict=True)

    assert calls["config"] == ("user/repo", {"revision": "abc123", "token": "t"})
    assert calls["snapshot"] == (
        "user/repo",
        [f"{VLM_PROCESSOR_DIRNAME}/*"],
        {"revision": "abc123", "token": "t"},
    )
    assert config.processor_path == str(snapshot / VLM_PROCESSOR_DIRNAME)
    assert config.load_base_weights is False


def test_local_checkpoint_without_processor_dir_keeps_config_path(tmp_path):
    assert policy_utils.checkpoint_subdir(tmp_path, VLM_PROCESSOR_DIRNAME) is None
    (tmp_path / VLM_PROCESSOR_DIRNAME).mkdir()
    assert policy_utils.checkpoint_subdir(tmp_path, VLM_PROCESSOR_DIRNAME) == tmp_path / VLM_PROCESSOR_DIRNAME
