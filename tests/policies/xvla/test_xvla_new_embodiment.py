"""XVLA on a new embodiment: 78-D actions / 28-D state from a 20-D checkpoint, domain-id override,
stretch resize and the original two-stage (freeze, warmup, cosine) per-group schedule."""

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.xvla.configuration_xvla import XVLAConfig  # noqa: E402
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_STATE  # noqa: E402

TINY_FLORENCE = {
    "model_type": "florence2",
    "bos_token_id": 0,
    "eos_token_id": 2,
    "pad_token_id": 1,
    "projection_dim": 32,
    "vocab_size": 100,
    "text_config": {
        "vocab_size": 100,
        "d_model": 32,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "encoder_attention_heads": 2,
        "decoder_attention_heads": 2,
        "encoder_ffn_dim": 64,
        "decoder_ffn_dim": 64,
        "max_position_embeddings": 128,
    },
    "vision_config": {
        "model_type": "davit",
        "patch_size": [7, 3, 3, 3],
        "patch_stride": [4, 2, 2, 2],
        "patch_padding": [3, 1, 1, 1],
        "patch_prenorm": [False, True, True, True],
        "dim_embed": [8, 16, 32, 32],
        "num_heads": [1, 1, 1, 1],
        "num_groups": [1, 1, 1, 1],
        "depths": [1, 1, 1, 1],
        "window_size": 2,
        "projection_dim": 32,
        "visual_temporal_embedding": {"type": "COSINE", "max_temporal_embeddings": 4},
        "image_pos_embed": {"type": "learned_abs_2d", "max_pos_embeddings": 8},
        "image_feature_source": ["spatial_avg_pool", "temporal_avg_pool"],
    },
}


def _config(action_dim: int, state_dim: int, **kwargs) -> XVLAConfig:
    return XVLAConfig(
        device="cpu",
        florence_config=TINY_FLORENCE,
        hidden_size=32,
        depth=1,
        num_heads=2,
        num_domains=4,
        len_soft_prompts=2,
        chunk_size=4,
        n_action_steps=4,
        max_len_seq=64,
        action_mode="auto",
        max_action_dim=action_dim,
        max_state_dim=state_dim,
        resize_imgs_with_padding=(64, 64),
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 48, 64)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        **kwargs,
    )


def test_new_embodiment_load_forward_and_schedule(tmp_path):
    torch.manual_seed(0)
    base = XVLAPolicy(_config(20, 20))
    base.save_pretrained(tmp_path)

    cfg = _config(78, 28, domain_id=3, resize_imgs_mode="stretch", official_freeze_steps=10)
    policy = XVLAPolicy.from_pretrained(tmp_path, config=cfg)

    base_sd, new_sd = base.state_dict(), policy.state_dict()
    changed = {k for k in new_sd if new_sd[k].shape != base_sd[k].shape}
    assert changed == {
        "model.transformer.action_encoder.fc.weight",
        "model.transformer.action_decoder.fc.weight",
        "model.transformer.action_decoder.bias.weight",
    }
    for k in new_sd.keys() - changed:
        torch.testing.assert_close(new_sd[k], base_sd[k], msg=k)

    batch = {
        OBS_LANGUAGE_TOKENS: torch.randint(3, 100, (2, 5)),
        OBS_STATE: torch.randn(2, 28),
        "observation.images.cam": torch.rand(2, 3, 48, 64),
        ACTION: torch.randn(2, 4, 78),
        "domain_id": torch.zeros(2, dtype=torch.long),
    }
    assert policy._get_domain_id(batch, 2, torch.device("cpu")).tolist() == [3, 3]
    policy.train()
    loss, _ = policy.forward(batch)
    assert torch.isfinite(loss)
    chunk = policy.predict_action_chunk(batch)
    assert chunk.shape == (2, 4, 78) and torch.isfinite(chunk).all()

    optimizer = cfg.get_optimizer_preset().build(policy.get_optim_params())
    names = [g["name"] for g in optimizer.param_groups]
    assert names == ["vlm", "transformer_core", "soft_prompts", "action_heads"]
    scheduler = cfg.get_scheduler_preset().build(optimizer, num_training_steps=1000)
    peak = {g["name"]: g["initial_lr"] for g in optimizer.param_groups}
    assert peak["vlm"] == pytest.approx(1e-5) and peak["action_heads"] == pytest.approx(1e-4)

    def lrs():
        return {g["name"]: g["lr"] for g in optimizer.param_groups}

    assert lrs()["vlm"] == 0 and lrs()["transformer_core"] == 0  # freeze
    assert lrs()["action_heads"] == pytest.approx(1e-4) and lrs()["soft_prompts"] == pytest.approx(1e-5)
    for _ in range(10 + 500):  # halfway through the 1000-step warmup
        scheduler.step()
    assert 0 < lrs()["transformer_core"] < 1e-4
    for _ in range(2000):
        scheduler.step()
    assert lrs()["transformer_core"] == pytest.approx(1e-4 * cfg.scheduler_decay_lr / cfg.optimizer_lr)
