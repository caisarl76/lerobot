import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature

msgpack = pytest.importorskip("msgpack")
pytest.importorskip("zmq")  # the server imports it at module level
sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
from sonic_policy_server import info_reply  # noqa: E402
from sonic_policy_streamer import ChunkPolicy, RemotePolicy  # noqa: E402

IMAGE = {"observation.images.egocentric": PolicyFeature(FeatureType.VISUAL, (3, 480, 640))}


def _chunk_policy(features: dict) -> ChunkPolicy:
    """ChunkPolicy on a config with these input features; model and processor loading mocked."""
    cfg = SimpleNamespace(type="groot", input_features=features, n_obs_steps=1, device="cuda")
    with (
        mock.patch.object(PreTrainedConfig, "from_pretrained", return_value=cfg),
        mock.patch("lerobot.policies.factory.get_policy_class"),
        mock.patch("lerobot.policies.factory.make_pre_post_processors", return_value=(None, None)),
    ):
        return ChunkPolicy("/models/x", "cpu")


# None: no state input (LaWAM) -> the 28D state, which the policy ignores
@pytest.mark.parametrize("state_dim", [46, 28, None])
def test_the_policy_configs_state_width_reaches_the_streamer(state_dim):
    state = {"observation.state": PolicyFeature(FeatureType.STATE, (state_dim,))} if state_dim else {}
    policy = _chunk_policy({**state, **IMAGE})
    wire = msgpack.unpackb(msgpack.packb(info_reply(policy, "/models/x"), use_bin_type=True), raw=False)
    with (
        mock.patch.object(RemotePolicy, "_ask", return_value=wire),
        mock.patch.object(RemotePolicy, "_close"),
    ):
        remote = RemotePolicy("tcp://unused:1", 1.0)  # no socket: _ask answers with the server's reply
    assert remote.state_dim == (state_dim or 28)
    assert remote.model == "/models/x" and remote.n_obs == 1
    assert remote.shapes == {"observation.images.egocentric": (3, 480, 640)}
