import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import msgpack

from lerobot.configs.types import FeatureType, PolicyFeature

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
from sonic_policy_server import info_reply
from sonic_policy_streamer import RemotePolicy, policy_state_dim

IMAGE = {"observation.images.egocentric": PolicyFeature(FeatureType.VISUAL, (3, 480, 640))}


def _state(n: int) -> dict:
    return {"observation.state": PolicyFeature(FeatureType.STATE, (n,)), **IMAGE}


def test_state_width_comes_from_the_policy_config():
    assert policy_state_dim(_state(46)) == 46
    assert policy_state_dim(_state(28)) == 28
    assert policy_state_dim(IMAGE) == 28  # no state input (LaWAM): the 28D state, ignored by the policy


def test_server_info_carries_the_state_width_to_the_streamer():
    policy = SimpleNamespace(
        image_keys=list(IMAGE), n_obs=1, shapes={k: v.shape for k, v in IMAGE.items()}, state_dim=46
    )
    wire = msgpack.unpackb(msgpack.packb(info_reply(policy, "/models/x"), use_bin_type=True), raw=False)
    with (
        mock.patch.object(RemotePolicy, "_ask", return_value=wire),
        mock.patch.object(RemotePolicy, "_close"),
    ):
        remote = RemotePolicy("tcp://unused:1", 1.0)  # no socket: _ask answers with the server's reply
    assert remote.state_dim == 46 and remote.model == "/models/x"
    assert remote.shapes == {"observation.images.egocentric": (3, 480, 640)} and remote.n_obs == 1
