import json
import sys
import tempfile
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
import official_match_dataset

V = "videos/observation.images.egocentric/"


def _fixed(values: np.ndarray) -> pa.Array:
    return pa.FixedSizeListArray.from_arrays(pa.array(values.ravel(), pa.float32()), values.shape[1])


def _video(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=50)
        s.width, s.height, s.pix_fmt = 32, 32, "yuv420p"
        for _ in range(n):
            c.mux(s.encode(av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), np.uint8), format="rgb24")))
        c.mux(s.encode())


def test_episodes_in_raw_get_teleop_hands_and_their_original_video():
    rng = np.random.default_rng(0)
    n = {0: 3, 1: 2}  # episode 1 is missing from RAW (held out): it keeps everything
    action, state = rng.normal(size=(5, 78)).astype(np.float32), rng.normal(size=(5, 46)).astype(np.float32)
    teleop = rng.normal(size=(3, 14)).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        src, raw, out = Path(d) / "src", Path(d) / "raw", Path(d) / "out"
        (src / "data/chunk-000").mkdir(parents=True)
        (src / "meta/episodes/chunk-000").mkdir(parents=True)
        _video(src / V / "chunk-000/file-000.mp4", 5)
        pq.write_table(
            pa.table(
                {
                    "observation.state": _fixed(state),
                    "action": _fixed(action),
                    "episode_index": pa.array([0, 0, 0, 1, 1], pa.int64()),
                }
            ),
            src / "data/chunk-000/file-000.parquet",
        )
        stat = {k: [pa.array([[0.0] * 78] * 2, pa.list_(pa.float64()))] for k in ("min", "max")}
        pq.write_table(
            pa.table(
                {
                    "episode_index": pa.array([0, 1], pa.int64()),
                    V + "chunk_index": pa.array([0, 0], pa.int64()),
                    V + "file_index": pa.array([0, 0], pa.int64()),
                    V + "from_timestamp": pa.array([0.0, 0.06]),
                    V + "to_timestamp": pa.array([0.06, 0.1]),
                    **{f"stats/action/{k}": v[0] for k, v in stat.items()},
                }
            ),
            src / "meta/episodes/chunk-000/file-000.parquet",
        )
        features = {
            "action": {"names": [f"a{i}" for i in range(78)]},
            "observation.images.egocentric": {"info": {}},
        }
        (src / "meta/info.json").write_text(json.dumps({"fps": 50, "features": features}))
        (src / "meta/stats.json").write_text(json.dumps({"action": {}, "observation.state": {}}))
        (raw / "data/chunk-000").mkdir(parents=True)
        pq.write_table(
            pa.table({
                "action.motion_token": _fixed(action[:3, :64]),
                "teleop.left_hand_joints": _fixed(teleop[:, :7]),
                "teleop.right_hand_joints": _fixed(teleop[:, 7:]),
            }),
            raw / "data/chunk-000/episode_000000.parquet",
        )  # fmt: skip
        _video(raw / "videos/chunk-000/observation.images.ego_view/episode_000000.mp4", 3)
        sys.argv = ["x", "--src", str(src), "--raw", str(raw), "--out", str(out), "--hands", "teleop"]
        official_match_dataset.main()

        a = np.stack(
            pq.read_table(out / "data/chunk-000/file-000.parquet")["action"].to_numpy(zero_copy_only=False)
        )
        eps = pq.read_table(out / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
        st = json.loads((out / "meta/stats.json").read_text())
        assert (out / V / "chunk-001/file-000.mp4").exists() and (out / V / "chunk-000/file-000.mp4").exists()
    assert np.array_equal(a[:3, 64:], teleop) and np.array_equal(a[:3, :64], action[:3, :64])
    assert np.array_equal(a[3:], action[3:])  # held-out episode untouched
    assert [eps[0][V + k] for k in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")] == [
        1,
        0,
        0,
        0.06,
    ]
    assert [eps[1][V + k] for k in ("chunk_index", "from_timestamp")] == [0, 0.06]
    assert (
        np.allclose(eps[0]["stats/action/max"][64:], teleop.max(0))
        and eps[1]["stats/action/max"] == [0.0] * 78
    )
    assert st["action"]["count"] == [n[0]] and np.allclose(st["observation.state"]["mean"], state[:3].mean(0))
