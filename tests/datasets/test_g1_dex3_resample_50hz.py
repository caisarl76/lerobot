import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
from resample_50hz import n_out, resample_file


def _fixed(values: np.ndarray) -> pa.Array:
    return pa.FixedSizeListArray.from_arrays(pa.array(values.ravel(), pa.float32()), values.shape[1])


def test_episodes_resample_to_50hz_on_the_same_time_span():
    assert [n_out(n, 30, 50) for n in (1, 2, 4, 31)] == [1, 2, 6, 51]  # last frame at or before (n - 1) / 30
    lengths, starts = {3: 31, 5: 10}, {3: 100, 5: 0}  # episode 5 comes first in the dataset
    rows = [(ep, i) for ep, n in lengths.items() for i in range(n)]
    t = np.array([i / 30 for _, i in rows])
    table = pa.table(
        {
            "observation.state": _fixed(np.column_stack([t, 2 * t])),
            "action": _fixed(np.column_stack([t, t * t, -t])),
            "timestamp": pa.array(t, pa.float32()),
            "frame_index": pa.array([i for _, i in rows], pa.int64()),
            "episode_index": pa.array([ep for ep, _ in rows], pa.int64()),
            "index": pa.array(np.arange(len(rows)), pa.int64()),
            "task_index": pa.array([ep for ep, _ in rows], pa.int64()),
        }
    )
    with tempfile.TemporaryDirectory() as d:
        src, dest = Path(d) / "in.parquet", Path(d) / "out" / "out.parquet"
        pq.write_table(table, src)
        _, eps = resample_file((src, dest, 30, 50, starts))
        out = pq.read_table(dest)
    assert out.schema == table.schema
    assert out["episode_index"].to_pylist() == [5] * n_out(10, 30, 50) + [3] * n_out(31, 30, 50)
    for ep, n in lengths.items():
        e = eps[ep]
        t50 = np.arange(n_out(n, 30, 50)) / 50
        assert np.allclose(e["timestamp"], t50) and e["index"][0] == starts[ep]
        # linear in time, so exact everywhere; t^2 only on the shared grid
        assert np.allclose(e["observation.state"], np.column_stack([t50, 2 * t50]))
        assert np.allclose(e["action"][::5, 1], (np.arange(n) / 30)[::3][: len(t50[::5])] ** 2)
