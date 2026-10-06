"""Joint28 reference for scoring sim runs of the G1 WBT sonic78 models (sonic_stream_eval.py JOINT28_ROOT).

Copies the sonic78 dataset written by prepare_wbt_sonic78.py without its videos and replaces `action` with the
recording's commanded arm + Dex3 hand joints (action.wbc, by name in sonic_targets.ACTION_NAMES order), so the scorer
can compare the reached palm with FK of the original joint action at the same episode and frame.

Usage: python wbt_joint28_reference.py SRC_V21 SONIC78_ROOT OUT
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sonic_targets import ACTION_NAMES

src, sonic, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
info_src = json.loads((src / "meta/info.json").read_text())
idx = [info_src["features"]["action.wbc"]["names"].index(n) for n in ACTION_NAMES]
shutil.copytree(sonic / "meta", out / "meta")
for f in sorted((sonic / "data").rglob("*.parquet")):
    t = pq.read_table(f)
    ep, fr = t["episode_index"].to_numpy(), t["frame_index"].to_numpy()
    joints = np.zeros((len(ep), 28), np.float32)
    for e in np.unique(ep):
        df = pd.read_parquet(src / info_src["data_path"].format(episode_chunk=e // 1000, episode_index=e))
        wbc = np.stack(df["action.wbc"])[:, idx]
        rows = ep == e
        joints[rows] = wbc[fr[rows]]
    col = pa.FixedSizeListArray.from_arrays(pa.array(joints.ravel()), 28)
    t = t.set_column(t.schema.get_field_index("action"), "action", col)
    (out / f.relative_to(sonic)).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, out / f.relative_to(sonic))
info = json.loads((out / "meta/info.json").read_text())
info["features"]["action"] = {"dtype": "float32", "shape": [28], "names": list(ACTION_NAMES)}
(out / "meta/info.json").write_text(json.dumps(info, indent=4))
print(f"{out}: action = action.wbc arm + hand joints (28)")
