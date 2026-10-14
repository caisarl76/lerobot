"""Joint28 reference for scoring sim runs of the G1 WBT sonic78 models (sonic_stream_eval.py JOINT28_ROOT).

Copies the sonic78 dataset written by prepare_wbt_sonic78.py without its videos and sets `action` to the measured
arm + Dex3 hand joints (`observation.state`, sonic_targets.ACTION_NAMES order), so the scorer compares the reached
palm with FK of the pose the real robot actually reached at the same episode and frame.

Not the commanded joints (`action.wbc`): SONIC's decoder output is a PD setpoint, and in this recording the right
arm stays ~9 deg from it (it holds the bottle). Replaying the stored tokens in sim reaches the measured arm joints
within 1-3 deg, but is 13 cm (p50) from the commanded palms (2026-10-06, episode 27).

Usage: python wbt_joint28_reference.py SONIC78_ROOT OUT
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
from sonic_targets import ACTION_NAMES

sonic, out = Path(sys.argv[1]), Path(sys.argv[2])
shutil.copytree(sonic / "meta", out / "meta")
for f in sorted((sonic / "data").rglob("*.parquet")):
    t = pq.read_table(f)
    t = t.set_column(t.schema.get_field_index("action"), "action", t["observation.state"])
    (out / f.relative_to(sonic)).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, out / f.relative_to(sonic))
info = json.loads((out / "meta/info.json").read_text())
info["features"]["action"] = {"dtype": "float32", "shape": [28], "names": list(ACTION_NAMES)}
(out / "meta/info.json").write_text(json.dumps(info, indent=4))
print(f"{out}: action = measured arm + hand joints (observation.state, 28)")
