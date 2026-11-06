"""Copy a v3.0 sonic78 handover dataset with the training data of the official Isaac-GR00T run (s25 recipe).

The official run trains on the recorded NVIDIA-exporter episodes as they are; our sonic78 copy differs in two ways
(2026-11-06 check, tokens and 46-D state are identical):
  videos  the per-episode H.264 recordings were re-encoded to AV1 in LeRobot's merged files; here every episode found
          in RAW points at a copy of its original H.264 file (videos/<key>/chunk-001/file-<episode>.mp4, from 0 s).
          Other episodes (e.g. held-out ones missing from a filtered RAW) keep their AV1 segments.
  hands   (--hands teleop) action[64:78] = teleop.{left,right}_hand_joints as recorded, the official targets: the
          deploy's pose-message slot order (the streamer sends hands[:7] / hands[7:] in the same fields as the teleop),
          thumb 0-2, index 0-1, middle 0-1 per hand by the exporter's labels. Ours are action.wbc's hands: the WBC's
          hand command, ~2 frames later and not equal to teleop (right thumb_2 and middle differ), in
          sonic_targets.ACTION_NAMES order, whose left hand (thumb, middle, index) swaps index_0 / middle_0 against
          those slots (2026-11-06). --hands wbc keeps them.
Statistics of observation.state and action are recomputed over the episodes found in RAW (the official run's
statistics cover its training episodes only); image statistics are kept.

Usage (lerobot image, CPU only): python official_match_dataset.py --src DS --raw RAW --out OUT --hands teleop|wbc
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from prepare_state46 import stats

VIDEO = "observation.images.egocentric"
TELEOP_HAND = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
RAW_VIDEO = "videos/chunk-000/observation.images.ego_view/episode_{:06d}.mp4"


def teleop_hands(raw_df: pd.DataFrame) -> np.ndarray:
    return np.concatenate(
        [np.stack(raw_df[f"teleop.{s}_hand_joints"].values) for s in ("left", "right")], axis=1
    )


def frame_count(path: Path) -> int:
    with av.open(str(path)) as c:
        return c.streams.video[0].frames


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hands", choices=("teleop", "wbc"), required=True)
    a = p.parse_args()
    assert not a.out.exists(), a.out
    info = json.loads((a.src / "meta/info.json").read_text())
    fps = info["fps"]
    tmp = a.out.with_name(a.out.name + ".incomplete")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(a.src / "meta", tmp / "meta")

    (vdir := tmp / "videos" / VIDEO).mkdir(parents=True)
    src_videos = (a.src / "videos").resolve() / VIDEO / "chunk-000"
    os.symlink(os.path.relpath(src_videos, vdir), vdir / "chunk-000")  # AV1 segments of episodes not in RAW
    (vdir / "chunk-001").mkdir()

    data_files = sorted((a.src / "data").rglob("*.parquet"))
    ep_files = sorted((tmp / "meta/episodes").rglob("*.parquet"))
    used, st_rows, act_rows, upd = [], [], [], {}
    for f in data_files:
        table = pq.read_table(f)
        df = table.to_pandas()
        action = np.stack(df["action"].values).astype(np.float32)
        for ep in np.unique(df["episode_index"]):
            raw_parquet = a.raw / f"data/chunk-000/episode_{ep:06d}.parquet"
            if not raw_parquet.exists():
                continue
            m = (df["episode_index"] == ep).to_numpy()
            raw = pd.read_parquet(raw_parquet)
            assert len(raw) == m.sum(), (ep, len(raw), m.sum())
            # same recording: SONIC tokens are copied unchanged by our converter
            assert np.array_equal(
                action[m, :64], np.stack(raw["action.motion_token"].values).astype(np.float32)
            ), ep
            if a.hands == "teleop":
                action[m, 64:] = teleop_hands(raw)
            dst = vdir / f"chunk-001/file-{ep:03d}.mp4"
            shutil.copy2(a.raw / RAW_VIDEO.format(ep), dst)
            assert frame_count(dst) == m.sum(), (ep, "video frames != episode length")
            v = f"videos/{VIDEO}/"
            upd[int(ep)] = {v + "chunk_index": 1, v + "file_index": int(ep), v + "from_timestamp": 0.0}
            upd[int(ep)][v + "to_timestamp"] = m.sum() / fps
            if a.hands == "teleop":
                upd[int(ep)].update({f"stats/action/{k}": x for k, x in stats(action[m]).items()})
            used.append(int(ep))
            st_rows.append(np.stack(df["observation.state"].values)[m])
            act_rows.append(action[m])
        col = pa.array(list(action), type=table.schema.field("action").type)
        out = tmp / f.relative_to(a.src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.set_column(table.schema.get_field_index("action"), "action", col), out)

    for f in ep_files:  # video mapping (+ per-episode action stats) of the episodes found in RAW
        t = pq.read_table(f)
        ids = t["episode_index"].to_pylist()
        for name in {k for u in upd.values() for k in u} & set(t.column_names):
            vals = [upd.get(e, {}).get(name, old) for e, old in zip(ids, t[name].to_pylist(), strict=True)]
            t = t.set_column(t.schema.get_field_index(name), name, pa.array(vals, t.schema.field(name).type))
        pq.write_table(t, f)

    st = json.loads((tmp / "meta/stats.json").read_text())
    st["observation.state"] = stats(np.concatenate(st_rows))
    st["action"] = stats(np.concatenate(act_rows))
    (tmp / "meta/stats.json").write_text(json.dumps(st, indent=2))
    if a.hands == "teleop":
        info["features"]["action"]["names"][64:] = [
            f"{s}_hand_{j}_joint" for s in ("left", "right") for j in TELEOP_HAND
        ]
    info["features"][VIDEO]["info"]["video.codec"] = "h264 (chunk-001, original recordings); av1 (chunk-000)"
    (tmp / "meta/info.json").write_text(json.dumps(info, indent=2))
    (tmp / "meta/official_match_provenance.json").write_text(
        json.dumps(
            {
                "src": str(a.src),
                "raw": str(a.raw),
                "hands": a.hands,
                "h264_episodes": used,
                "stats_episodes": used,
            },
            indent=2,
        )
    )
    tmp.rename(a.out)
    print(f"{a.out}: {len(used)} episodes on H.264, hands {a.hands}, stats over {len(used)} episodes")


if __name__ == "__main__":
    main()
