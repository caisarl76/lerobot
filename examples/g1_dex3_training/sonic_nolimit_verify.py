"""Verify a published no-limit sonic78 dataset against its source and the replay-tested tokens."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/code")
from sonic_roundtrip_audit import v3_episode, v3_meta

src, conv, variants = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
name = sys.argv[4]  # unitree | humanoid_everyday: prefix of the replay-tested variant files
info_s, info_c = (json.loads((p / "meta/info.json").read_text()) for p in (src, conv))
prov = json.loads((conv / "meta/provenance.json").read_text())
reps = prov["episode_reports"]
out = {
    "published": conv.exists() and not Path(str(conv) + ".incomplete").exists(),
    "episodes_frames_match_source": [info_c["total_episodes"], info_c["total_frames"]]
    == [info_s["total_episodes"], info_s["total_frames"]],
    "action_shape": info_c["features"]["action"]["shape"],
    "arm_speed_limits_recorded": sorted({str(r["arm_speed_limit_rad_s"]) for r in reps}),
    "hand_speed_limits_recorded": sorted({str(r["hand_speed_limit_rad_s"]) for r in reps}),
    "rate_limited_values_total": int(sum(r["rate_limited_values"] for r in reps)),
    "clipped_values_total": int(sum(r["clipped_values"] for r in reps)),
    "encoder_sha256": prov["encoder_sha256"],
}
rows, info = v3_meta(src)
by_ep = {r["episode_index"]: r for r in rows}
checks = []
for f in sorted(variants.glob(f"nolimit_hold__{name}__*.npz")):
    ep = int(f.stem.rsplit("__", 1)[1])
    stored = v3_episode(conv, by_ep[ep], info, ["frame_index", "action"])["action"]
    orig = v3_episode(src, by_ep[ep], info, ["frame_index", "action"])["action"]
    tested = np.load(f)
    checks.append(
        {
            "episode": ep,
            "tokens_equal_replay_tested": bool(np.array_equal(stored[:, :64], tested["tokens"])),
            "hands_equal_replay_tested": bool(np.array_equal(stored[:, 64:], tested["hands"])),
            "hands_vs_source_maxabs": round(float(np.abs(stored[:, 64:] - orig[:, 14:]).max()), 4),
        }
    )
out["replay_tested_episodes"] = checks
print(json.dumps(out, indent=1))
