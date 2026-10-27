"""Offline check of the official GR00T path (gr00t_official_server.py + the streamer's 46D state) on recorded episodes.

1. State: rebuilds the deploy's g1_debug message (body_q, raw Dex3 hand_q, base_quat) from each recorded frame and
   checks that stream_backends.sonic46_state gives back the recorded observation.state parts and projected_gravity.
2. Open loop: every --replan-s, sends that frame's 46D state and ego_view frame (JPEG, RGB) through the streamer's
   RemotePolicy to a running server, and compares the 40-step chunk with the recorded action.motion_token and
   teleop.*_hand_joints. "hold" is the error of repeating the current recorded action (a no-model baseline).

  python official46_offline_check.py --server tcp://localhost:5562 \
    --dataset /mnt/data/jihun/datasets/G1_WBT_GR00T/handover_bottle_merged_261020 --episodes 27 35
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sonic_policy_streamer import RemotePolicy
from stream_backends import OFFICIAL_HAND_ORDER, sonic46_state

INVERSE_HAND = np.argsort(OFFICIAL_HAND_ORDER)


def g1_debug(state43: np.ndarray, base_quat: np.ndarray) -> dict:
    """The deploy message that produced a recorded 43D observation.state (legs, waist, left arm + hand, right arm +
    hand; hands in robot-model order): body_q in motor order, Dex3 hand_q in raw order."""
    return {
        "body_q": np.r_[state43[0:22], state43[29:36]],
        "left_hand_q": state43[22:29][INVERSE_HAND],
        "right_hand_q": state43[36:43][INVERSE_HAND],
        "base_quat": base_quat,
    }


def frames(path: Path) -> list[np.ndarray]:
    cap, out = cv2.VideoCapture(str(path)), []
    while (ok_img := cap.read())[0]:
        out.append(cv2.cvtColor(ok_img[1], cv2.COLOR_BGR2RGB))  # the head camera streams RGB
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--episodes", type=int, nargs="+", required=True)
    p.add_argument("--replan-s", type=float, default=0.4)
    p.add_argument("--fps", type=float, default=50.0)
    p.add_argument("--out", type=Path, help="write the per-chunk errors as JSON")
    a = p.parse_args()

    task = json.loads((a.dataset / "meta/tasks.jsonl").read_text().splitlines()[0])["task"]
    policy = RemotePolicy(a.server, timeout_s=60.0)
    assert policy.state_dim == 46, f"server state_dim {policy.state_dim}, expected 46"
    key, step = policy.image_keys[0], int(round(a.replan_s * a.fps))
    results = {}
    for ep in a.episodes:
        df = pd.read_parquet(a.dataset / f"data/chunk-000/episode_{ep:06d}.parquet")
        state43 = np.stack(df["observation.state"]).astype(np.float32)
        quat = np.stack(df["observation.root_orientation"])
        recorded = np.c_[state43[:, 0:22], state43[:, 29:36], state43[:, 22:29], state43[:, 36:43],
                         np.stack(df["observation.projected_gravity"])]  # fmt: skip
        built = np.stack([sonic46_state(g1_debug(s, q)) for s, q in zip(state43, quat, strict=True)])
        state_err = np.abs(built - recorded).max(axis=0)
        actions = np.c_[np.stack(df["action.motion_token"]), np.stack(df["teleop.left_hand_joints"]),
                        np.stack(df["teleop.right_hand_joints"])].astype(np.float32)  # fmt: skip
        video = frames(a.dataset / f"videos/chunk-000/observation.images.ego_view/episode_{ep:06d}.mp4")
        assert len(video) == len(df), f"episode {ep}: {len(video)} frames vs {len(df)} rows"
        rows = []
        for t in range(0, len(df) - 40, step):
            chunk = policy.chunk([built[t]], [{key: video[t]}], task)
            truth = actions[t : t + 40]
            rows.append({
                "t": t,
                "token_mae": float(np.abs(chunk[:, :64] - truth[:, :64]).mean()),
                "hand_mae": float(np.abs(chunk[:, 64:] - truth[:, 64:]).mean()),
                "hold_token_mae": float(np.abs(truth[0, :64] - truth[:, :64]).mean()),
                "hold_hand_mae": float(np.abs(truth[0, 64:] - truth[:, 64:]).mean()),
                "seam": float(np.abs(chunk[0, :64] - truth[0, :64]).mean()),
            })  # fmt: skip
        summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "t"}
        results[ep] = {"state_max_err": float(state_err.max()), "gravity_max_err": float(state_err[-3:].max()),
                       "chunks": len(rows), **summary, "rows": rows}  # fmt: skip
        print(f"episode {ep}: state max err {state_err.max():.2e} (gravity {state_err[-3:].max():.2e}), "
              f"{len(rows)} chunks, " + ", ".join(f"{k} {v:.4f}" for k, v in summary.items()), flush=True)  # fmt: skip
    if a.out:
        a.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
