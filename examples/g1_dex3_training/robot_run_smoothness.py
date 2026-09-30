"""Smoothness of real-robot runs from sonic_policy_streamer.py --log files, over the 'episode' phase only.

Per run:
  seam / step  - mean jump at a chunk switch vs the normal tick step, tokens (0:64) and hands (64:78), in units of
                 the dataset's per-dimension action std (logged tokens are after the slew limit; hands are not limited)
  slewed %     - ticks whose token step hit --max-token-step
  arm speed    - p95 over ticks of the fastest measured arm joint (rad/s)
  arm jerk     - p95 over ticks of the fastest joint's 2-tick windowed third difference (rad/s^3)
  1-2 Hz share - share of measured arm-position power in 1-2 Hz, of the power >= 0.2 Hz
The definitions reproduce the numbers of runs 15-17 in docs/research/2026-09-29-sonic-real-robot-first-runs.md.
Usage: python robot_run_smoothness.py DATASET_ROOT RUN.npz [RUN.npz ...]
"""

import json
import sys
from pathlib import Path

import numpy as np

DT = 0.02  # streamer tick


def windowed_diff(x: np.ndarray, k: int = 2) -> np.ndarray:
    return (x[k:] - x[:-k]) / (k * DT)


def main():
    root, paths = sys.argv[1], sys.argv[2:]
    with open(f"{root}/meta/stats.json") as f:
        std = np.asarray(json.load(f)["action"]["std"], np.float64).clip(1e-6)
    print(
        f"{'run':34s} {'slew':>5s} {'blend':>5s} {'dur s':>5s} {'slewed%':>7s} {'token seam/step':>15s} "
        f"{'hands seam/step':>15s} {'spd p95':>7s} {'jerk p95':>8s} {'1-2Hz%':>6s}"
    )
    for path in paths:
        z = np.load(path, allow_pickle=True)
        args = json.loads(str(z["args"])) if "args" in z.files else {}  # older logs have no args
        ep = z["phase"] == "episode"
        token, hands = z["token"][ep].astype(np.float64), z["hands"][ep].astype(np.float64)
        arm = z["state"][ep, :14].astype(np.float64)

        switch = np.diff(z["chunk_t0"][ep]) != 0  # tick i + 1 starts a new chunk
        token_step = np.abs(np.diff(token, axis=0))
        seams = []
        for step in (token_step / std[:64], np.abs(np.diff(hands, axis=0)) / std[64:78]):
            s = step.mean(1)
            seams.append(f"{s[switch].mean():.3f}/{s[~switch].mean():.3f}")
        limit = args.get("max_token_step", 0)
        slewed = f"{100 * np.mean(token_step.max(1) >= limit - 1e-5):.0f}" if limit else "-"

        speed = np.percentile(np.abs(np.diff(arm, axis=0) / DT).max(1), 95)
        jerk = np.percentile(np.abs(windowed_diff(windowed_diff(windowed_diff(arm)))).max(1), 95)
        power = (np.abs(np.fft.rfft(arm - arm.mean(0), axis=0)) ** 2).sum(1)
        freq = np.fft.rfftfreq(len(arm), DT)
        share = 100 * power[(freq >= 1) & (freq < 2)].sum() / power[freq >= 0.2].sum()

        print(
            f"{Path(path).stem:34s} {str(limit or '?'):>5s} {str(args.get('chunk_blend_s', '?')):>5s} "
            f"{ep.sum() * DT:5.1f} {slewed:>7s} {seams[0]:>15s} {seams[1]:>15s} {speed:7.2f} {jerk:8.0f} {share:6.1f}"
        )


if __name__ == "__main__":
    main()
