"""Per-phase report of a streamed sim run (sonic_official_sim_host.py + sonic_policy_streamer.py): for each streamer
phase (planner token, handoff blend, table startup, blend in, episode, ..., handback blend) its duration, the closest
arm-to-table gap, table contacts, the fewest foot contacts and the largest pelvis tilt.
Usage: python sonic_stream_phases.py RUN_DIR   (RUN_DIR holds sim/sim_state.npz and streamer.npz)
"""

import sys

import numpy as np

run = sys.argv[1]
s = np.load(f"{run}/sim/sim_state.npz")
z = np.load(f"{run}/streamer.npz", allow_pickle=True)
ph, wall = z["phase"], z["wall"]
names = [p for i, p in enumerate(ph) if i == 0 or ph[i - 1] != p]
print("phase                  dur(s)  min table gap (cm)  table hits  min feet  max tilt(deg)")
for p in names:
    w = wall[ph == p]
    m = (s["wall"] >= w[0]) & (s["wall"] <= w[-1] + 0.02)
    if not m.any():
        continue
    q = s["pelvis"][m][:, 3:7]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1)))
    print(
        f"{p:22s} {w[-1] - w[0]:6.1f}  {100 * np.nanmin(s['table_clear'][m]):8.1f}           {int(s['table_hits'][m].max()):4d}      {int(s['floor_contacts'][m].min()):4d}     {tilt.max():6.2f}"
    )
