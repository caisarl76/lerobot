"""Build replay inputs for the speed-limit and 30/50 Hz token-rate comparison (one npz per variant x episode).

limited_hold: stored sonic78 tokens (1 rad/s arm, 2 rad/s hand limits), held at 50 Hz.
nolimit_hold / nolimit_interp: no-limit converter tokens at 30 Hz, held or linearly interpolated at 50 Hz.
nolimit_50hz: original actions linearly interpolated to 50 Hz, then encoded at 50 Hz (a 50 Hz dataset).
All no-limit variants share the 30 Hz no-limit hands (held), so only the token feed differs.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/code")
from sonic_targets import SonicEncoder, build_encoder_inputs, load_joint_limits

inp, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
lim = load_joint_limits(Path("/gear_sonic_deploy/g1/g1_29dof_with_hand.xml"))
enc = SonicEncoder(Path("/sonic-model/model_encoder.onnx"), Path("/sonic-model/observation_config.yaml"))
for stem in sys.argv[3:]:
    e = dict(np.load(inp / f"{stem}.npz", allow_pickle=True))
    a, fps = e["original"].astype(np.float64), float(e["fps"])
    x, hands, _ = build_encoder_inputs(a, lim, arm_speed_limit=None, hand_speed_limit=None)
    tok30 = enc.encode(x)
    t50 = np.arange(int(np.floor((len(a) - 1) / fps * 50)) + 1) / 50
    a50 = np.column_stack([np.interp(t50, np.arange(len(a)) / fps, a[:, i]) for i in range(28)])
    x50, _, _ = build_encoder_inputs(a50, lim, arm_speed_limit=None, hand_speed_limit=None, fps=50)
    tok50 = enc.encode(x50)
    common = {"original": e["original"], "fps": e["fps"], "task": e["task"]}
    np.savez(
        out / f"limited_hold__{stem}.npz",
        tokens=e["tokens"],
        hands=e["hands"],
        token_fps=fps,
        interp=False,
        **common,
    )
    np.savez(
        out / f"nolimit_hold__{stem}.npz", tokens=tok30, hands=hands, token_fps=fps, interp=False, **common
    )
    np.savez(
        out / f"nolimit_interp__{stem}.npz", tokens=tok30, hands=hands, token_fps=fps, interp=True, **common
    )
    np.savez(
        out / f"nolimit_50hz__{stem}.npz", tokens=tok50, hands=hands, token_fps=50.0, interp=False, **common
    )
    print(
        stem,
        len(a),
        "tokens30",
        tok30.shape,
        "tokens50",
        tok50.shape,
        "off-grid50",
        bool(np.abs(tok50 * 16 - np.rint(tok50 * 16)).max() > 1e-5),
    )
