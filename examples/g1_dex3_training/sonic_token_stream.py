"""Look-ahead resampling of policy SONIC-token chunks onto the 50 Hz control ticks of the C++ deploy.

A VLA trained on the 30 Hz sonic78 datasets predicts a chunk of tokens spaced 1/30 s apart. NVIDIA's
g1_deploy_onnx_ref decodes at 50 Hz, so the streamer must supply a token every 20 ms. Linear interpolation
between consecutive chunk tokens (look-ahead) tracked better than holding each token, while blending from
the previous token after a new one arrives (causal, one frame late) tracked worse than holding. See
docs/research/2026-09-23-sonic-roundtrip-audit.md.
"""

from __future__ import annotations

import numpy as np


class ChunkResampler:
    """Token at any time from the latest chunk, linearly interpolated at the chunk's source rate.

    ``set_chunk(chunk, t0)``: ``chunk[i]`` is the token for time ``t0 + i / source_fps``. A new chunk replaces
    the old one from its arrival on. Before ``t0`` the first token is returned; after the last one it is held.
    """

    def __init__(self, source_fps: float = 30.0):
        if source_fps <= 0:
            raise ValueError("source_fps must be positive")
        self.source_fps = float(source_fps)
        self._chunk: np.ndarray | None = None
        self._t0 = 0.0

    def set_chunk(self, chunk, t0: float) -> None:
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or len(chunk) == 0 or not np.isfinite(chunk).all():
            raise ValueError(f"expected a nonempty finite [N, D] token chunk, got shape {chunk.shape}")
        self._chunk, self._t0 = chunk, float(t0)

    @property
    def ready(self) -> bool:
        return self._chunk is not None

    def token_at(self, t: float) -> np.ndarray:
        if self._chunk is None:
            raise RuntimeError("no chunk set")
        x = min(max((t - self._t0) * self.source_fps, 0.0), len(self._chunk) - 1)
        i = int(np.floor(x + 1e-9))
        w = x - i
        if w < 1e-9 or i + 1 >= len(self._chunk):
            return self._chunk[i].copy()
        return ((1 - w) * self._chunk[i] + w * self._chunk[i + 1]).astype(np.float32)


if __name__ == "__main__":
    r = ChunkResampler(30)
    r.set_chunk(np.arange(4, dtype=np.float32)[:, None], t0=1.0)
    assert r.token_at(0.5)[0] == 0 and r.token_at(1.0)[0] == 0
    assert abs(r.token_at(1.02)[0] - 0.6) < 1e-6  # 0.6 of the way from token 0 to 1
    assert r.token_at(9.0)[0] == 3  # held after the chunk ends
    print("ok")
