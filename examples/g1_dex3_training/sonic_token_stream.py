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
    the old one from its arrival on; with ``blend_s > 0`` and the arrival time ``now``, the output instead
    cross-fades linearly from the old chunk to the new one over ``blend_s`` seconds (both evaluated at the same
    time), so consecutive chunks that disagree do not make the target jump. Before ``t0`` the first token is
    returned; after the last one it is held.

    With ``anchor_s > 0`` (``--chunk-anchor-s``; overrides ``blend_s``) the new chunk is instead offset to start
    at the current output and keeps its own velocity; the offset decays as ``exp(-(t - now) / anchor_s)``, so the
    output returns to the new chunk's absolute targets within a few ``anchor_s``. A policy that restarts its plan
    from an unchanged observation at every replan then still makes progress; with the cross-fade each switch pulls
    the command back toward the restart point, which kept the G1 arm inside its dead zone (stall "treadmill",
    2026-10-21). ``anchor_dims`` limits the offset to the first dims (the 64 SONIC token dims): the rest (absolute
    Dex3 hand targets, which have no slew limit downstream) keep the plain cross-fade, so they stay inside the
    chunks' own range. ``current`` is the command actually sent (after a downstream slew limit); the offset is taken
    from it rather than from this resampler's output.
    """

    def __init__(self, source_fps: float = 30.0):
        if source_fps <= 0:
            raise ValueError("source_fps must be positive")
        self.source_fps = float(source_fps)
        self._chunk: np.ndarray | None = None
        self._t0 = 0.0
        self._prev: tuple[np.ndarray, float] | None = None  # outgoing chunk during a cross-fade
        self._blend = (0.0, 0.0)  # (start, duration) of the cross-fade
        # (offset, start, time constant, anchored dims)
        self._anchor: tuple[np.ndarray, float, float, int] | None = None

    def set_chunk(
        self,
        chunk,
        t0: float,
        blend_s: float = 0.0,
        now: float | None = None,
        anchor_s: float = 0.0,
        anchor_dims: int | None = None,
        current=None,
    ) -> None:
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or len(chunk) == 0 or not np.isfinite(chunk).all():
            raise ValueError(f"expected a nonempty finite [N, D] token chunk, got shape {chunk.shape}")
        if anchor_s > 0 and now is not None and self._chunk is not None:
            # continue from the command being sent now (itself possibly still offset) with the new chunk's motion
            n = chunk.shape[1] if anchor_dims is None else min(int(anchor_dims), chunk.shape[1])
            ref = self.token_at(now)
            if current is not None:
                cur = np.asarray(current, np.float32)[:n]
                ref[: len(cur)] = cur
            offset = np.zeros(chunk.shape[1], np.float32)
            offset[:n] = ref[:n] - self._at(chunk, float(t0), now)[:n]
            if blend_s > 0 and n < chunk.shape[1]:  # the dims that are not anchored keep the cross-fade
                self._prev = (
                    (self.token_at(now)[None], now) if self._prev is not None else (self._chunk, self._t0)
                )
                self._blend = (float(now), float(blend_s))
            else:
                self._prev = None
            self._chunk, self._t0 = chunk, float(t0)
            self._anchor = (offset, float(now), float(anchor_s), n)
            return
        self._anchor = None
        if blend_s > 0 and now is not None and self._chunk is not None:
            # fade from what is being output now (itself possibly mid-fade) to the new chunk
            self._prev = (
                (self.token_at(now)[None], now) if self._prev is not None else (self._chunk, self._t0)
            )
            self._blend = (float(now), float(blend_s))
        else:
            self._prev = None
        self._chunk, self._t0 = chunk, float(t0)

    @property
    def ready(self) -> bool:
        return self._chunk is not None

    def _at(self, chunk: np.ndarray, t0: float, t: float) -> np.ndarray:
        x = min(max((t - t0) * self.source_fps, 0.0), len(chunk) - 1)
        i = int(np.floor(x + 1e-9))
        w = x - i
        if w < 1e-9 or i + 1 >= len(chunk):
            return chunk[i].copy()
        return ((1 - w) * chunk[i] + w * chunk[i + 1]).astype(np.float32)

    def token_at(self, t: float) -> np.ndarray:
        if self._chunk is None:
            raise RuntimeError("no chunk set")
        new = self._at(self._chunk, self._t0, t)
        out = new
        if self._prev is not None:
            start, duration = self._blend
            w = (t - start) / duration
            if w < 1:
                old = self._at(*self._prev, t)
                w = max(w, 0.0)
                out = ((1 - w) * old + w * new).astype(np.float32)
            else:
                self._prev = None
        if self._anchor is not None:
            offset, start, tau, n = self._anchor
            out = out.copy()
            out[:n] = new[:n] + offset[:n] * np.float32(np.exp(-max(t - start, 0.0) / tau))
        return out


if __name__ == "__main__":
    r = ChunkResampler(30)
    r.set_chunk(np.arange(4, dtype=np.float32)[:, None], t0=1.0)
    assert r.token_at(0.5)[0] == 0 and r.token_at(1.0)[0] == 0
    assert abs(r.token_at(1.02)[0] - 0.6) < 1e-6  # 0.6 of the way from token 0 to 1
    assert r.token_at(9.0)[0] == 3  # held after the chunk ends
    print("ok")
