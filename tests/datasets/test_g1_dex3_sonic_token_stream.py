import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
from sonic_token_stream import ChunkResampler


class ChunkResamplerTests(unittest.TestCase):
    def test_interpolates_at_50hz_ticks_between_30hz_tokens(self):
        tokens = np.stack([np.full(64, i / 16, np.float32) for i in range(10)])
        r = ChunkResampler(30)
        r.set_chunk(tokens, t0=0.0)
        for j in range(15):
            x = j * 30 / 50
            np.testing.assert_allclose(r.token_at(j / 50), np.full(64, x / 16), atol=1e-6)

    def test_holds_first_before_start_and_last_after_end(self):
        tokens = np.arange(6, dtype=np.float32).reshape(3, 2)
        r = ChunkResampler(30)
        r.set_chunk(tokens, t0=2.0)
        np.testing.assert_array_equal(r.token_at(1.0), tokens[0])
        np.testing.assert_array_equal(r.token_at(10.0), tokens[-1])

    def test_new_chunk_replaces_old_from_its_time_base(self):
        r = ChunkResampler(30)
        r.set_chunk(np.zeros((40, 64)), t0=0.0)
        new = np.ones((40, 64), np.float32)
        r.set_chunk(new, t0=0.3)
        np.testing.assert_array_equal(r.token_at(0.4), new[3])

    def test_blend_cross_fades_from_old_to_new_chunk(self):
        r = ChunkResampler(30)
        r.set_chunk(np.zeros((40, 4)), t0=0.0)
        r.set_chunk(np.ones((40, 4)), t0=0.3, blend_s=0.2, now=0.5)
        np.testing.assert_allclose(r.token_at(0.5), np.zeros(4), atol=1e-6)  # starts at the old chunk
        np.testing.assert_allclose(r.token_at(0.6), np.full(4, 0.5), atol=1e-6)  # halfway
        np.testing.assert_allclose(r.token_at(0.7), np.ones(4), atol=1e-6)  # new chunk from the end on
        np.testing.assert_allclose(r.token_at(1.0), np.ones(4), atol=1e-6)

    def test_blend_restarted_mid_fade_continues_from_current_output(self):
        r = ChunkResampler(30)
        r.set_chunk(np.zeros((40, 4)), t0=0.0)
        r.set_chunk(np.ones((40, 4)), t0=0.0, blend_s=0.4, now=0.0)
        r.set_chunk(np.full((40, 4), 2.0), t0=0.2, blend_s=0.4, now=0.2)  # arrives when output is 0.5
        np.testing.assert_allclose(r.token_at(0.2), np.full(4, 0.5), atol=1e-6)  # no jump
        np.testing.assert_allclose(r.token_at(0.4), np.full(4, 1.25), atol=1e-6)
        np.testing.assert_allclose(r.token_at(0.6), np.full(4, 2.0), atol=1e-6)

    def test_zero_blend_keeps_the_instant_switch(self):
        r = ChunkResampler(30)
        r.set_chunk(np.zeros((40, 4)), t0=0.0)
        r.set_chunk(np.ones((40, 4)), t0=0.3, blend_s=0.0, now=0.5)
        np.testing.assert_array_equal(r.token_at(0.5), np.ones(4))

    def test_exact_source_times_return_stored_tokens_bitwise(self):
        tokens = np.random.default_rng(0).integers(-16, 16, (30, 64)).astype(np.float32) / 16
        r = ChunkResampler(30)
        r.set_chunk(tokens, t0=0.0)
        for i in range(0, 30, 3):  # every 3rd 30 Hz frame coincides with a 50 Hz tick
            np.testing.assert_array_equal(r.token_at(i / 30), tokens[i])

    def test_rejects_bad_input(self):
        r = ChunkResampler(30)
        with self.assertRaises(RuntimeError):
            r.token_at(0.0)
        for bad in (np.zeros((0, 64)), np.zeros(64), np.full((2, 64), np.nan)):
            with self.assertRaises(ValueError):
                r.set_chunk(bad, 0.0)
        with self.assertRaises(ValueError):
            ChunkResampler(0)


if __name__ == "__main__":
    unittest.main()


class ChunkAnchorTests(unittest.TestCase):
    """--chunk-anchor-s: a new chunk is offset to start at the current output, the offset decays with time
    constant anchor_s, so a policy that keeps restarting its plan from an unchanged observation still makes
    progress (the stall "treadmill" of 2026-10-21) instead of being pulled back at every switch."""

    def test_switch_is_continuous_and_keeps_the_new_velocity(self):
        r = ChunkResampler(50)
        r.set_chunk(np.linspace(0, 0.8, 41)[:, None].repeat(4, 1), t0=0.0)  # slope 1 / s
        before = r.token_at(0.5)
        r.set_chunk(np.linspace(0, 0.8, 41)[:, None].repeat(4, 1), t0=0.4, now=0.5, anchor_s=1.0)
        np.testing.assert_allclose(r.token_at(0.5), before, atol=1e-6)  # no jump at the switch
        # 0.1 s later: new chunk advanced 0.1, offset (0.4) decayed by exp(-0.1)
        np.testing.assert_allclose(r.token_at(0.6), np.full(4, 0.2 + 0.4 * np.exp(-0.1)), atol=1e-5)

    def test_offset_decays_back_to_the_new_chunk(self):
        r = ChunkResampler(50)
        r.set_chunk(np.ones((40, 2), np.float32), t0=0.0)
        r.set_chunk(np.zeros((400, 2), np.float32), t0=0.0, now=0.0, anchor_s=0.5)
        np.testing.assert_allclose(r.token_at(0.0), np.ones(2), atol=1e-6)
        np.testing.assert_allclose(r.token_at(1.5), np.full(2, np.exp(-3)), atol=1e-5)  # 3 time constants

    def test_restarting_plans_still_make_progress(self):
        """Every chunk restarts at 0 with slope 1/s (observation never changes), replanned every 0.4 s and
        arriving 0.2 s after its t0. The cross-fade never gets past one chunk's own travel (0.8); anchoring
        builds a lead (bounded by the decay: steady offset 0.4 / (1 - exp(-0.4)) = 1.21 at anchor_s 1)."""

        def run(**kw):
            r = ChunkResampler(50)
            chunk = np.linspace(0, 0.8, 41)[:, None].astype(np.float32)
            r.set_chunk(chunk, t0=0.0)
            for k in range(1, 11):
                t0, now = 0.4 * k, 0.4 * k + 0.2
                r.set_chunk(chunk, t0=t0, now=now, **kw)
            return float(r.token_at(4.4)[0])

        blended, anchored = run(blend_s=0.3), run(anchor_s=1.0)
        self.assertLessEqual(blended, 0.8 + 1e-6)  # pulled back toward the restart point at every switch
        self.assertGreater(anchored, 1.3)  # a lead well past one chunk's travel ...
        steady = 0.4 + 0.4 / (1 - np.exp(-0.4)) * np.exp(-0.2)
        self.assertLess(anchored, steady)  # ... but bounded by the decay

    def test_anchor_overrides_blend_and_zero_keeps_old_behaviour(self):
        r = ChunkResampler(30)
        r.set_chunk(np.zeros((40, 4)), t0=0.0)
        r.set_chunk(np.ones((40, 4)), t0=0.3, blend_s=0.2, now=0.5, anchor_s=0.0)
        np.testing.assert_allclose(r.token_at(0.6), np.full(4, 0.5), atol=1e-6)  # plain cross-fade
        r2 = ChunkResampler(30)
        r2.set_chunk(np.zeros((40, 4)), t0=0.0)
        r2.set_chunk(np.ones((40, 4)), t0=0.3, blend_s=0.2, now=0.5, anchor_s=1.0)
        np.testing.assert_allclose(r2.token_at(0.5), np.zeros(4), atol=1e-6)  # anchored: starts where it was
        np.testing.assert_allclose(r2.token_at(0.6), np.full(4, 1 - np.exp(-0.1)), atol=1e-5)
