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
