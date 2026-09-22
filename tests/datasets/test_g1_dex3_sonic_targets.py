import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
from sonic_targets import ISAAC_FROM_MOTOR, NOMINAL_BODY, build_encoder_inputs, combine_tokens_and_hands


class SonicTargetTests(unittest.TestCase):
    def setUp(self):
        self.limits = np.tile([-10.0, 10.0], (28, 1))

    def test_stationary_g1_input_matches_native_offsets_and_heading(self):
        actions = np.tile(np.arange(28) / 100.0, (60, 1))
        inputs, hands, report = build_encoder_inputs(actions, self.limits)
        self.assertEqual(inputs.shape, (60, 1751))
        body = NOMINAL_BODY.copy()
        body[15:] = actions[0, :14]
        np.testing.assert_allclose(inputs[0, 4:294].reshape(10, 29), np.tile(body[ISAAC_FROM_MOTOR], (10, 1)))
        np.testing.assert_array_equal(inputs[:, :4], 0)  # mode id 0, three zeros; NOT one-hot
        np.testing.assert_allclose(inputs[:, 294:584], 0, atol=1e-6)
        np.testing.assert_array_equal(inputs[0, 584:644].reshape(10, 6), np.tile([1, 0, 0, 1, 0, 0], (10, 1)))
        np.testing.assert_array_equal(inputs[:, 644:], 0)
        np.testing.assert_allclose(hands, actions[:, 14:])
        self.assertEqual(report["encoder_mode"], 0)

    def test_future_window_advances_100ms_and_holds_at_episode_end(self):
        actions = np.zeros((61, 28))
        actions[:, 0] = np.arange(61) / 30 * 0.5
        inputs, _, _ = build_encoder_inputs(actions, self.limits)
        arm_slot = int(np.flatnonzero(ISAAC_FROM_MOTOR == 15)[0])
        np.testing.assert_allclose(
            inputs[0, 4:294].reshape(10, 29)[:, arm_slot], np.arange(10) * 0.05, atol=1e-6
        )
        np.testing.assert_allclose(inputs[-1, 4:294].reshape(10, 29)[:, arm_slot], 1.0, atol=1e-6)
        np.testing.assert_array_equal(inputs[-1, 294:584], 0)

    def test_joint_limit_and_speed_limits_are_applied_without_adding_rows(self):
        actions = np.zeros((31, 28))
        actions[1:] = 100
        limits = np.tile([-0.5, 0.5], (28, 1))
        inputs, hands, report = build_encoder_inputs(actions, limits)
        self.assertEqual(len(inputs), len(actions))
        self.assertGreater(report["clipped_values"], 0)
        self.assertGreater(report["rate_limited_values"], 0)
        self.assertLessEqual(float(np.max(np.abs(hands))), 0.5)
        # First 100ms reference is bounded by the 1rad/s arm limit.
        arm_slot = int(np.flatnonzero(ISAAC_FROM_MOTOR == 15)[0])
        self.assertLessEqual(inputs[0, 4:294].reshape(10, 29)[1, arm_slot], 0.100001)

    def test_one_frame_and_hand_order(self):
        actions = np.arange(28, dtype=float)[None] / 100
        inputs, hands, _ = build_encoder_inputs(actions, self.limits)
        np.testing.assert_array_equal(inputs[:, 294:584], 0)
        tokens = np.tile(np.arange(64) / 16, (1, 1))
        combined = combine_tokens_and_hands(tokens, hands)
        self.assertEqual(combined.shape, (1, 78))
        np.testing.assert_allclose(combined[0, 64:], actions[0, 14:])

    def test_invalid_actions_or_tokens_fail(self):
        for values in (np.zeros((0, 28)), np.zeros((3, 27)), np.full((2, 28), np.nan)):
            with self.assertRaises(ValueError):
                build_encoder_inputs(values, self.limits)
        with self.assertRaises(ValueError):
            combine_tokens_and_hands(np.full((1, 64), 0.01), np.zeros((1, 14)))


if __name__ == "__main__":
    unittest.main()
