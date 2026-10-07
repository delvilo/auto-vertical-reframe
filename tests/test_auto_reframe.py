import unittest
import sys
from unittest.mock import MagicMock

# Mock heavy video dependencies if not installed
for mod in ['cv2', 'scenedetect', 'ultralytics']:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

import numpy as np
from reframe.saliency.backends.handcrafted import HandcraftedSaliencyHelper
from reframe.camera import regression_velocity


class TestAutoReframe(unittest.TestCase):
    def test_regression_velocity_insufficient_points(self):
        """Test history with fewer than 3 points returns (0.0, 0.0)."""
        self.assertEqual(regression_velocity([]), (0.0, 0.0))
        self.assertEqual(regression_velocity([(0.0, 10.0, 20.0)]), (0.0, 0.0))
        self.assertEqual(regression_velocity([(0.0, 10.0, 20.0), (1.0, 12.0, 25.0)]), (0.0, 0.0))

    def test_regression_velocity_invalid_time_deltas(self):
        """Test history with identical or near-zero time steps returns (0.0, 0.0)."""
        history = [
            (1.0, 10.0, 20.0),
            (1.0, 15.0, 25.0),
            (1.0, 20.0, 30.0),
        ]
        self.assertEqual(regression_velocity(history), (0.0, 0.0))

    def test_regression_velocity_constant_motion(self):
        """Test history with constant velocity returns expected vx and vy slopes."""
        history = [
            (0.0, 10.0, 20.0),
            (1.0, 15.0, 17.0),
            (2.0, 20.0, 14.0),
            (3.0, 25.0, 11.0),
        ]
        vx, vy = regression_velocity(history)
        self.assertAlmostEqual(vx, 5.0)
        self.assertAlmostEqual(vy, -3.0)

    def test_regression_velocity_robustness_to_outliers(self):
        """Test that median pairwise slope estimation is robust to single-frame outliers (Theil-Sen)."""
        history = [
            (0.0, 0.0, 0.0),
            (1.0, 10.0, 20.0),
            (2.0, 100.0, -500.0),
            (3.0, 30.0, 60.0),
            (4.0, 40.0, 80.0),
        ]
        vx, vy = regression_velocity(history)
        self.assertAlmostEqual(vx, 10.0)
        self.assertAlmostEqual(vy, 20.0)

    def test_regression_velocity_stationary(self):
        """Test history with zero motion returns (0.0, 0.0)."""
        history = [
            (0.0, 50.0, 50.0),
            (1.0, 50.0, 50.0),
            (2.0, 50.0, 50.0),
        ]
        vx, vy = regression_velocity(history)
        self.assertAlmostEqual(vx, 0.0)
        self.assertAlmostEqual(vy, 0.0)

    def test_handcrafted_saliency_helper_compute_map(self):
        """Test HandcraftedSaliencyHelper compute_map execution and telemetry updates."""
        helper = HandcraftedSaliencyHelper()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        saliency_map = helper.compute_map(frame)
        self.assertIsNotNone(saliency_map)
        telemetry = helper.get_telemetry()
        self.assertEqual(telemetry["frames_total"], 1)
        self.assertEqual(telemetry["frames_backend"], 1)


if __name__ == "__main__":
    unittest.main()

