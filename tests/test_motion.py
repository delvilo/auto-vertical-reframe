"""Real sparse optical flow checks using deterministic synthetic image pairs."""
from dataclasses import replace
import unittest

import cv2
import numpy as np

from reframe.contracts import TrackObservation
from reframe.perception.motion import SparseMotion


class SparseMotionTests(unittest.TestCase):
    def setUp(self):
        self.gray = np.full((180, 320), 20, np.uint8)
        rng = np.random.default_rng(32)
        self.gray[30:150, 90:220] = rng.integers(20, 240, (120, 130), dtype=np.uint8)
        self.track = TrackObservation(0, 0, .95, (90., 30., 220., 150.), 9)
        self.motion = SparseMotion()

    def shifted(self, dx, dy):
        return cv2.warpAffine(self.gray, np.float32([[1, 0, dx], [0, 1, dy]]), (320, 180),
                              borderValue=20)

    def test_translation_and_stationary_video(self):
        for dx, dy in [(0, 0), (4, 2), (-3, 1)]:
            with self.subTest(dx=dx, dy=dy):
                moved = self.shifted(dx, dy)
                original = self.gray.copy()
                result, reason = self.motion.estimate(self.gray, moved, (self.track,), 320, 180)
                self.assertIsNone(reason)
                np.testing.assert_allclose(result[9][:2], (dx, dy), atol=.15)
                self.assertGreaterEqual(result[9][2], .6)
                np.testing.assert_array_equal(self.gray, original)

    def test_translation_returns_full_inference_coordinates(self):
        track = replace(self.track, box=tuple(x * 4 for x in self.track.box))
        result, reason = self.motion.estimate(self.gray, self.shifted(3, -2), (track,), 1280, 720)
        self.assertIsNone(reason)
        np.testing.assert_allclose(result[9][:2], (12, -8), atol=.6)

    def test_small_local_occlusion_does_not_bias_translation(self):
        moved = self.shifted(4, 2)
        moved[40:62, 100:125] = 0
        result, reason = self.motion.estimate(self.gray, moved, (self.track,), 320, 180)
        self.assertIsNone(reason)
        np.testing.assert_allclose(result[9][:2], (4, 2), atol=.3)

    def test_zoom_is_not_misreported_as_safe_translation(self):
        transform = cv2.getRotationMatrix2D((155, 90), 0, 1.15)
        moved = cv2.warpAffine(self.gray, transform, (320, 180), borderValue=20)
        result, reason = self.motion.estimate(self.gray, moved, (self.track,), 320, 180)
        self.assertEqual((result, reason), ({}, "flow_deformation"))

    def test_large_gray_frames_use_same_translation_coordinates(self):
        previous = cv2.resize(self.gray, (640, 360), interpolation=cv2.INTER_NEAREST)
        current = cv2.resize(self.shifted(4, 2), (640, 360), interpolation=cv2.INTER_NEAREST)
        track = replace(self.track, box=tuple(x * 2 for x in self.track.box))
        result, reason = self.motion.estimate(previous, current, (track,), 640, 360)
        self.assertIsNone(reason)
        np.testing.assert_allclose(result[9][:2], (8, 4), atol=.3)

    def test_blank_and_severe_occlusion_reject_prediction(self):
        blank = np.zeros_like(self.gray)
        for previous, current in [(blank, blank), (self.gray, blank)]:
            result, reason = self.motion.estimate(previous, current, (self.track,), 320, 180)
            self.assertEqual(result, {})
            self.assertIn(reason, {"flow_feature_poor", "flow_unreliable", "flow_deformation"})

    def test_invalid_geometry_never_returns_prediction(self):
        for box, reason in [((0, 30, 130, 150), "flow_boundary"),
                            ((90, 30, 340, 150), "flow_boundary"),
                            ((90, 30, 50, 150), "flow_invalid_box"),
                            ((90, np.nan, 220, 150), "flow_invalid_box")]:
            with self.subTest(box=box):
                result, actual = self.motion.estimate(self.gray, self.gray,
                                                      (replace(self.track, box=box),), 320, 180)
                self.assertEqual((result, actual), ({}, reason))

    def test_crowding_rejects_whole_proposal(self):
        competitor = replace(self.track, track_id=12, box=(100, 40, 230, 160))
        self.assertEqual(self.motion.estimate(self.gray, self.gray,
                                             (self.track, competitor), 320, 180), ({}, "flow_crowding"))

    def test_large_motion_rejects_prediction(self):
        result, reason = self.motion.estimate(self.gray, self.shifted(22, 0), (self.track,), 320, 180)
        self.assertEqual(result, {})
        self.assertIn(reason, {"flow_jump", "flow_unreliable", "flow_deformation"})

    def test_nonfinite_frame_rejects_prediction(self):
        invalid = self.gray.astype(np.float32)
        invalid[0, 0] = np.inf
        self.assertEqual(self.motion.estimate(invalid, invalid, (self.track,), 320, 180),
                         ({}, "invalid_flow_frame"))


if __name__ == "__main__":
    unittest.main()
