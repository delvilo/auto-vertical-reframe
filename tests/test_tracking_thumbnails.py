"""CPU grayscale-first thumbnails keep geometry and conservative flow behavior."""
import unittest

import cv2
import numpy as np

from reframe.config import AppConfig
from reframe.contracts import TrackObservation
from reframe.perception.motion import SparseMotion
from reframe.perception.segmentation import SegmentationTracker


def tracker(method):
    instance = SegmentationTracker.__new__(SegmentationTracker)
    instance.config = AppConfig(seg_thumbnail_method=method)
    return instance


class TrackingThumbnailTests(unittest.TestCase):
    def test_precrop_views_preserve_source_and_geometry_with_bounded_rounding(self):
        rng = np.random.default_rng(91)
        for height, width in ((2160, 3840), (1081, 1921), (180, 320)):
            full = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            original = full.copy()
            for start in (0, width // 4, width // 2):
                with self.subTest(height=height, width=width, start=start):
                    frame = full[:, start:start + width // 2]
                    frame.flags.writeable = False
                    gray = tracker("gray-area")._thumbnail(frame)
                    reference = tracker("bgr-area")._thumbnail(frame)
                    self.assertEqual(gray.shape, reference.shape)
                    self.assertEqual(gray.dtype, np.uint8)
                    self.assertLessEqual(max(gray.shape), 320)
                    # Integer color/area rounding is allowed, not bitwise identity.
                    self.assertLessEqual(np.abs(gray.astype(int) - reference).max(), 2)
                    self.assertFalse(np.shares_memory(gray, frame))
            np.testing.assert_array_equal(full, original)

    def test_reference_is_previous_bgr_area_algorithm(self):
        frame = np.random.default_rng(2).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        previous = cv2.cvtColor(cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA),
                                cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(tracker("bgr-area")._thumbnail(frame), previous)

    def test_real_flow_recovers_translation_in_inference_coordinates(self):
        small = np.full((180, 320, 3), 20, np.uint8)
        small[30:150, 90:220] = np.random.default_rng(32).integers(
            20, 240, (120, 130, 3), dtype=np.uint8)
        previous = cv2.resize(small, (1920, 1080), interpolation=cv2.INTER_NEAREST)
        current = cv2.warpAffine(previous, np.float32([[1, 0, 18], [0, 1, -12]]),
                                 (1920, 1080), borderValue=(20, 20, 20))
        subject = TrackObservation(0, 0, .95, (540., 180., 1320., 900.), 9)
        for method in ("gray-area", "bgr-area"):
            with self.subTest(method=method):
                helper = tracker(method)
                result, reason = SparseMotion().estimate(helper._thumbnail(previous),
                    helper._thumbnail(current), (subject,), 1920, 1080)
                self.assertIsNone(reason)
                np.testing.assert_allclose(result[9][:2], (18, -12), atol=1.)

    def test_new_local_subject_and_global_change_still_force_detection(self):
        source = np.full((1080, 1920, 3), 20, np.uint8)
        local = source.copy()
        local[180:400, 480:800] = (100, 240, 240)
        global_change = np.full_like(source, 240)
        for method in ("gray-area", "bgr-area"):
            helper = tracker(method)
            helper._gray = helper._thumbnail(source)
            with self.subTest(method=method):
                self.assertFalse(helper._changed(helper._thumbnail(source)))
                self.assertTrue(helper._changed(helper._thumbnail(local)))
                self.assertTrue(helper._changed(helper._thumbnail(global_change)))


if __name__ == "__main__":
    unittest.main()
