"""Backend scheduling/provenance tests. OpenCV is real; no model weights needed."""
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from reframe.contracts import BackendPrediction, FrameContext, FrameObservations
from reframe.saliency.base import SaliencyBackend
from reframe.saliency.cache import SaliencyCache
from reframe.saliency.service import SaliencyService
from reframe.saliency.regions import extract_saliency_region
from reframe.saliency.backends.handcrafted import HandcraftedSaliencyHelper


def context(index, scene=1, width=128, height=64):
    return FrameContext(index, (index - 1) / 30, width, height, scene)


class TemporalBackend(SaliencyBackend):
    """Test backend requiring 16 consecutive observations before neural output."""
    def __init__(self):
        self.events = []
        self.window = []

    def load(self):
        self.events.append(("load",))

    def observe(self, frame, context):
        self.events.append(("observe", context.frame_index))
        self.window.append(context.frame_index)

    def predict(self, frame, context):
        self.events.append(("predict", context.frame_index))
        ready = len(self.window) >= 16
        return BackendPrediction(np.full(frame.shape[:2], .8 if ready else .2, np.float32),
                                 "neural" if ready else "handcrafted",
                                 "predicted" if ready else "warmup",
                                 None if ready else "temporal_window")

    def reset(self):
        self.events.append(("reset",))
        self.window.clear()

    def close(self):
        self.events.append(("close",))

    def telemetry(self):
        return {"requested_backend": "neural"}


@unittest.skipIf(isinstance(cv2, MagicMock), "requires real OpenCV")
class SaliencyContractTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((64, 128, 3), np.uint8)
        self.backend = TemporalBackend()
        self.service = SaliencyService(self.backend, interval=3, max_side=128)
        self.addCleanup(self.service.close)

    def process(self, index, scene=1):
        return self.service.process(self.frame, FrameObservations(context(index, scene)))

    def test_temporal_ingestion_continues_during_skipped_predictions(self):
        results = [self.process(i) for i in range(1, 20)]
        self.assertEqual([event[1] for event in self.backend.events if event[0] == "observe"],
                         list(range(1, 20)))
        predictions = [event[1] for event in self.backend.events if event[0] == "predict"]
        self.assertEqual(predictions, [1, 4, 7, 10, 13, 16, 19])
        for i in predictions:
            self.assertLess(self.backend.events.index(("observe", i)),
                            self.backend.events.index(("predict", i)))
        self.assertEqual(self.backend.events.count(("load",)), 1)
        self.assertEqual((results[0].backend, results[0].status), ("handcrafted", "warmup"))
        self.assertEqual((results[15].backend, results[15].status), ("neural", "predicted"))
        self.assertEqual(self.service.telemetry()["sample_propagated"], 12)

    def test_propagation_does_not_advance_inference_time(self):
        first, second, third, fourth = [self.process(i) for i in range(1, 5)]
        self.assertEqual(second.inferred_at, first.inferred_at)
        self.assertEqual(third.inferred_at, first.inferred_at)
        self.assertEqual(third.frame.frame_index, 3)
        self.assertEqual(third.source, "propagated")
        self.assertEqual(fourth.inferred_at.frame_index, 4)
        self.assertEqual(fourth.source, "ema")
        self.assertEqual(self.service.telemetry()["inferred_frame_index"], 4)

    def test_scene_change_clears_window_and_cached_map(self):
        for i in range(1, 18):
            self.process(i)
        result = self.process(18, scene=2)
        self.assertEqual(self.backend.window, [18])
        self.assertEqual(result.source, "refresh")
        self.assertEqual(result.status, "warmup")
        self.assertEqual(result.inferred_at.scene_index, 2)
        np.testing.assert_allclose(result.map, .2)

    def test_geometry_change_requires_fresh_map(self):
        self.process(1)
        frame = np.zeros((96, 160, 3), np.uint8)
        result = self.service.process(frame, FrameObservations(context(2, width=160, height=96)))
        self.assertEqual(result.source, "refresh")
        self.assertEqual(result.map.shape, (77, 128))
        self.assertEqual((result.frame.width, result.frame.height), (160, 96))
        self.assertEqual(self.backend.window, [2])

    def test_ema_retains_existing_weights_and_values(self):
        with patch.object(self.backend, "predict", side_effect=[
                BackendPrediction(np.full((64, 128), .2, np.float32), "handcrafted"),
                BackendPrediction(np.full((64, 128), .8, np.float32), "neural")]):
            for i in range(1, 6):
                result = self.process(i)
        np.testing.assert_allclose(result.map, .65 * .8 + .35 * .2)
        self.assertEqual(result.backend, "neural")
        self.assertAlmostEqual(result.history_weight, .35)
        self.assertEqual(result.inferred_at.frame_index, 4)

    def test_flow_moves_map_without_changing_inference_provenance(self):
        cache = SaliencyCache()
        saliency = np.zeros((64, 128), np.float32)
        saliency[20:30, 30:40] = 1
        cache.refresh(BackendPrediction(saliency, "test"), context(1), None, saliency.shape)
        cache.previous_gray = np.zeros(saliency.shape, np.uint8)
        points = np.array([[[30, 20]], [[39, 20]], [[30, 29]], [[39, 29]]], np.float32)
        with patch.object(cv2, "goodFeaturesToTrack", return_value=points), \
                patch.object(cv2, "calcOpticalFlowPyrLK", return_value=(points + [4, 2], np.ones((4, 1)), None)):
            moved = cache.propagate(cache.previous_gray)
        result = cache.reuse(moved, context(2))
        np.testing.assert_array_equal(result.map[22:32, 34:44], np.ones((10, 10)))
        self.assertEqual(result.inferred_at.frame_index, 1)
        self.assertEqual(result.frame.frame_index, 2)

    def test_map_coordinates_recover_original_image_geometry(self):
        saliency = np.zeros((32, 64), np.float32)
        saliency[10:12, 30:35] = 1
        region = extract_saliency_region(saliency, (0, 0, 640, 320), (320, 640))
        np.testing.assert_allclose(region[:6], [320, 105, 300, 100, 350, 120])

    def test_invalid_backend_results_are_rejected(self):
        for values in (np.full((8, 8), np.nan), np.zeros((8, 8, 3)), np.full((8, 8), -1)):
            with self.subTest(shape=values.shape), \
                    patch.object(self.backend, "predict", return_value=BackendPrediction(values, "broken")):
                self.service.reset()
                with self.assertRaises(ValueError):
                    self.process(1)

    def test_handcrafted_implements_same_contract_with_real_opencv(self):
        backend = HandcraftedSaliencyHelper()
        backend.load()
        frame = np.random.default_rng(31).integers(0, 256, (128, 256, 3), dtype=np.uint8)
        backend.observe(frame, context(1, width=256, height=128))
        result = backend.predict(frame, context(1, width=256, height=128))
        self.assertEqual(result.map.shape, frame.shape[:2])
        self.assertEqual(result.map.dtype, np.float32)
        self.assertTrue(np.isfinite(result.map).all())
        self.assertGreaterEqual(float(result.map.min()), -1e-6)
        self.assertLessEqual(float(result.map.max()), 1 + 1e-6)
        backend.close()
        self.assertIsNone(backend.prev_gray_small)
