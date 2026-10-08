"""Lazy DeepGaze window tests with real OpenCV and no weights or GPU."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from reframe.contracts import FrameContext
from reframe.saliency.backends import deepgazemr


class ArrayTensor:
    """Minimal tensor double that preserves ring-buffer views and copy semantics."""

    def __init__(self, array):
        self.array = array
        self.shape = array.shape

    def __getitem__(self, index):
        return ArrayTensor(self.array[index])

    def copy_(self, other, **kwargs):
        self.array[...] = other.array
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class RecordingModel:
    def __init__(self):
        self.clips = []
        self.fail = False

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, clip):
        self.clips.append(clip.array.copy())
        if self.fail:
            raise RuntimeError("fake model forward failed")
        return ArrayTensor(np.arange(64, dtype=np.float32).reshape(8, 8) / 64)


@unittest.skipIf(isinstance(cv2, MagicMock), "requires real OpenCV")
class LazyDeepGazeTests(unittest.TestCase):
    def setUp(self):
        self.model = RecordingModel()
        self.fake_torch = SimpleNamespace(
            hub=SimpleNamespace(load=MagicMock(return_value=self.model)),
            empty=MagicMock(side_effect=lambda shape, **kwargs: ArrayTensor(
                np.empty(shape, dtype=np.float32))),
            from_numpy=MagicMock(side_effect=ArrayTensor),
            inference_mode=nullcontext,
            is_tensor=lambda tensor: isinstance(tensor, ArrayTensor),
            float32=np.float32,
        )
        self.torch_patch = patch.object(deepgazemr, "torch", self.fake_torch)
        self.torch_patch.start()
        self.addCleanup(self.torch_patch.stop)
        self.helper = deepgazemr.DeepGazeMRSaliencyHelper(device="cpu", max_side=128)
        self.addCleanup(self.helper.close)
        self.helper.fallback.compute_map = MagicMock(
            side_effect=lambda frame: np.zeros(frame.shape[:2], np.float32))

    @staticmethod
    def frame(index, shape=(64, 128)):
        frame = np.empty((*shape, 3), np.uint8)
        frame[:] = (index, 100, 200)
        return frame

    def observe(self, start, stop, shape=(64, 128)):
        for index in range(start, stop + 1):
            self.helper.observe_frame(self.frame(index, shape))

    def predict(self, index, shape=(64, 128)):
        return self.helper.predict(self.frame(index, shape), FrameContext(
            index, (index - 1) / 30, shape[1], shape[0], 1))

    def assert_clip(self, first, last):
        clip = self.model.clips[-1]
        np.testing.assert_allclose(clip[:, 2, 0, 0] * 255, range(first, last + 1),
                                   atol=1e-5)
        np.testing.assert_allclose(clip[:, 0, 0, 0] * 255, 200, atol=1e-5)
        np.testing.assert_allclose(clip[:, 1, 0, 0] * 255, 100, atol=1e-5)

    def test_observing_stable_video_never_loads_or_stages_model(self):
        self.observe(1, 80)
        self.fake_torch.hub.load.assert_not_called()
        self.fake_torch.empty.assert_not_called()
        self.fake_torch.from_numpy.assert_not_called()
        self.assertEqual(self.helper.cpu_ring.dtype, np.uint8)
        self.assertEqual(self.helper.cpu_ring.shape, (16, 64, 128, 3))
        self.assertEqual(self.helper.telemetry()["temporal_window_frames"], 16)
        self.assertEqual(self.helper.telemetry()["observed_frames"], 80)
        self.assertEqual(self.helper.telemetry()["actual_forward_calls"], 0)

    def test_warmup_predictions_do_not_load_until_full_window(self):
        for index in range(1, 16):
            self.observe(index, index)
            result = self.predict(index)
            self.assertEqual((result.backend, result.status), ("handcrafted", "warmup"))
        self.fake_torch.hub.load.assert_not_called()
        self.fake_torch.empty.assert_not_called()
        self.observe(16, 16)
        self.assertEqual(self.predict(16).backend, "deepgazemr")
        self.fake_torch.hub.load.assert_called_once_with(
            "mtangemann/deepgazemr", "DeepGazeMR", pretrained=True, trust_repo=False)
        self.assert_clip(1, 16)
        self.assertEqual(self.helper.telemetry()["prediction_requests"], 16)
        self.assertEqual(self.helper.telemetry()["actual_forward_calls"], 1)
        self.assertEqual(self.helper.telemetry()["frames_fallback"], 15)

    def test_late_first_prediction_uploads_latest_window_in_order(self):
        self.observe(1, 40)
        self.predict(40)
        self.assert_clip(25, 40)
        self.assertEqual(self.fake_torch.from_numpy.call_count, 16)

    def test_incremental_upload_and_long_gap_preserve_chronology(self):
        self.observe(1, 19)
        self.predict(19)
        self.fake_torch.from_numpy.reset_mock()
        self.observe(20, 22)
        self.predict(22)
        self.assert_clip(7, 22)
        self.assertEqual(self.fake_torch.from_numpy.call_count, 3)
        self.fake_torch.from_numpy.reset_mock()
        self.observe(23, 65)
        self.predict(65)
        self.assert_clip(50, 65)
        self.assertEqual(self.fake_torch.from_numpy.call_count, 16)
        self.fake_torch.hub.load.assert_called_once()
        self.assertEqual(self.helper.telemetry()["actual_forward_calls"], 3)

    def test_scene_reset_retains_loaded_model_but_requires_new_window(self):
        self.observe(1, 19)
        self.predict(19)
        self.helper.reset()
        self.assertIs(self.helper.model, self.model)
        self.assertEqual(self.helper.telemetry()["temporal_window_frames"], 0)
        self.observe(20, 34)
        self.assertEqual(self.predict(34).status, "warmup")
        self.observe(35, 35)
        self.predict(35)
        self.assert_clip(20, 35)
        self.fake_torch.hub.load.assert_called_once()
        self.assertEqual(self.helper.telemetry()["actual_forward_calls"], 2)

    def test_geometry_change_restarts_window_without_reloading(self):
        self.observe(1, 16)
        self.predict(16)
        self.observe(17, 17, (96, 128))
        self.assertEqual(self.predict(17, (96, 128)).status, "warmup")
        self.observe(18, 32, (96, 128))
        result = self.predict(32, (96, 128))
        self.assertEqual(result.map.shape, (96, 128))
        self.assert_clip(17, 32)
        self.fake_torch.hub.load.assert_called_once()

    def test_forward_failure_is_counted_without_claiming_neural_output(self):
        self.observe(1, 16)
        self.model.fail = True
        with self.assertLogs(level="WARNING"):
            result = self.predict(16)
        self.assertEqual((result.backend, result.status), ("handcrafted", "fallback"))
        telemetry = self.helper.telemetry()
        self.assertEqual(telemetry["prediction_requests"], 1)
        self.assertEqual(telemetry["actual_forward_calls"], 1)
        self.assertEqual(telemetry["frames_backend"], 0)
        self.assertEqual(telemetry["frames_fallback"], 1)

    def test_no_pytorch_does_not_disrupt_cpu_window_or_warmup(self):
        with patch.object(deepgazemr, "torch", None):
            self.observe(1, 15)
            self.assertEqual(self.predict(15).status, "warmup")
            self.observe(16, 16)
            with self.assertLogs(level="WARNING"):
                self.assertEqual(self.predict(16).status, "fallback")
        self.assertEqual(self.helper.telemetry()["actual_forward_calls"], 0)
        self.assertTrue(self.helper._disabled)


if __name__ == "__main__":
    unittest.main()
