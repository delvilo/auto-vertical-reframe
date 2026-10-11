"""Tensor observation windows preserve all 16 source frames and stay lazy."""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from reframe.contracts import FrameContext
from reframe.frame_pixels import TensorFrame
from reframe.saliency.backends import deepgazemr


class RecordingTensorModel:
    def __init__(self):
        self.clips = []
        self.fail = False

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, clip):
        self.clips.append(clip.clone())
        if self.fail:
            raise RuntimeError("tensor model failed")
        return torch.arange(64, dtype=torch.float32, device=clip.device).reshape(8, 8) / 64


class TensorDeepGazeTests(unittest.TestCase):
    def setUp(self):
        self.metrics = {}
        self.model = RecordingTensorModel()
        loader = patch.object(deepgazemr, "load_deepgaze_mr", return_value=(self.model, {}))
        self.loader = loader.start()
        self.addCleanup(loader.stop)
        self.helper = deepgazemr.DeepGazeMRSaliencyHelper(device="cpu", max_side=128)
        self.addCleanup(self.helper.close)
        self.helper.fallback.compute_map = MagicMock(
            side_effect=lambda frame: np.zeros(frame.shape[:2], np.float32))

    def frame(self, index, shape=(64, 128), device="cpu"):
        pixels = torch.empty((3, *shape), dtype=torch.uint8, device=device)
        pixels[0].fill_(200)
        pixels[1].fill_(100)
        pixels[2].fill_(index)
        return TensorFrame(pixels, self.metrics)

    def observe(self, first, last, shape=(64, 128), device="cpu"):
        for index in range(first, last + 1):
            self.helper.observe_frame(self.frame(index, shape, device))

    def predict(self, index, shape=(64, 128), device="cpu"):
        return self.helper.predict(self.frame(index, shape, device), FrameContext(
            index, (index - 1) / 30, shape[1], shape[0], 1))

    def assert_clip(self, first, last):
        clip = self.model.clips[-1].cpu()
        torch.testing.assert_close(clip[:, 2, 0, 0] * 255,
                                   torch.arange(first, last + 1, dtype=torch.float32))
        torch.testing.assert_close(clip[:, 0, 0, 0] * 255, torch.full((16,), 200.0))
        torch.testing.assert_close(clip[:, 1, 0, 0] * 255, torch.full((16,), 100.0))

    def test_preprocess_matches_existing_numpy_for_all_uint8_values(self):
        rgb = torch.arange(256, dtype=torch.uint8).expand(3, 64, 256)
        bgr = rgb.flip(0).permute(1, 2, 0).contiguous().numpy()
        expected = torch.from_numpy(self.helper._preprocess_frame(bgr))
        actual = self.helper._preprocess_tensor_frame(rgb)
        self.assertTrue(torch.equal(actual, expected))

    def test_batched_preprocess_reuses_destination_and_matches_numpy_exactly(self):
        pixels = torch.arange(256, dtype=torch.uint8).repeat(16 * 3).reshape(16, 3, 4, 64)
        original = pixels.clone()
        destination = torch.empty_like(pixels, dtype=torch.float32)
        actual = self.helper._preprocess_tensor_frame(pixels, out=destination)
        expected = torch.stack([
            torch.from_numpy(self.helper._preprocess_frame(
                frame.flip(0).permute(1, 2, 0).contiguous().numpy()))
            for frame in pixels
        ])
        self.assertEqual(actual.data_ptr(), destination.data_ptr())
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(pixels, original))

    def test_observation_stays_uint8_bounded_without_model_or_download(self):
        self.observe(1, 80)
        self.loader.assert_not_called()
        self.assertIsNone(self.helper.tensor_ring)
        self.assertIsNone(self.helper.host_ring)
        self.assertIsNone(self.helper.cpu_ring)
        self.assertEqual(self.helper.observation_tensor_ring.dtype, torch.uint8)
        self.assertEqual(tuple(self.helper.observation_tensor_ring.shape), (16, 3, 64, 128))
        self.assertEqual(self.metrics.get("cpu_materializations", 0), 0)
        self.assertEqual(self.helper.telemetry()["tensor_preprocessed_frames"], 0)

    def test_window_copies_observations_instead_of_retaining_decoder_storage(self):
        frame = self.frame(7)
        self.helper.observe_frame(frame)
        frame.rgb.fill_(255)
        self.assertEqual(self.helper.observation_tensor_ring[0, 2, 0, 0].item(), 7)

    def test_lazy_warmup_incremental_updates_long_gaps_and_chronology(self):
        self.observe(1, 15)
        result = self.predict(15)
        self.assertEqual((result.backend, result.status), ("handcrafted", "warmup"))
        self.loader.assert_not_called()
        self.assertIsInstance(self.helper.fallback.compute_map.call_args.args[0], np.ndarray)
        self.observe(16, 19)
        self.assertEqual(self.predict(19).backend, "deepgazemr")
        self.assert_clip(4, 19)
        self.assertEqual(self.helper.telemetry()["tensor_preprocessed_frames"], 16)
        self.observe(20, 22)
        self.predict(22)
        self.assert_clip(7, 22)
        self.assertEqual(self.helper.telemetry()["tensor_preprocessed_frames"], 19)
        self.observe(23, 65)
        self.predict(65)
        self.assert_clip(50, 65)
        self.assertEqual(self.helper.telemetry()["tensor_preprocessed_frames"], 35)
        self.assertEqual(self.helper.telemetry()["tensor_preprocess_batches"], 5)
        self.assertEqual(self.helper.telemetry()["tensor_cross_device_copies"], 0)
        self.assertEqual(self.metrics.get("cpu_materializations", 0), 1)  # Warmup fallback only.
        self.assertIsNone(self.helper.host_ring)
        self.loader.assert_called_once_with()

    def test_wrapped_batches_match_reference_pixels_and_update_only_unseen_frames(self):
        expected_frames = []
        pattern = torch.arange(3 * 64 * 128, dtype=torch.int32).reshape(3, 64, 128)

        def observe(first, last):
            for index in range(first, last + 1):
                pixels = ((pattern + index) % 256).to(torch.uint8)
                self.helper.observe_frame(TensorFrame(pixels, self.metrics))
                expected_frames.append(torch.from_numpy(self.helper._preprocess_frame(
                    pixels.flip(0).permute(1, 2, 0).contiguous().numpy())))

        observe(1, 30)  # Full window starts at slot 14: two contiguous spans.
        with patch.object(self.helper, "_preprocess_tensor_frame",
                          wraps=self.helper._preprocess_tensor_frame) as preprocess:
            self.predict(30)
        self.assertEqual([call.args[0].shape[0] for call in preprocess.call_args_list], [2, 14])
        self.assertTrue(torch.equal(self.model.clips[-1], torch.stack(expected_frames[-16:])))
        ring_storage = self.helper.tensor_ring.data_ptr()
        unchanged = self.helper.tensor_ring[3:14].clone()

        observe(31, 35)  # Five unseen frames cross the physical ring boundary.
        with patch.object(self.helper, "_preprocess_tensor_frame",
                          wraps=self.helper._preprocess_tensor_frame) as preprocess:
            self.predict(35)
        self.assertEqual([call.args[0].shape[0] for call in preprocess.call_args_list], [2, 3])
        self.assertEqual(self.helper.tensor_ring.data_ptr(), ring_storage)
        self.assertTrue(torch.equal(self.helper.tensor_ring[3:14], unchanged))
        self.assertTrue(torch.equal(self.helper.tensor_ring[:16], self.helper.tensor_ring[16:]))
        self.assertTrue(torch.equal(self.model.clips[-1], torch.stack(expected_frames[-16:])))
        self.assertEqual(self.helper.telemetry()["tensor_preprocessed_frames"], 21)
        self.assertEqual(self.helper.telemetry()["tensor_preprocess_batches"], 4)

    def test_repeated_prediction_without_observations_does_not_preprocess_again(self):
        self.observe(1, 16)
        self.predict(16)
        original = self.model.clips[-1].clone()
        with patch.object(self.helper, "_preprocess_tensor_frame",
                          wraps=self.helper._preprocess_tensor_frame) as preprocess:
            self.predict(16)
        preprocess.assert_not_called()
        self.assertTrue(torch.equal(self.model.clips[-1], original))
        self.assertEqual(self.helper.telemetry()["tensor_preprocessed_frames"], 16)
        self.assertEqual(self.helper.telemetry()["tensor_preprocess_batches"], 1)

    def test_reset_and_geometry_change_require_new_complete_window(self):
        self.observe(1, 16)
        self.predict(16)
        self.helper.reset()
        self.assertEqual(self.helper.ring_count, 0)
        self.observe(17, 31)
        self.assertEqual(self.predict(31).status, "warmup")
        self.observe(32, 32)
        self.predict(32)
        self.assert_clip(17, 32)
        self.observe(33, 47, shape=(96, 128))
        self.assertEqual(self.predict(47, (96, 128)).status, "warmup")
        self.observe(48, 48, shape=(96, 128))
        self.assertEqual(self.predict(48, (96, 128)).map.shape, (96, 128))
        self.assert_clip(33, 48)
        self.assertEqual(self.helper.telemetry()["tensor_preprocessed_frames"], 48)
        self.assertEqual(self.helper.telemetry()["tensor_preprocess_batches"], 3)
        self.loader.assert_called_once_with()

    def test_numpy_tensor_mode_transitions_do_not_reuse_stale_windows(self):
        self.observe(1, 16)
        self.predict(16)
        for index in range(17, 33):
            frame = self.frame(index).bgr()
            self.helper.observe_frame(frame)
        self.assertIsNone(self.helper.observation_tensor_ring)
        context = FrameContext(32, 31 / 30, 128, 64, 1)
        self.helper.predict(frame, context)
        self.assert_clip(17, 32)
        self.assertIsNotNone(self.helper.host_ring)
        self.observe(33, 48)
        self.predict(48)
        self.assert_clip(33, 48)
        self.assertIsNone(self.helper.host_ring)
        self.assertIsNone(self.helper.cpu_ring)
        self.loader.assert_called_once_with()

    def test_forward_failure_materializes_only_small_fallback_image(self):
        self.observe(1, 16)
        self.model.fail = True
        with self.assertLogs(level="WARNING"):
            result = self.predict(16)
        self.assertEqual((result.backend, result.status), ("handcrafted", "fallback"))
        self.assertEqual(self.metrics["cpu_materialized_bytes"], 64 * 128 * 3)
        self.assertEqual(self.helper.telemetry()["actual_forward_calls"], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for device transfer checks")
    def test_cuda_observations_preprocess_without_host_round_trip(self):
        self.helper.device_name = "cuda:0"
        self.observe(1, 16, device="cuda:0")
        self.predict(16, device="cuda:0")
        self.assertEqual(self.helper.observation_tensor_ring.device.type, "cuda")
        self.assertEqual(self.helper.tensor_ring.device.type, "cuda")
        self.assertEqual(self.helper.telemetry()["tensor_cross_device_copies"], 0)
        self.assertEqual(self.metrics.get("device_to_host_bytes", 0), 0)
        self.assert_clip(1, 16)
        self.observe(17, 30, device="cuda:0")
        self.predict(30, device="cuda:0")
        self.observe(31, 35, device="cuda:0")
        self.predict(35, device="cuda:0")
        self.assert_clip(20, 35)
        self.assertEqual(self.helper.telemetry()["tensor_preprocessed_frames"], 35)
        self.assertEqual(self.helper.telemetry()["tensor_preprocess_batches"], 4)
        self.assertEqual(self.helper.telemetry()["tensor_cross_device_copies"], 0)
        self.assertEqual(self.metrics.get("device_to_host_bytes", 0), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for normalization parity")
    def test_cuda_batched_normalization_matches_numpy_within_float32_precision(self):
        pixels = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 16, 16).expand(16, 3, 16, 16)
        expected = self.helper._preprocess_tensor_frame(pixels)
        device_pixels = pixels.to("cuda:0")
        output = torch.empty_like(device_pixels, dtype=torch.float32)
        actual = self.helper._preprocess_tensor_frame(device_pixels, out=output)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=torch.finfo(torch.float32).eps)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for device transfer checks")
    def test_cuda_observations_cpu_saliency_copies_small_uint8_frames(self):
        self.observe(1, 16, device="cuda:0")
        self.predict(16, device="cuda:0")
        self.assertEqual(self.helper.tensor_ring.device.type, "cpu")
        self.assertEqual(self.helper.telemetry()["tensor_cross_device_copies"], 16)
        self.assertEqual(self.helper.telemetry()["tensor_cross_device_bytes"], 16 * 64 * 128 * 3)
        self.assertEqual(self.helper.telemetry()["tensor_cross_device_batches"], 1)
        self.assert_clip(1, 16)
        self.observe(17, 35, device="cuda:0")
        self.predict(35, device="cuda:0")
        self.assert_clip(20, 35)
        self.assertEqual(self.helper.telemetry()["tensor_cross_device_copies"], 32)
        self.assertEqual(self.helper.telemetry()["tensor_cross_device_bytes"], 32 * 64 * 128 * 3)
        self.assertEqual(self.helper.telemetry()["tensor_cross_device_batches"], 3)


if __name__ == "__main__":
    unittest.main()
