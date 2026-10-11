"""Tensor frames preserve geometry and expose their CPU transfer costs."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.data.augment import LetterBox

from reframe.frame_pixels import TensorFrame, _bilinear_sampled_resize
from reframe.precrop import InferenceRegion


def frame_from_bgr(pixels, metrics=None):
    return TensorFrame(torch.from_numpy(pixels[..., ::-1].copy()).permute(2, 0, 1), metrics)


class TensorFrameTests(unittest.TestCase):
    def test_cpu_large_downscale_preserves_native_pixels_and_path(self):
        rgb = torch.randint(0, 256, (3, 217, 383), dtype=torch.uint8)
        expected = F.interpolate(rgb[None].float(), size=(29, 47), mode="bilinear",
                                 align_corners=False)[0].round().clamp_(0, 255).to(torch.uint8)
        with patch("reframe.frame_pixels._bilinear_sampled_resize",
                   side_effect=AssertionError("CPU must keep native resize")):
            frame = TensorFrame(rgb)
            actual = frame.resized_rgb((47, 29), "bilinear")
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(frame.metrics.get("bilinear_sampled_calls", 0), 0)

    def assert_sampled_bilinear_matches_reference(self, rgb, size):
        expected = F.interpolate(rgb[None].float(), size=size, mode="bilinear",
                                 align_corners=False)[0].round().clamp_(0, 255)
        actual = _bilinear_sampled_resize(rgb, *size).round().clamp_(0, 255)
        self.assertEqual(actual.shape, (3, *size))
        self.assertEqual(actual.device, rgb.device)
        self.assertLessEqual(int((actual - expected).abs().max()), 1)

    def test_sampled_bilinear_matches_native_for_odd_tall_wide_and_single_pixel_axes(self):
        generator = torch.Generator().manual_seed(137)
        for shape, size in (((217, 383), (47, 91)), ((109, 37), (23, 7)),
                            ((997, 579), (113, 73)), ((1, 257), (1, 61)),
                            ((257, 1), (61, 1)), ((2160, 3840), (360, 640))):
            with self.subTest(shape=shape, size=size):
                rgb = torch.randint(0, 256, (3, *shape), dtype=torch.uint8,
                                    generator=generator)
                self.assert_sampled_bilinear_matches_reference(rgb, size)

    def test_sampled_bilinear_preserves_strided_precrops_and_contrast_edges(self):
        height, width = 347, 619
        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        # Decoder RGB may have channels-last strides; cropping further retains
        # the parent row stride. Alternating bright/dark pixels expose errors
        # that a constant-color geometry test would hide.
        hwc = torch.stack((((xx + yy) % 2) * 255, (xx * 31 + yy * 17) % 256,
                           (xx >= width // 2) * 255), dim=-1).to(torch.uint8)
        frame = TensorFrame(hwc.permute(2, 0, 1))
        for mode in ("left", "middle", "right"):
            with self.subTest(mode=mode):
                cropped = InferenceRegion.from_frame(width, height, mode).crop(frame)
                self.assertFalse(cropped.rgb.is_contiguous())
                self.assert_sampled_bilinear_matches_reference(cropped.rgb, (61, 53))

    def test_sampled_geometry_cache_does_not_reuse_pixels_from_another_frame(self):
        for value in (0, 255, 19, 173):
            with self.subTest(value=value):
                rgb = torch.full((3, 133, 251), value, dtype=torch.uint8)
                output = _bilinear_sampled_resize(rgb, 31, 47).round().to(torch.uint8)
                self.assertTrue(torch.all(output == value))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU is unavailable")
    def test_cuda_sampled_downscale_matches_reference_and_reuses_pixel_cache(self):
        rgb = torch.randint(0, 256, (719, 1283, 3), dtype=torch.uint8,
                            device="cuda").permute(2, 0, 1)[:, :, 319:960]
        frame = TensorFrame(rgb)
        actual = frame.resized_rgb((149, 167), "bilinear")
        expected = F.interpolate(rgb[None].float(), size=(167, 149), mode="bilinear",
                                 align_corners=False)[0].round().clamp_(0, 255)
        self.assertLessEqual(int((actual.float() - expected).abs().max()), 1)
        self.assertIs(frame.resized_rgb((149, 167), "bilinear"), actual)
        self.assertEqual(frame.metrics["bilinear_sampled_calls"], 1)
        self.assertEqual(frame.metrics["resize_calls"], 1)
        self.assertEqual(frame.metrics["cache_hits"], 1)
        self.assertEqual(frame.metrics.get("cpu_materializations", 0), 0)
        # A smaller reduction and an upsample retain the native path.
        for size in ((600, 670), (750, 850)):
            with self.subTest(size=size):
                frame.resized_rgb(size, "bilinear")
                self.assertEqual(frame.metrics["bilinear_sampled_calls"], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU is unavailable")
    def test_cuda_sampled_4k_downscale_reduces_peak_allocated_memory(self):
        rgb = torch.zeros((3, 2160, 3840), dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        frame = TensorFrame(rgb)
        actual = frame.resized_rgb((640, 360), "bilinear")
        torch.cuda.synchronize()
        sampled_peak = torch.cuda.max_memory_allocated() - baseline
        # Keep the actual output alive and subtract its allocation, as well as
        # the coordinate cache, from the reference measurement's baseline.
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        reference = F.interpolate(rgb[None].float(), size=(360, 640), mode="bilinear",
                                  align_corners=False)[0].round().clamp_(0, 255).to(torch.uint8)
        torch.cuda.synchronize()
        native_peak = torch.cuda.max_memory_allocated() - baseline
        self.assertTrue(torch.equal(actual, reference))
        self.assertLess(sampled_peak, native_peak / 2)

    def test_rgb_to_bgr_preserves_bytes_and_materializes_once(self):
        rgb = torch.tensor([[[1, 2], [3, 4]], [[11, 12], [13, 14]],
                            [[21, 22], [23, 24]]], dtype=torch.uint8)
        metrics = {}
        frame = TensorFrame(rgb, metrics)
        expected = np.array([[[21, 11, 1], [22, 12, 2]],
                             [[23, 13, 3], [24, 14, 4]]], np.uint8)
        self.assertEqual(frame.shape, (2, 2, 3))
        actual = frame.bgr()
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(actual.flags.c_contiguous)
        self.assertIs(frame.bgr(), actual)
        self.assertEqual(metrics["cpu_materializations"], 1)
        self.assertEqual(metrics["cpu_materialized_bytes"], expected.nbytes)
        self.assertEqual(metrics.get("device_to_host_calls", 0), 0)
        self.assertEqual(metrics["cache_hits"], 1)

    def test_precrop_is_a_view_with_shared_accounting_and_correct_coordinates(self):
        bgr = np.arange(13 * 21 * 3, dtype=np.uint8).reshape(13, 21, 3)
        metrics = {}
        frame = frame_from_bgr(bgr, metrics)
        for mode in ("left", "middle", "right"):
            with self.subTest(mode=mode):
                region = InferenceRegion.from_frame(21, 13, mode)
                left, top, right, bottom = region.bounds
                cropped = region.crop(frame)
                self.assertEqual(cropped.shape, (bottom - top, right - left, 3))
                self.assertIs(cropped.metrics, metrics)
                self.assertEqual(cropped.rgb.untyped_storage().data_ptr(),
                                 frame.rgb.untyped_storage().data_ptr())
                np.testing.assert_array_equal(cropped.bgr(), bgr[top:bottom, left:right])
        self.assertEqual(metrics["cpu_materialized_bytes"], 3 * 13 * 10 * 3)
        self.assertNotIn("bgr", frame._cache)

    def test_resize_downloads_only_thumbnail_and_reuses_cached_pixels(self):
        bgr = np.full((48, 80, 3), (17, 101, 233), dtype=np.uint8)
        metrics = {}
        frame = frame_from_bgr(bgr, metrics)
        resized = frame.resized_bgr((20, 12))
        np.testing.assert_array_equal(resized, cv2.resize(bgr, (20, 12), interpolation=cv2.INTER_AREA))
        self.assertIs(frame.resized_bgr((20, 12)), resized)
        child = frame.resized((20, 12))
        self.assertEqual(child.shape, (12, 20, 3))
        self.assertIs(child.metrics, metrics)
        self.assertIs(child.rgb, frame.resized_rgb((20, 12)))
        self.assertEqual(metrics["resize_calls"], 1)
        self.assertEqual(metrics["cpu_materializations"], 1)
        self.assertEqual(metrics["cpu_materialized_bytes"], 20 * 12 * 3)
        self.assertLess(metrics["cpu_materialized_bytes"], bgr.nbytes)

    def test_same_size_rgb_resize_keeps_original_tensor_without_materialization(self):
        rgb = torch.zeros((3, 7, 11), dtype=torch.uint8)
        metrics = {}
        frame = TensorFrame(rgb, metrics)
        self.assertIs(frame.resized_rgb((11, 7)), rgb)
        self.assertEqual(metrics.get("resize_calls", 0), 0)
        self.assertEqual(metrics.get("cpu_materializations", 0), 0)

    def test_grayscale_integer_downsampling_matches_opencv_with_rounding_tolerance(self):
        bgr = np.random.default_rng(73).integers(0, 256, (72, 120, 3), dtype=np.uint8)
        size = (30, 18)
        for method in ("gray-area", "bgr-area"):
            with self.subTest(method=method):
                metrics = {}
                frame = frame_from_bgr(bgr, metrics)
                if method == "gray-area":
                    expected = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), size,
                                          interpolation=cv2.INTER_AREA)
                else:
                    expected = cv2.cvtColor(cv2.resize(bgr, size, interpolation=cv2.INTER_AREA),
                                             cv2.COLOR_BGR2GRAY)
                actual = frame.gray_thumbnail(size, method)
                self.assertEqual(actual.dtype, np.uint8)
                self.assertEqual(actual.shape, (size[1], size[0]))
                self.assertLessEqual(np.abs(actual.astype(np.int16) - expected).max(), 1)
                self.assertIs(frame.gray_thumbnail(size, method), actual)
                self.assertEqual(metrics["cpu_materializations"], 1)
                self.assertEqual(metrics["cpu_materialized_bytes"],
                                 size[0] * size[1] * (1 if method == "gray-area" else 3))

    def test_fractional_area_downsampling_preserves_opencv_thumbnail_values(self):
        # A regression to torch's adaptive-average "area" mode causes errors
        # exceeding 40/255 on this noise input because it uses different support.
        noise = np.random.default_rng(1).integers(0, 256, (216, 384, 3), dtype=np.uint8)
        yy, xx = np.mgrid[:71, :119]
        smooth = np.stack((xx * 2, yy * 3, xx + yy), axis=-1).astype(np.uint8)
        for bgr, size in ((noise, (160, 90)), (smooth, (17, 13))):
            with self.subTest(shape=bgr.shape, size=size):
                frame = frame_from_bgr(bgr)
                actual_bgr = frame.resized_bgr(size)
                expected_bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
                self.assertLessEqual(np.abs(actual_bgr.astype(np.int16) - expected_bgr).max(), 1)
                for method in ("gray-area", "bgr-area"):
                    if method == "gray-area":
                        expected = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), size,
                                              interpolation=cv2.INTER_AREA)
                    else:
                        expected = cv2.cvtColor(expected_bgr, cv2.COLOR_BGR2GRAY)
                    actual = frame.gray_thumbnail(size, method)
                    self.assertLessEqual(np.abs(actual.astype(np.int16) - expected).max(), 1)

    def test_letterbox_geometry_matches_ultralytics_for_wide_tall_and_odd_frames(self):
        shapes = ((37, 109), (109, 37), (71, 119), (33, 33))
        for height, width in shapes:
            for target in ((96, 96), (80, 96)):
                for auto in (False, True):
                    with self.subTest(shape=(height, width), target=target, auto=auto):
                        bgr = np.full((height, width, 3), (17, 101, 233), dtype=np.uint8)
                        expected = LetterBox(new_shape=target, auto=auto, stride=16)(image=bgr)
                        frame = frame_from_bgr(bgr)
                        tensor, metadata = frame.letterbox(size=target, auto=auto, stride=16)
                        actual = tensor[0].permute(1, 2, 0).numpy()[..., ::-1]
                        actual = np.rint(actual * 255).astype(np.uint8)
                        self.assertEqual(tensor.dtype, torch.float32)
                        self.assertEqual(tensor.shape[0:2], (1, 3))
                        np.testing.assert_array_equal(actual, expected)
                        # Infer content bounds from the independently generated
                        # reference image, rather than repeating the resize math.
                        yy, xx = np.nonzero(np.any(expected != 114, axis=2))
                        content_h, content_w = yy.max() - yy.min() + 1, xx.max() - xx.min() + 1
                        expected_padding = (int(xx.min()), int(yy.min()),
                                            int(expected.shape[1] - xx.max() - 1),
                                            int(expected.shape[0] - yy.max() - 1))
                        self.assertEqual(metadata["padding"], expected_padding)
                        self.assertEqual(metadata["resized_shape"], (content_h, content_w))
                        self.assertEqual(metadata["original_shape"], (height, width))
                        self.assertAlmostEqual(metadata["gain"], min(target[0] / height,
                                                                     target[1] / width))
                        self.assertEqual(frame.metrics.get("cpu_materializations", 0), 0)

    def test_letterbox_bilinear_pixels_match_opencv_with_rounding_tolerance(self):
        bgr = np.random.default_rng(18).integers(0, 256, (71, 119, 3), dtype=np.uint8)
        expected = LetterBox(new_shape=(80, 96), auto=False)(image=bgr)
        tensor, _ = frame_from_bgr(bgr).letterbox(size=(80, 96))
        actual = np.rint(tensor[0].permute(1, 2, 0).numpy()[..., ::-1] * 255).astype(np.int16)
        self.assertLessEqual(np.abs(actual - expected).max(), 1)

    def test_invalid_inputs_and_spatial_slices_are_rejected(self):
        for invalid in (np.zeros((3, 4, 5), np.uint8), torch.zeros((3, 4, 5))):
            with self.subTest(kind=type(invalid), dtype=invalid.dtype), self.assertRaises(TypeError):
                TensorFrame(invalid)
        for shape in ((4, 5), (1, 4, 5), (3, 0, 5), (3, 4, 0), (1, 3, 4, 5)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                TensorFrame(torch.zeros(shape, dtype=torch.uint8))
        frame = TensorFrame(torch.zeros((3, 4, 5), dtype=torch.uint8))
        for slices in (0, slice(None), (0, slice(None)), (slice(None), slice(None), slice(None)),
                       (slice(None, None, -1), slice(None)), (slice(None), slice(None, None, 2))):
            with self.subTest(slices=slices), self.assertRaises(TypeError):
                frame[slices]
        with self.assertRaises(ValueError):
            frame[0:0, :]
        for size in ((0, 4), (4, 0), (-1, 4)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                frame.resized_rgb(size)
            for method in ("gray-area", "bgr-area"):
                with self.subTest(size=size, method=method), self.assertRaises(ValueError):
                    frame.gray_thumbnail(size, method)
        with self.assertRaises(ValueError):
            frame.resized_rgb((2, 2), "nearest")
        with self.assertRaises(ValueError):
            frame.gray_thumbnail((2, 2), "unknown")
        for target, stride in (((0, 8), 32), ((8, -1), 32), ((8, 8), 0)):
            with self.subTest(target=target, stride=stride), self.assertRaises(ValueError):
                frame.letterbox(target, stride=stride)

    def test_letterbox_rejects_extreme_aspect_that_would_round_to_empty_pixels(self):
        for shape in ((1, 10000), (10000, 1)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                TensorFrame(torch.zeros((3, *shape), dtype=torch.uint8)).letterbox()


if __name__ == "__main__":
    unittest.main()
