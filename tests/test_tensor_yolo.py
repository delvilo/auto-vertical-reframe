"""Tensor YOLO geometry and native tracking; no automatic model downloads.

Set REFRAME_MODEL_DIR to a directory containing the two yolo26n checkpoints to
also exercise actual inference on Ultralytics' packaged bus image.
"""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import ASSETS

from reframe.config import AppConfig
from reframe.frame_pixels import TensorFrame
from reframe.perception import pose, segmentation
from reframe.perception.tensor_yolo import (
    TensorPosePredictor, TensorSegmentationPredictor, _LetterboxedTensorLoader,
    geometry_image, predictor_options,
)


def tensor_frame(height=80, width=160):
    pixels = torch.zeros((3, height, width), dtype=torch.uint8)
    pixels[0] = 240
    pixels[1] = 100
    pixels[2] = 10
    return TensorFrame(pixels)


def predictor(cls=TensorSegmentationPredictor, **overrides):
    instance = cls(overrides={"imgsz": 320, "rect": True, "verbose": False, "save": False, **overrides})
    instance.model = SimpleNamespace(format="pt", stride=32, names={0: "person"},
                                     kpt_shape=(17, 3))
    return instance


class TensorPredictorTests(unittest.TestCase):
    def test_geometry_placeholder_is_read_only_and_three_bytes(self):
        image = geometry_image((2160, 3840, 3))
        self.assertEqual(image.shape, (2160, 3840, 3))
        self.assertEqual(image.strides, (0, 0, 1))
        self.assertEqual(image.base.nbytes, 3)
        self.assertFalse(image.flags.writeable)

    def test_loader_does_not_scan_pixels_or_download(self):
        pixels = torch.zeros((2, 3, 320, 320), dtype=torch.float32)
        with patch.object(torch.Tensor, "max", side_effect=AssertionError("pixel scan")), \
                patch.object(torch.Tensor, "cpu", side_effect=AssertionError("download")):
            source = _LetterboxedTensorLoader(pixels)
            batches = list(source)
        self.assertEqual(len(batches), 1)
        self.assertIs(batches[0][1], pixels)
        self.assertTrue(source.source_type.tensor)

    def test_loader_rejects_broken_internal_shapes_and_dtypes(self):
        for shape, dtype in [((1, 3, 0, 320), torch.float32),
                             ((1, 3, 321, 320), torch.float32),
                             ((1, 4, 320, 320), torch.float32),
                             ((1, 3, 320, 320), torch.uint8)]:
            with self.subTest(shape=shape, dtype=dtype), self.assertRaises(ValueError):
                _LetterboxedTensorLoader(torch.zeros(shape, dtype=dtype))

    def test_source_keeps_rgb_and_original_geometry_without_materialization(self):
        frame = tensor_frame()
        instance = predictor()
        with patch.object(frame, "bgr", side_effect=AssertionError("full-frame copy")):
            instance.setup_source(frame)
        _, pixels, _ = next(iter(instance.dataset))
        self.assertEqual(tuple(pixels.shape), (1, 3, 160, 320))
        torch.testing.assert_close(pixels[0, :, 50, 50], torch.tensor([240, 100, 10]) / 255)
        self.assertEqual(instance._tensor_originals[0].shape, frame.shape)
        self.assertEqual(frame.metrics.get("cpu_materializations", 0), 0)

    def test_mixed_shape_pose_batch_uses_common_letterbox_size(self):
        instance = predictor(TensorPosePredictor)
        frames = [tensor_frame(80, 160), tensor_frame(160, 80)]
        instance.setup_source(frames)
        _, pixels, _ = next(iter(instance.dataset))
        self.assertEqual(tuple(pixels.shape), (2, 3, 320, 320))
        torch.testing.assert_close(pixels[0, :, 0, 0], torch.full((3,), 114 / 255))
        self.assertEqual([image.shape for image in instance._tensor_originals],
                         [frame.shape for frame in frames])

    def test_segmentation_boxes_and_masks_use_original_roi_coordinates(self):
        frame = tensor_frame()
        instance = predictor(rect=False)
        instance.setup_source(frame)
        _, pixels, _ = next(iter(instance.dataset))
        # Source box (20, 10, 140, 70): gain 2, vertical padding 80.
        prediction = torch.tensor([[40., 100., 280., 220., .9, 0., 1.]])
        for retina in (False, True):
            with self.subTest(retina=retina):
                instance.args.retina_masks = retina
                result = instance.construct_result(prediction.clone(), pixels,
                                                   instance._tensor_originals[0],
                                                   "roi", torch.ones((1, 80, 80)))
                np.testing.assert_allclose(result.boxes.xyxy.numpy(), [[20, 10, 140, 70]], atol=1e-5)
                self.assertEqual(result.orig_shape, (80, 160))
                stats = segmentation.mask_statistics(result.masks.data, [0], (80, 160))
                self.assertIn(0, stats)
                _, center_x, center_y, top_y = stats[0]
                self.assertAlmostEqual(center_x, 79.5, delta=1)
                self.assertAlmostEqual(center_y, 39.5, delta=1)
                self.assertAlmostEqual(top_y, 10, delta=1)

    def test_pose_points_are_scaled_to_original_roi(self):
        frame = tensor_frame()
        instance = predictor(TensorPosePredictor, rect=False)
        instance.setup_source(frame)
        _, pixels, _ = next(iter(instance.dataset))
        keypoints = torch.tensor([[160., 160., .9]] * 17)
        prediction = torch.cat((torch.tensor([40., 100., 280., 220., .9, 0.]),
                                keypoints.flatten()))[None]
        result = instance.construct_result(prediction, pixels, instance._tensor_originals[0], "roi")
        np.testing.assert_allclose(result.keypoints.xy.numpy(), [[[80, 40]] * 17], atol=1e-5)
        self.assertEqual(result.orig_shape, (80, 160))

    def test_botsort_is_rejected_before_receiving_placeholder_pixels(self):
        instance = predictor(mode="track", tracker="botsort.yaml")
        with self.assertRaisesRegex(ValueError, "ByteTrack only"):
            instance.setup_source(tensor_frame())

    def test_custom_tracker_and_multiple_tracking_frames_are_rejected(self):
        instance = predictor(mode="track", tracker="bytetrack.yaml")
        instance.trackers = [SimpleNamespace()]
        with self.assertRaisesRegex(ValueError, "unmodified native ByteTrack"):
            instance.setup_source(tensor_frame())
        with self.assertRaisesRegex(ValueError, "exactly one"):
            instance.setup_source([tensor_frame(), tensor_frame()])

    def test_visualization_non_pt_and_non_tensor_sources_are_rejected(self):
        instance = predictor()
        with self.assertRaises(TypeError):
            instance.setup_source(np.zeros((80, 160, 3), np.uint8))
        instance.model.format = "onnx"
        with self.assertRaisesRegex(ValueError, "PyTorch"):
            instance.setup_source(tensor_frame())
        instance.model.format = "pt"
        instance.args.save_crop = True
        with self.assertRaisesRegex(ValueError, "visualization/export"):
            instance.setup_source(tensor_frame())

    def test_existing_numpy_predictor_cannot_be_silently_reused(self):
        model = SimpleNamespace(predictor=SimpleNamespace())
        with self.assertRaisesRegex(ValueError, "start a new processing run"):
            predictor_options(model, TensorPosePredictor)
        model.predictor = predictor(TensorPosePredictor)
        self.assertIs(predictor_options(model, TensorPosePredictor)["predictor"], TensorPosePredictor)

    def test_tracker_thumbnail_uses_small_gray_copy_and_shared_cache(self):
        instance = object.__new__(segmentation.SegmentationTracker)
        instance.config = AppConfig(seg_thumbnail_method="gray-area")
        frame = tensor_frame(360, 640)
        first = instance._thumbnail(frame)
        second = instance._thumbnail(frame)
        self.assertIs(first, second)
        self.assertEqual(first.shape, (180, 320))
        self.assertEqual(frame.metrics["cpu_materialized_bytes"], 180 * 320)


MODEL_DIR = Path(os.environ["REFRAME_MODEL_DIR"]) if os.environ.get("REFRAME_MODEL_DIR") else None


@unittest.skipUnless(MODEL_DIR is not None, "set REFRAME_MODEL_DIR for existing local checkpoints")
class ExistingCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for task in ("seg", "pose"):
            if not (MODEL_DIR / f"yolo26n-{task}.pt").is_file():
                raise unittest.SkipTest("Both local checkpoints are required; downloads are disabled")
        cls.image = cv2.imread(str(ASSETS / "bus.jpg"))
        if cls.image is None:
            raise unittest.SkipTest("Ultralytics packaged bus image is unavailable")

    def frame(self):
        return TensorFrame(torch.from_numpy(self.image[:, :, ::-1].copy()).permute(2, 0, 1))

    def test_native_bytetrack_persists_ids_on_tensor_frames(self):
        model = YOLO(str(MODEL_DIR / "yolo26n-seg.pt"))
        frame = self.frame()
        kwargs = predictor_options(model, TensorSegmentationPredictor)
        outputs = [model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0],
                               device="cpu", verbose=False, **kwargs)[0] for _ in range(2)]
        self.assertGreater(len(outputs[0].boxes), 0)
        torch.testing.assert_close(outputs[0].boxes.id, outputs[1].boxes.id)
        self.assertEqual(model.predictor.trackers[0].frame_id, 2)
        self.assertEqual(outputs[0].orig_shape, frame.shape[:2])
        self.assertEqual(frame.metrics.get("cpu_materializations", 0), 0)

    def test_actual_tensor_and_numpy_model_coordinates_stay_close(self):
        # CPU evidence for this packaged image only, not a CUDA quality guarantee.
        for task, cls in (("seg", TensorSegmentationPredictor), ("pose", TensorPosePredictor)):
            with self.subTest(task=task):
                path = str(MODEL_DIR / f"yolo26n-{task}.pt")
                reference = YOLO(path).predict(self.image, device="cpu", verbose=False)[0]
                model = YOLO(path)
                actual = model.predict(self.frame(), device="cpu", verbose=False,
                                       **predictor_options(model, cls))[0]
                self.assertGreater(len(actual.boxes), 0)
                np.testing.assert_allclose(actual.boxes.xyxy, reference.boxes.xyxy, atol=3, rtol=0)
                np.testing.assert_allclose(actual.boxes.conf, reference.boxes.conf, atol=.01, rtol=0)
                if task == "pose":
                    np.testing.assert_allclose(actual.keypoints.xy, reference.keypoints.xy, atol=3, rtol=0)

    def test_real_pose_helper_batches_tensor_rois_in_source_coordinates(self):
        frame = self.frame()
        model = YOLO(str(MODEL_DIR / "yolo26n-seg.pt"))
        results = model.predict(frame, classes=[0], conf=.7, device="cpu", verbose=False,
                                **predictor_options(model, TensorSegmentationPredictor))[0]
        people = {index: tuple(row.tolist()) for index, row in enumerate(results.boxes.xyxy)}
        helper = pose.YOLOPoseHelper(AppConfig(pose_model=str(MODEL_DIR / "yolo26n-pose.pt"),
                                              yolo_device="cpu", pose_batch_size=2))
        observed = helper.detect_many(frame, [(index, box, index) for index, box in people.items()], people)
        self.assertGreaterEqual(sum(value is not None for value in observed.values()), 2)
        self.assertIsInstance(helper.model.predictor, TensorPosePredictor)
        for index, detection in observed.items():
            if detection is not None:
                self.assertEqual(detection.keypoints.shape, (17, 3))
                self.assertGreater(float(detection.keypoints[:, 1].mean()), people[index][1])
        self.assertEqual(frame.metrics.get("cpu_materializations", 0), 0)


if __name__ == "__main__":
    unittest.main()
