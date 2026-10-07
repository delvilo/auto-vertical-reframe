"""Pose geometry, identity matching and failure propagation without model downloads."""
import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

for module in ("cv2", "scenedetect", "ultralytics"):
    try:
        __import__(module)
    except ImportError:
        sys.modules[module] = MagicMock()

import numpy as np
from reframe.perception import pose as app, segmentation
from reframe import cli, config, contracts, subjects, pipeline, scenes
from reframe.contracts import FrameContext, PoseDetection


def arguments(*extra):
    with patch.object(sys, "argv", ["auto_reframe.py", "in.mp4", "out.mp4", *extra]):
        args = cli.parse_args()
        args.yolo_device = "cpu"
        return args


def skeleton():
    # Person box is (100, 20, 200, 220), feet end well above the box bottom.
    points = np.array([(150, 45), (143, 40), (157, 40), (130, 44), (170, 44),
                       (120, 75), (180, 75), (110, 110), (190, 110),
                       (105, 145), (195, 145), (130, 130), (170, 130),
                       (130, 165), (170, 165), (130, 200), (170, 200)], dtype=float)
    return np.column_stack((points, np.ones(17) * 0.9))


def fake_tensor(data):
    tensor = MagicMock()
    tensor.detach.return_value.cpu.return_value.numpy.return_value = np.asarray(data)
    return tensor


def result(boxes, points):
    bounding = MagicMock()
    bounding.__len__.return_value = len(boxes)
    bounding.xyxy = fake_tensor(boxes)
    return SimpleNamespace(boxes=bounding, keypoints=SimpleNamespace(data=fake_tensor(points)))


def model():
    return SimpleNamespace(task="pose", names={0: "person"},
                           model=SimpleNamespace(model=[SimpleNamespace(kpt_shape=[17, 3])]),
                           predictor=SimpleNamespace(device="cpu"), predict=MagicMock())


class GeometryTests(unittest.TestCase):
    def test_head_and_body_geometry_keep_box_bottom(self):
        cues = app.pose_cues(skeleton(), (100, 20, 200, 220), (240, 400), .35)
        self.assertTrue(cues["has_pose"])
        self.assertEqual(cues["eye_y"], 40)
        self.assertEqual(cues["shoulder_span"], 60)
        self.assertEqual(cues["body_bottom_y"], 220)
        self.assertTrue(cues["body_bottom_confident"])
        self.assertLess(cues["head_box"][1], 40)
        self.assertLessEqual(cues["head_box"][2], 200)

    def test_visible_knees_do_not_imply_visible_feet(self):
        points = skeleton()
        points[15:, 2] = .1
        cues = app.pose_cues(points, (100, 20, 200, 220), (240, 400), .35)
        self.assertFalse(cues["body_bottom_confident"])
        self.assertEqual(cues["body_bottom_y"], 220)

    def test_occluded_head_keeps_body_without_fabricating_face(self):
        points = skeleton()
        points[:5, 2] = 0
        cues = app.pose_cues(points, (100, 20, 200, 220), (240, 400), .35)
        self.assertIsNone(cues["head_box"])
        self.assertIsNone(cues["eye_y"])
        self.assertTrue(cues["has_pose"])

    def test_head_only_does_not_count_as_body_pose(self):
        points = skeleton()
        points[5:, 2] = 0
        cues = app.pose_cues(points, (100, 20, 200, 220), (240, 400), .35)
        self.assertFalse(cues["has_pose"])
        self.assertIsNone(cues["body_bottom_y"])
        self.assertIsNotNone(cues["head_box"])

    def test_invalid_low_confidence_and_outside_points_are_ignored(self):
        points = skeleton()
        points[:, 2] = .1
        points[0] = (float("nan"), 20, .9)
        points[1] = (350, 200, .9)
        self.assertIsNone(app.pose_cues(points, (100, 20, 200, 220), (240, 400), .35))

    def test_wrong_keypoint_layout_rejected(self):
        with self.assertRaises(ValueError):
            app.pose_cues(np.zeros((33, 3)), (100, 20, 200, 220), (240, 400), .35)

    def test_identity_matching_ignores_detection_order_and_rejects_ambiguity(self):
        people = {7: (10, 10, 60, 210), 2: (150, 10, 200, 210)}
        self.assertEqual(app.pose_owner((148, 12, 202, 212), people), 2)
        self.assertIsNone(app.pose_owner((300, 10, 360, 210), people))
        self.assertIsNone(app.pose_owner((10, 10, 60, 210), {7: people[7], 2: people[7]}))


class PredictorTests(unittest.TestCase):
    def helper(self, fake, *extra):
        with patch.object(app, "YOLO", return_value=fake):
            return app.YOLOPoseHelper(arguments(*extra))

    def test_bgr_roi_offset_and_shared_head_body_inference(self):
        fake = model()
        # 12% padding -> ROI origin (88, 0). Ultralytics already returns ROI pixels.
        local_points = skeleton()
        local_points[:, 0] -= 88
        fake.predict.return_value = [result([[12, 20, 112, 220]], [local_points])]
        helper = self.helper(fake)
        frame = np.zeros((240, 400, 3), np.uint8)
        frame[:] = [5, 10, 250]
        cues = helper.detect_many(frame, [(3, (100, 20, 200, 220), 91)],
                                  {3: (100, 20, 200, 220)})[3]
        fake.predict.assert_called_once()
        np.testing.assert_array_equal(fake.predict.call_args.kwargs["source"][0][0, 0], [5, 10, 250])
        self.assertEqual(cues.cues["body_cx"], 150)
        np.testing.assert_array_equal(cues.keypoints, skeleton())
        self.assertEqual(cues.cues["eye_y"], 40)
        self.assertEqual(helper.actual_device, "cpu")

    def test_neighbor_pose_cannot_be_assigned_to_requested_track(self):
        fake = model()
        fake.predict.return_value = [result([[22, 20, 122, 220]], [skeleton()])]
        helper = self.helper(fake)
        people = {3: (100, 20, 200, 220), 4: (110, 20, 210, 220)}
        cues = helper.detect_many(np.zeros((240, 400, 3), np.uint8), [(3, people[3], 91)], people)
        self.assertIsNone(cues[3])

    def test_batch_size_bounds_inference_and_empty_detections(self):
        fake = model()
        fake.predict.side_effect = lambda **kw: [result([], []) for _ in kw["source"]]
        helper = self.helper(fake, "--pose-batch-size", "2")
        requests = [(i, (i * 70 + 1, 10, i * 70 + 51, 210), i) for i in range(5)]
        people = {index: box for index, box, _ in requests}
        out = helper.detect_many(np.zeros((240, 400, 3), np.uint8), requests, people)
        self.assertEqual(len(out), 5)
        self.assertTrue(all(value is None for value in out.values()))
        self.assertEqual([len(c.kwargs["source"]) for c in fake.predict.call_args_list], [2, 2, 1])

    def test_inference_error_logs_traceback_and_propagates(self):
        fake = model()
        fake.predict.side_effect = RuntimeError("CUDA failure")
        helper = self.helper(fake)
        with self.assertLogs(level="ERROR") as logs, self.assertRaisesRegex(RuntimeError, "CUDA failure"):
            helper.detect_many(np.zeros((240, 400, 3), np.uint8),
                               [(0, (100, 20, 200, 220), 91)], {0: (100, 20, 200, 220)})
        self.assertIn("Traceback", "\n".join(logs.output))
        self.assertIn("tracks=[91]", "\n".join(logs.output))

    def test_gpu_fallback_is_rejected_for_pose_too(self):
        fake = model()
        fake.predict.return_value = [result([], [])]
        helper = self.helper(fake)
        helper.device = "cuda:0"
        with self.assertLogs(level="ERROR"), self.assertRaisesRegex(RuntimeError, "device mismatch"):
            helper.detect_many(np.zeros((240, 400, 3), np.uint8),
                               [(0, (100, 20, 200, 220), 91)], {0: (100, 20, 200, 220)})

    def test_wrong_model_task_layout_and_missing_file_stop_initialization(self):
        fake = model()
        fake.task = "segment"
        with self.assertRaisesRegex(ValueError, "Expected pose"):
            self.helper(fake)
        fake.task = "pose"
        fake.model.model[-1].kpt_shape = [21, 3]
        with self.assertRaisesRegex(ValueError, "COCO-17"):
            self.helper(fake)
        with self.assertRaises(FileNotFoundError):
            self.helper(model(), "--pose-model", "/missing/custom-pose.pt")

    def test_missing_keypoints_is_error_not_empty_detection(self):
        fake = model()
        prediction = result([[12, 20, 112, 220]], [])
        prediction.keypoints = None
        fake.predict.return_value = [prediction]
        helper = self.helper(fake)
        with self.assertLogs(level="ERROR"), self.assertRaisesRegex(ValueError, "without keypoints"):
            helper.detect_many(np.zeros((240, 400, 3), np.uint8),
                               [(0, (100, 20, 200, 220), 91)], {0: (100, 20, 200, 220)})


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.helper = MagicMock()
        self.cues = app.pose_cues(skeleton(), (100, 20, 200, 220), (240, 400), .35)
        self.helper.detect_many.side_effect = lambda frame, req, people: {i: PoseDetection(skeleton(), self.cues) if self.cues else None for i, _, _ in req}
        self.cache = app.PoseCueCache(self.helper, .2, 60)
        self.frame = np.zeros((300, 500, 3), np.uint8)

    def get(self, frame_idx, box=(100, 20, 200, 220), track=91, others=None):
        return self.cache.get_many(self.frame, [(0, box, track)], {0: box, **(others or {})}, FrameContext(frame_idx, (frame_idx - 1) / 60, 500, 300, 1))[0]

    def test_coordinates_head_box_and_boolean_survive_cache_remap(self):
        first = self.get(1)
        moved = self.get(2, (110, 25, 220, 245))
        self.helper.detect_many.assert_called_once()
        self.assertAlmostEqual(moved.cues["eye_y"], 47)
        self.assertAlmostEqual(moved.cues["body_cx"], 165)
        self.assertAlmostEqual(moved.cues["shoulder_span"], 66)
        self.assertAlmostEqual(moved.cues["head_box"][0], 110 + (first.cues["head_box"][0] - 100) * 1.1)
        self.assertIs(moved.cues["has_pose"], True)

    def test_empty_pose_is_cached_until_expiration(self):
        self.cues = None
        self.assertIsNone(self.get(1).cues)
        self.assertIsNone(self.get(2).cues)
        self.helper.detect_many.assert_called_once()
        self.get(13)
        self.assertEqual(self.helper.detect_many.call_count, 2)

    def test_cut_motion_new_track_and_overlap_invalidate_cache(self):
        self.get(1)
        self.get(2, track=92)
        self.get(3, box=(200, 20, 300, 220), track=92)
        self.cache.clear()
        self.get(4)
        self.get(5, others={1: (110, 20, 210, 220)})
        self.assertEqual(self.helper.detect_many.call_count, 5)

    def test_interval_zero_refreshes_every_frame(self):
        self.cache = app.PoseCueCache(self.helper, 0, 60)
        self.get(1)
        self.get(2)
        self.assertEqual(self.helper.detect_many.call_count, 2)

    def test_cached_points_keep_confidence_and_true_inference_time(self):
        first = self.get(1)
        second = self.get(2, (110, 25, 220, 245))
        third = self.get(3, (110, 25, 220, 245))
        self.assertEqual(second.keypoints.shape, (17, 3))
        np.testing.assert_array_equal(second.keypoints[:, 2], first.keypoints[:, 2])
        np.testing.assert_array_equal(second.inferred_keypoints, skeleton())
        np.testing.assert_allclose(second.keypoints[:, 0], 110 + (skeleton()[:, 0] - 100) * 1.1)
        np.testing.assert_allclose(third.keypoints, second.keypoints)
        self.assertEqual(third.inferred_at.frame_index, 1)
        self.assertEqual(third.inferred_at.timestamp, 0)
        self.assertEqual(third.frame.frame_index, 3)
        self.assertEqual(third.source, "remapped")
        self.assertEqual(third.track_id, 91)
        self.assertEqual(self.get(13).inferred_at.frame_index, 13)

    def test_cached_missing_pose_keeps_original_miss_time(self):
        self.cues = None
        first, second = self.get(1), self.get(2)
        self.assertIsNone(second.keypoints)
        self.assertIsNone(second.cues)
        self.assertEqual(second.source, "remapped")
        self.assertEqual(second.inferred_at, first.inferred_at)


class PipelineTests(unittest.TestCase):
    def build(self, rows, helper, top_k=0):
        boxes = MagicMock()
        boxes.data = np.asarray(rows, dtype=float)
        boxes.__len__.return_value = len(rows)
        state = contracts.CameraState(crop_center_x=200, crop_center_y=120, zoom=1,
                                target_center_x=200, target_center_y=120, target_zoom=1,
                                tracked_id=91)
        frame = np.zeros((240, 400, 3), np.uint8)
        tracks = segmentation.parse_tracks(SimpleNamespace(boxes=boxes, masks=None), frame.shape[:2], [0, 16])
        observations = app.observe_poses(frame, tracks, helper, state, FrameContext(1, 0, 400, 240, 1), top_k)
        with patch.object(subjects, "extract_saliency_region", return_value=None):
            return subjects.build_candidates(observations, np.zeros((240, 400)),
                                             subjects.SubjectRankingModel(), {0: "person", 16: "dog"},
                                             state, 60, [])

    def test_top_k_keeps_tracked_subject_and_matching_sees_all_people(self):
        helper = MagicMock()
        helper.get_many.return_value = {}
        rows = [[100, 20, 200, 220, 91, .7, 0], [220, 10, 395, 230, 92, .99, 0]]
        candidates = self.build(rows, helper, top_k=1)
        request = helper.get_many.call_args.args[1]
        self.assertEqual([r[2] for r in request], [91])
        self.assertEqual(len(helper.get_many.call_args.args[2]), 2)
        self.assertEqual(len(candidates), 2)

    def test_fatal_pose_error_cannot_be_swallowed_by_candidate_loop(self):
        helper = MagicMock()
        helper.get_many.side_effect = RuntimeError("pose failure")
        with self.assertRaisesRegex(RuntimeError, "pose failure"):
            self.build([[100, 20, 200, 220, 91, .9, 0]], helper)

    def test_missing_pose_keeps_person_and_non_person_candidates(self):
        candidates = self.build([[100, 20, 200, 220, 91, .9, 0], [220, 30, 300, 180, 92, .8, 16]], None)
        self.assertEqual({c.cls_name for c in candidates}, {"person", "dog"})
        self.assertTrue(all(c.head_box is None and not c.has_pose for c in candidates))

    def test_cli_rejects_old_models_and_invalid_pose_settings(self):
        for extra in (("--face-model", "old.tflite"), ("--pose-model", "old.task"),
                      ("--pose-conf", "nan"), ("--keypoint-conf", "0"),
                      ("--pose-imgsz", "0"), ("--pose-imgsz", "641"), ("--pose-batch-size", "0")):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(*extra)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires FFmpeg")
class FullPipelineTests(unittest.TestCase):
    def test_three_frame_pipeline_emits_pose_summary_and_real_video(self):
        """Fake model outputs exercise the real camera loop, writer and final summary."""
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "in.mp4", Path(directory) / "out.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                            "color=black:s=400x240:r=30", "-frames:v", "3", str(source)],
                           check=True, capture_output=True)
            args = arguments("--device", "cpu", "--video-encoder", "libx264",
                             "--output-width", "128", "--output-height", "192", "--max-frames", "3")
            args.input, args.output = str(source), str(output)
            args = config.apply_preset(args)
            capture = MagicMock()
            capture.isOpened.return_value = True
            props = {pipeline.cv2.CAP_PROP_FPS: 30, pipeline.cv2.CAP_PROP_FRAME_WIDTH: 400,
                     pipeline.cv2.CAP_PROP_FRAME_HEIGHT: 240, pipeline.cv2.CAP_PROP_FRAME_COUNT: 3}
            capture.get.side_effect = lambda prop: props[prop]
            boxes = MagicMock()
            boxes.__len__.return_value = 1
            boxes.data = np.array([[100, 20, 200, 220, 91, .9, 0]])
            segment = SimpleNamespace(task="segment", names={0: "person"},
                                      predictor=SimpleNamespace(device="cpu", trackers=[]),
                                      track=MagicMock(return_value=[SimpleNamespace(boxes=boxes, masks=None)]))
            pose = model()
            points = skeleton()
            points[:, 0] -= 88
            pose.predict.return_value = [result([[12, 20, 112, 220]], [points])]
            saliency = MagicMock()
            saliency.process.return_value = SimpleNamespace(map=np.zeros((240, 400)))
            saliency.telemetry.return_value = {"active_backend": "handcrafted"}
            frames = ((i, np.zeros((240, 400, 3), np.uint8)) for i in range(1, 4))
            with patch.object(pipeline.cv2, "VideoCapture", return_value=capture), \
                    patch.object(pipeline.cv2, "resize", side_effect=lambda image, size, **kw: np.zeros((size[1], size[0], 3), np.uint8)), \
                    patch.object(pipeline, "iter_video_frames", return_value=frames), \
                    patch.object(segmentation, "YOLO", return_value=segment), \
                    patch.object(app, "YOLO", return_value=pose), \
                    patch.object(pipeline, "build_saliency_helper", return_value=saliency), \
                    patch.object(subjects, "extract_saliency_region", return_value=None), \
                    patch.object(scenes.InlineSceneDetector, "update", side_effect=[True, False, False]), \
                    self.assertLogs(level="INFO") as logs:
                pipeline.process_video(args)
            summary = json.loads(next(line.split("Summary: ", 1)[1]
                                      for line in logs.output if "Summary: " in line))
            self.assertEqual(summary["frames_processed"], 3)
            self.assertEqual(summary["frames_with_pose"], 3)
            self.assertEqual(summary["frames_with_head_cues"], 3)
            self.assertEqual(summary["pose_device"], "cpu")
            self.assertEqual(summary["pose_rois_inferred"], 1)
            self.assertEqual(summary["pose_cache_hits"], 2)
            probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                    "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(output)],
                                   check=True, capture_output=True, text=True)
            self.assertEqual(probe.stdout.strip(), "3")


if __name__ == "__main__":
    unittest.main()
