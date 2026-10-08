"""Inference regions must never change source geometry or invent unseen saliency."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from reframe.precrop import InferenceRegion
from reframe.contracts import FrameObservations, TrackObservation, PoseObservation, CameraState
from reframe.subjects import build_candidates, SubjectRankingModel
from reframe.saliency.regions import extract_saliency_region
from reframe.camera import build_global_saliency_observation


class PrecropTests(unittest.TestCase):
    def test_three_regions_and_default(self):
        for mode, bounds in [(None, (0, 0, 3840, 2160)), ("left", (0, 0, 1920, 2160)),
                             ("middle", (960, 0, 2880, 2160)), ("right", (1920, 0, 3840, 2160))]:
            with self.subTest(mode=mode):
                self.assertEqual(InferenceRegion.from_frame(3840, 2160, mode).bounds, bounds)
        self.assertEqual(InferenceRegion.from_frame(641, 360, "right").bounds, (321, 0, 641, 360))
        self.assertEqual(InferenceRegion.from_frame(641, 360, "middle").bounds, (160, 0, 480, 360))

    def test_crop_is_view_and_preserves_original(self):
        frame = np.arange(400 * 240 * 3, dtype=np.uint8).reshape(240, 400, 3)
        region = InferenceRegion.from_frame(400, 240, "right")
        actual = region.crop(frame)
        self.assertEqual(actual.shape, (240, 200, 3))
        np.testing.assert_array_equal(actual, frame[:, 200:])
        self.assertTrue(np.shares_memory(actual, frame))
        with self.assertRaises(ValueError):
            region.crop(np.zeros((100, 100, 3), np.uint8))

    def observations(self, region):
        context = region.context(8, .7, 1)
        points = np.tile([30., 50., .9], (17, 1))
        cues = dict(has_pose=True, head_box=(20., 10., 40., 30.), body_cx=30.,
                    body_min_x=10., body_max_x=90., eye_y=20., shoulder_y=40.,
                    body_top_y=10., body_bottom_y=200., shoulder_span=50.)
        pose = PoseObservation(points, points.copy(), cues, context,
                               replace(context, frame_index=5, timestamp=.4), "remapped", 4)
        track = TrackObservation(0, 0, .9, (10., 10., 90., 200.), 4, (1000., 45., 110., 12.))
        return FrameObservations(context, (track,), {0: pose})

    def test_box_pose_mask_and_inference_provenance_mapping(self):
        region = InferenceRegion.from_frame(400, 240, "right")
        original = self.observations(region)
        mapped = region.to_source(original)
        self.assertEqual(mapped.tracks[0].box, (210., 10., 290., 200.))
        self.assertEqual(mapped.tracks[0].mask_statistics, (1000., 245., 110., 12.))
        pose = mapped.poses[0]
        self.assertEqual(pose.frame.width, 400)
        self.assertEqual(pose.inferred_at.width, 400)
        self.assertEqual(pose.inferred_at.timestamp, .4)
        self.assertEqual(pose.source, "remapped")
        np.testing.assert_array_equal(pose.keypoints[:, 0], 230.)
        np.testing.assert_array_equal(pose.inferred_keypoints[:, 0], 230.)
        self.assertEqual(pose.cues["head_box"], (220., 10., 240., 30.))
        self.assertEqual(pose.cues["body_cx"], 230.)
        self.assertEqual(pose.cues["shoulder_span"], 50.)
        np.testing.assert_array_equal(original.poses[0].keypoints[:, 0], 30.)

    def test_map_stays_in_roi_and_unobserved_area_has_no_result(self):
        saliency = np.zeros((24, 20), np.float32)
        saliency[10:15, 8:12] = 1
        bounds = (200, 0, 400, 240)
        focus = extract_saliency_region(saliency, (0, 0, 400, 240),
                                        frame_shape=(240, 400), map_bounds=bounds)
        self.assertGreaterEqual(focus[2], 280)
        self.assertLessEqual(focus[4], 320)
        self.assertIsNone(extract_saliency_region(saliency, (0, 0, 100, 240), map_bounds=bounds))
        observation = build_global_saliency_observation(saliency, 135, 240, 400, 240, 1, 1.85,
                                                       saliency_bounds=bounds)
        self.assertGreater(observation.center_x, 270)

    def test_pose_only_candidate_does_not_gain_fake_saliency_evidence(self):
        region = InferenceRegion.from_frame(400, 240, "right")
        observations = region.to_source(self.observations(region))
        state = CameraState(200, 120, 1, 200, 120, 1)
        candidates = build_candidates(observations, None, SubjectRankingModel(), {0: "person"},
                                      state, 10., [], saliency_bounds=region.bounds)
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0].has_pose)
        self.assertEqual(candidates[0].saliency_confidence, 0.)
        self.assertIsNone(candidates[0].salient_x1)
        self.assertEqual(candidates[0].framing_cx, 230.)
        self.assertIsNone(build_global_saliency_observation(None, 135, 240, 400, 240, 1, 1.85))

    def test_pipeline_infers_half_frame_and_renders_full_source(self):
        from reframe import pipeline, cli, config
        args = config.apply_preset(cli.parse_args(["input.mp4", "/tmp/precrop-test.mp4",
                                                  "--precrop", "right"]))
        frame = np.zeros((240, 400, 3), np.uint8)
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.side_effect = {pipeline.cv2.CAP_PROP_FPS: 30,
                                   pipeline.cv2.CAP_PROP_FRAME_WIDTH: 400,
                                   pipeline.cv2.CAP_PROP_FRAME_HEIGHT: 240,
                                   pipeline.cv2.CAP_PROP_FRAME_COUNT: 1}.__getitem__
        tracker = MagicMock(class_names={0: "person"}, actual_device="cpu")
        tracker.track.return_value = (TrackObservation(0, 0, .9, (30, 20, 110, 220), 1),)
        pose = MagicMock()
        pose.helper.actual_device = None
        for attr in ("cache_hits", "primary_only_frames", "full_scan_frames", "rois_skipped"):
            setattr(pose, attr, 0)
        pose.helper.rois_inferred = pose.helper.rois_matched = 0
        service = MagicMock()
        service.process.return_value = SimpleNamespace(map=None)
        service.telemetry.return_value = {"active_backend": "pose", "tier": "pose_only"}
        writer = MagicMock(encoder="test")
        frames = ((1, frame) for _ in range(1))
        with patch.object(pipeline.cv2, "VideoCapture", return_value=capture), \
                patch.object(pipeline.shutil, "which", return_value="ffmpeg"), \
                patch.object(pipeline, "iter_video_frames", return_value=frames), \
                patch.object(pipeline, "SegmentationTracker", return_value=tracker), \
                patch.object(pipeline, "YOLOPoseHelper"), \
                patch.object(pipeline, "PoseCueCache", return_value=pose), \
                patch.object(pipeline, "observe_poses", side_effect=lambda f, t, h, s, c, k: FrameObservations(c, t)), \
                patch.object(pipeline, "build_saliency_helper", return_value=service), \
                patch.object(pipeline, "DirectVideoWriter", return_value=writer), \
                patch.object(pipeline.InlineSceneDetector, "update", return_value=True), \
                patch.object(pipeline, "build_candidates", wraps=build_candidates) as candidates, \
                patch.object(pipeline, "crop_frame", wraps=pipeline.crop_frame) as crop:
            pipeline.process_video(args)
        self.assertEqual(tracker.track.call_args.args[0].shape, (240, 200, 3))
        self.assertEqual(service.process.call_args.args[0].shape, (240, 200, 3))
        mapped = candidates.call_args.kwargs["observations"]
        self.assertEqual(mapped.tracks[0].box, (230, 20, 310, 220))
        self.assertIs(crop.call_args.args[0], frame)
        self.assertGreater(crop.call_args.args[1], 200)
        writer.write.assert_called_once()


if __name__ == "__main__":
    unittest.main()
