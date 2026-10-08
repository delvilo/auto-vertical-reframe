"""Detector skipping against real ByteTrack clocks and actual sparse optical flow."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from ultralytics.engine.results import Boxes, Results
from ultralytics.trackers.byte_tracker import BYTETracker

from reframe.config import AppConfig
from reframe.contracts import FrameContext, FrameObservations, PoseObservation
from reframe.perception import segmentation
from reframe.perception.motion import SparseMotion


class SyntheticYOLO:
    """Use real matching/Kalman filtering while bypassing network inference."""
    task = "segment"
    names = {0: "person"}

    def __init__(self):
        args = SimpleNamespace(track_high_thresh=.25, track_low_thresh=.1,
                               new_track_thresh=.25, track_buffer=30,
                               match_thresh=.8, fuse_score=True)
        self.tracker = BYTETracker(args)
        self.tracker.update = Mock(wraps=self.tracker.update)
        self.predictor = SimpleNamespace(trackers=[self.tracker], device="cpu")
        self.calls = []
        self.index = 0
        self.offset = 0.

    def track(self, source, **kwargs):
        self.calls.append(self.index)
        height, width = source.shape[:2]
        data = np.array([[.28 * width + self.offset, .17 * height,
                          .69 * width + self.offset, .83 * height, .95, 0]], np.float32)
        tracked = self.tracker.update(Boxes(data, source.shape[:2]), img=source)
        boxes = tracked[:, :-1] if len(tracked) else np.empty((0, 6), np.float32)
        masks = np.zeros((len(boxes), height, width), np.float32)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box[:4].astype(int)
            masks[i, y1:y2, x1:x2] = 1
        return [Results(source, path="synthetic", names=self.names, boxes=boxes, masks=masks)]


class AdaptiveSegmentationTests(unittest.TestCase):
    def setUp(self):
        self.fake = SyntheticYOLO()
        self.config = AppConfig(seg_max_gap=3, seg_max_age=.1, runtime_fps=60., yolo_device="cpu")
        self.enterContext(patch.object(segmentation, "YOLO", return_value=self.fake))
        self.enterContext(patch.object(segmentation, "verify_yolo_device", return_value="cpu"))
        self.tracker = segmentation.SegmentationTracker(self.config, [0])
        self.frame = np.full((180, 320, 3), 20, np.uint8)
        rng = np.random.default_rng(32)
        texture = rng.integers(20, 240, (120, 130), dtype=np.uint8)
        self.frame[30:150, 90:220] = texture[:, :, None]
        self.index = 0
        self.scene = 0
        self.timestamp = 0.
        self.history = []

    def step(self, frame=None, timestamp=None):
        frame = self.frame if frame is None else frame
        self.index += 1
        self.timestamp = (self.index - 1) / self.config.runtime_fps if timestamp is None else timestamp
        context = FrameContext(self.index, self.timestamp, frame.shape[1], frame.shape[0], self.scene)
        self.fake.index = self.index
        tracks = self.tracker.track(frame, context)
        self.assertEqual(len(tracks), 1)
        track = tracks[0]
        x1, y1, x2, y2 = track.box
        points = np.column_stack((np.linspace(x1 + 10, x2 - 10, 17),
                                  np.linspace(y1 + 10, y2 - 10, 17), np.full(17, .95)))
        cues = dict(has_pose=True, body_cx=(x1 + x2) / 2, body_top_y=y1,
                    body_bottom_y=y2, body_min_x=x1, body_max_x=x2,
                    shoulder_span=.5 * (x2 - x1))
        pose = PoseObservation(points, points.copy(), cues, context, context, "inferred", track.track_id)
        self.tracker.feedback(FrameObservations(context, tracks, {track.row_index: pose}), track.track_id)
        self.history.append((context, track))
        return track

    def warm(self, count=60):
        for _ in range(count):
            self.step()

    def test_stable_frames_progress_through_two_and_three_frame_intervals(self):
        self.warm()
        self.assertEqual(self.fake.calls[:6], list(range(1, 7)))
        gaps = np.diff(self.fake.calls)
        self.assertIn(2, gaps)
        self.assertIn(3, gaps)
        self.assertLess(len(self.fake.calls), 40)
        self.assertTrue(np.all(gaps <= 3))
        self.assertEqual(len({track.track_id for _, track in self.history}), 1)
        self.assertTrue(any(track.source == "predicted" for _, track in self.history))

    def test_skip_advances_real_tracker_without_empty_detection_update(self):
        last_measured = None
        for _ in range(45):
            before = self.fake.tracker.update.call_count
            track = self.step()
            context = self.history[-1][0]
            self.assertEqual(self.fake.tracker.frame_id, self.index)
            if track.source == "detected":
                self.assertEqual(self.fake.tracker.update.call_count, before + 1)
                self.assertEqual(track.measured_at, context)
                last_measured = context
            else:
                self.assertEqual(self.fake.tracker.update.call_count, before)
                self.assertEqual(track.measured_at, last_measured)
                self.assertLessEqual(context.timestamp - track.measured_at.timestamp, .1 + 1e-9)
                self.assertTrue(track.reliable_at(context, self.config.seg_max_age))
        self.assertEqual(self.fake.tracker.update.call_count, len(self.fake.calls))
        self.assertTrue(all(len(call.args[0]) == 1 for call in self.fake.tracker.update.call_args_list))

    def test_age_limit_uses_video_timestamp(self):
        self.warm()
        before = len(self.fake.calls)
        refreshed = self.step(timestamp=self.timestamp + .101)
        self.assertEqual(len(self.fake.calls), before + 1)
        self.assertEqual(refreshed.source, "detected")
        self.assertEqual(refreshed.measured_at.timestamp, self.timestamp)

    def test_disabled_skipping_is_every_frame_baseline(self):
        self.config.seg_max_gap = 1
        self.warm(30)
        self.assertEqual(self.fake.calls, list(range(1, 31)))
        self.assertTrue(all(track.source == "detected" for _, track in self.history))

    def test_scene_change_resets_tracking_and_restarts_dense_detection(self):
        self.warm(35)
        self.scene += 1
        before = len(self.fake.calls)
        for _ in range(3):
            self.assertEqual(self.step().source, "detected")
        self.assertEqual(len(self.fake.calls), before + 3)
        self.assertEqual(self.fake.tracker.frame_id, 3)
        self.assertEqual(self.history[-1][1].measured_at.scene_index, self.scene)

    def test_geometry_change_restarts_tracking(self):
        self.warm(35)
        enlarged = cv2.resize(self.frame, (640, 360), interpolation=cv2.INTER_NEAREST)
        before = len(self.fake.calls)
        track = self.step(enlarged)
        self.assertEqual(len(self.fake.calls), before + 1)
        self.assertEqual(track.source, "detected")
        self.assertEqual(track.measured_at.width, 640)
        self.assertEqual(self.fake.tracker.frame_id, 1)

    def test_global_change_refreshes_immediately(self):
        self.warm(35)
        before = len(self.fake.calls)
        changed = np.full_like(self.frame, 240)
        track = self.step(changed)
        self.assertEqual(len(self.fake.calls), before + 1)
        self.assertEqual(track.source, "detected")

    def test_flow_rejection_refreshes_instead_of_publishing_prediction(self):
        self.warm(35)
        # The next frame after a mature detection would ordinarily be predicted.
        while self.history[-1][1].source != "detected":
            self.step()
        before = len(self.fake.calls)
        with patch.object(SparseMotion, "estimate", return_value=({}, "flow_unreliable")):
            track = self.step()
        self.assertEqual(len(self.fake.calls), before + 1)
        self.assertEqual(track.source, "detected")

    def test_slow_translation_keeps_id_and_corrects_intermediate_boxes(self):
        self.frame = cv2.GaussianBlur(self.frame, (5, 5), 1.2)
        self.warm(35)
        origin = self.history[-1][1].box[0]
        predicted = 0
        for offset in range(1, 13):
            previous = self.history[-1][1]
            self.fake.offset = float(offset)
            moved = cv2.warpAffine(self.frame, np.float32([[1, 0, offset], [0, 1, 0]]),
                                   (320, 180), borderValue=(20, 20, 20))
            track = self.step(moved)
            predicted += track.source == "predicted"
            if track.source == "predicted":
                self.assertEqual(track.mask_statistics[0], previous.mask_statistics[0])
                self.assertAlmostEqual(track.mask_statistics[1] - previous.mask_statistics[1],
                                       track.box[0] - previous.box[0])
                self.assertEqual(track.measured_at, previous.measured_at)
            self.assertAlmostEqual(track.box[0], origin + offset, delta=2.)
            self.assertEqual(track.track_id, self.history[0][1].track_id)
            self.assertEqual(self.fake.tracker.frame_id, self.index)
        self.assertGreater(predicted, 0)

    def test_lost_track_prevents_skip_and_ages_on_source_frames(self):
        self.warm(35)
        native = self.fake.tracker
        lost = native.tracked_stracks.pop()
        lost.mark_lost()
        native.lost_stracks.append(lost)
        before = len(self.fake.calls)
        track = self.step()
        self.assertEqual(track.source, "detected")
        self.assertEqual(track.track_id, self.history[0][1].track_id)
        self.assertEqual(len(self.fake.calls), before + 1)
        self.assertEqual(native.frame_id, self.index)

    def test_tracker_subclass_uses_every_frame_fallback(self):
        class UnsupportedTracker(BYTETracker):
            pass
        native = UnsupportedTracker(self.fake.tracker.args)
        native.update = Mock(wraps=native.update)
        self.fake.tracker = native
        self.fake.predictor.trackers = [native]
        self.warm(30)
        self.assertEqual(len(self.fake.calls), 30)
        self.assertEqual(self.tracker.telemetry()["refresh_reasons"]["unsupported_tracker"], 29)

    def test_weak_fresh_pose_forces_refresh_without_advancing_detection_age(self):
        self.warm(35)
        while self.history[-1][1].source != "detected":
            self.step()
        context, track = self.history[-1]
        pose = PoseObservation(None, None, {"has_pose": False}, context, context, "inferred", track.track_id)
        self.tracker.feedback(FrameObservations(context, (track,), {0: pose}), track.track_id)
        before = len(self.fake.calls)
        self.assertEqual(self.step().source, "detected")
        self.assertEqual(len(self.fake.calls), before + 1)
        self.assertEqual(self.tracker.telemetry()["refresh_reasons"]["weak_primary_pose"], 1)


if __name__ == "__main__":
    unittest.main()
