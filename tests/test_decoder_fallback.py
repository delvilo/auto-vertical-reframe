"""Decoder selection must not lose frames or disguise runtime failures."""
from contextlib import ExitStack
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from reframe.cli import parse_args
from reframe.frame_pixels import TensorFrame
from reframe.pipeline import _open_video_input


class FailingReader:
    fps, width, height, total_frames = 30., 64, 64, 7

    def __init__(self, failure, fail_at=1):
        self.failure = failure
        self.fail_at = fail_at
        self.index = 0
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        if self.index == self.fail_at:
            raise self.failure
        return self.index, TensorFrame(torch.zeros((3, 64, 64), dtype=torch.uint8))

    def close(self):
        self.closed = True


class DecoderFallbackTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(TemporaryDirectory())
        self.source = Path(self.directory) / 'source.mp4'
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*'mp4v'), 30, (64, 64))
        self.assertTrue(writer.isOpened())
        try:
            for index in range(7):
                writer.write(np.full((64, 64, 3), (20 + index * 20, 70, 180), np.uint8))
        finally:
            writer.release()
        self.args = parse_args([str(self.source), str(Path(self.directory) / 'out.mp4')])

    def assert_complete_opencv(self, frames):
        colors = []
        indices = []
        for index, frame in frames:
            indices.append(index)
            colors.append(frame[32, 32].tolist())
        self.assertEqual(indices, list(range(1, 8)))
        for index, color in enumerate(colors):
            self.assertTrue(all(abs(actual - expected) < 8 for actual, expected in
                                zip(color, (20 + index * 20, 70, 180))))

    def test_unavailable_torchcodec_restarts_with_complete_opencv_video(self):
        with patch('reframe.torchcodec_reader.TorchCodecReader', side_effect=ImportError('missing FFmpeg library')), \
                self.assertLogs(level='WARNING') as logs, ExitStack() as resources:
            reader, frames, metadata, reason = _open_video_input(self.args, self.source, resources)
            self.assertIsNone(reader)
            self.assertEqual(metadata, (30., 64, 64, 7))
            self.assertIn('missing FFmpeg library', reason)
            self.assert_complete_opencv(frames)
        self.assertIn('Traceback', '\n'.join(logs.output))
        self.assertIsNone(frames.gi_frame)

    def test_failed_first_batch_closes_reader_and_restarts_at_first_frame(self):
        for error in (RuntimeError('unverified CUDA hardware'), AssertionError('CUDA unavailable')):
            with self.subTest(error=type(error).__name__):
                failed = FailingReader(error)
                with patch('reframe.torchcodec_reader.TorchCodecReader', return_value=failed), \
                        self.assertLogs(level='WARNING'), ExitStack() as resources:
                    reader, frames, _, reason = _open_video_input(self.args, self.source, resources)
                    self.assertIsNone(reader)
                    self.assertTrue(failed.closed)
                    self.assertIn(str(error), reason)
                    self.assert_complete_opencv(frames)

    def test_midstream_error_is_preserved_without_replaying_frames(self):
        error = RuntimeError('native failure in second batch')
        reader = FailingReader(error, fail_at=2)
        with patch('reframe.torchcodec_reader.TorchCodecReader', return_value=reader), ExitStack() as resources:
            actual, frames, _, reason = _open_video_input(self.args, self.source, resources)
            self.assertIs(actual, reader)
            self.assertIsNone(reason)
            self.assertEqual(next(frames)[0], 1)
            with self.assertRaises(RuntimeError) as caught:
                next(frames)
            self.assertIs(caught.exception, error)
        self.assertTrue(reader.closed)

    def test_timestamp_rejection_and_interrupt_do_not_trigger_fallback(self):
        for error in (ValueError('Variable-frame-rate timestamps'), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                reader = FailingReader(error)
                with patch('reframe.torchcodec_reader.TorchCodecReader', return_value=reader), \
                        ExitStack() as resources, self.assertRaises(type(error)) as caught:
                    _open_video_input(self.args, self.source, resources)
                self.assertIs(caught.exception, error)
                self.assertTrue(reader.closed)

    def test_explicit_opencv_bypasses_broken_torchcodec(self):
        self.args.video_decoder = 'opencv'
        with patch('reframe.torchcodec_reader.TorchCodecReader', side_effect=AssertionError('must not load')), \
                ExitStack() as resources:
            reader, frames, _, reason = _open_video_input(self.args, self.source, resources)
            self.assertIsNone(reader)
            self.assertIsNone(reason)
            self.assert_complete_opencv(frames)

    @unittest.skipUnless(importlib.util.find_spec('torchcodec'), 'Optional TorchCodec is not installed')
    def test_real_torchcodec_prefetched_frame_is_not_lost_or_duplicated(self):
        self.args.decode_device = 'cpu'
        with ExitStack() as resources:
            reader, frames, _, reason = _open_video_input(self.args, self.source, resources)
            self.assertIsNotNone(reader)
            self.assertIsNone(reason)
            indices = []
            for index, frame in frames:
                indices.append(index)
                color = frame.rgb[:, 32, 32].tolist()
                self.assertTrue(all(abs(actual - expected) < 8 for actual, expected in
                                    zip(color, (180, 70, 20 + (index - 1) * 20))))
            self.assertEqual(indices, list(range(1, 8)))
            self.assertEqual(reader.telemetry()['frames_yielded'], 7)
