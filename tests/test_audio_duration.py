"""Real FFmpeg regression checks: rendered video, not source audio, bounds output."""
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import cv2
import numpy as np

from reframe.config import AppConfig
from reframe.video_io import DirectVideoWriter, LosslessWriter, run_ffmpeg_mux


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                     "FFmpeg and ffprobe are required")
class AudioDurationTests(unittest.TestCase):
    FPS = 60
    FULL_FRAMES = 120
    SIZE = (64, 64)

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.sources = {}
        # Eight missing frames at 60 FPS reproduces the long-video failure.
        for name, audio_seconds in (("short", 112 / cls.FPS), ("equal", 2),
                                    ("long", 3), ("none", None)):
            path = cls.directory / f"source-{name}.mkv"
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                       "-f", "lavfi", "-i", "color=c=gray:s=64x64:r=60:d=2"]
            if audio_seconds is not None:
                command += ["-f", "lavfi", "-i",
                            f"sine=frequency=440:sample_rate=44100:duration={audio_seconds}",
                            "-c:a", "pcm_s16le"]
            command += ["-c:v", "ffv1", str(path)]
            subprocess.run(command, check=True, capture_output=True, timeout=30)
            cls.sources[name] = path

    def render(self, mode, source, frames, label):
        output = self.directory / f"{mode}-{label}.mp4"
        args = AppConfig(video_encoder="libx264", preset_ffmpeg="ultrafast",
                         ffmpeg_log_level="error")
        if mode == "direct":
            writer = DirectVideoWriter(output, source, self.FPS, self.SIZE, args)
        else:
            intermediate = self.directory / f"{mode}-{label}.mkv"
            writer = LosslessWriter(str(intermediate), self.FPS, self.SIZE, "error")
        try:
            for index in range(frames):
                # Losing or duplicating early frames must not hide a missing tail.
                shade = 240 if index >= frames - 8 else 40
                writer.write(np.full((*self.SIZE[::-1], 3), shade, dtype=np.uint8))
            writer.release()
        finally:
            writer.abort()
        if mode == "lossless":
            run_ffmpeg_mux(str(intermediate), str(source), str(output), "libx264",
                           args.audio_bitrate, args.crf, args.preset_ffmpeg, None,
                           ffmpeg_log_level="error")
        return output

    def assert_media(self, path, frames, has_audio):
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-show_streams",
             "-show_format", "-of", "json", str(path)],
            check=True, capture_output=True, text=True, timeout=30)
        media = json.loads(result.stdout)
        video = next(stream for stream in media["streams"] if stream["codec_type"] == "video")
        audio = [stream for stream in media["streams"] if stream["codec_type"] == "audio"]
        self.assertEqual(int(video["nb_read_frames"]), frames)
        self.assertEqual(Fraction(video["avg_frame_rate"]), self.FPS)
        self.assertAlmostEqual(float(video["duration"]), frames / self.FPS, places=5)
        self.assertEqual(bool(audio), has_audio)
        # FFmpeg 7.1.5 can mux about 0.5 s of buffered audio past video EOF with
        # -shortest, in addition to AAC sample-frame rounding. Preserve exact
        # video length and bound this tail; do not promise sample-exact audio EOF.
        self.assertGreaterEqual(float(media["format"]["duration"]), frames / self.FPS)
        self.assertLess(float(media["format"]["duration"]), frames / self.FPS + .6)
        if audio:
            self.assertGreaterEqual(float(audio[0]["duration"]), frames / self.FPS - .025)
            self.assertLess(float(audio[0]["duration"]), frames / self.FPS + .6)
        capture = cv2.VideoCapture(str(path))
        try:
            decoded = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                decoded.append(float(frame.mean()))
            self.assertEqual(len(decoded), frames)
            self.assertTrue(all(value > 220 for value in decoded[-8:]))
            self.assertLess(decoded[-9], 60)
        finally:
            capture.release()

    def test_both_writers_keep_all_frames_for_short_equal_long_or_absent_audio(self):
        for mode in ("direct", "lossless"):
            for name, source in self.sources.items():
                with self.subTest(mode=mode, audio=name):
                    output = self.render(mode, source, self.FULL_FRAMES, name)
                    self.assert_media(output, self.FULL_FRAMES, name != "none")
                    if name == "short":
                        result = subprocess.run(
                            ["ffmpeg", "-v", "error", "-i", str(output), "-map", "0:a:0",
                             "-f", "s16le", "-ac", "1", "-ar", "44100", "pipe:1"],
                            check=True, capture_output=True, timeout=30)
                        samples = np.frombuffer(result.stdout, dtype="<i2").astype(np.float64)
                        original = samples[int(.2 * 44100):int(.4 * 44100)]
                        padded = samples[int(1.95 * 44100):int(1.99 * 44100)]
                        self.assertGreater(np.sqrt(np.mean(original ** 2)), 500)
                        self.assertEqual(len(padded), int(1.99 * 44100) - int(1.95 * 44100))
                        self.assertLess(np.sqrt(np.mean(padded ** 2)), 50)

    def test_partial_render_stops_before_long_source_audio(self):
        for mode in ("direct", "lossless"):
            with self.subTest(mode=mode):
                output = self.render(mode, self.sources["long"], 30, "partial")
                self.assert_media(output, 30, True)


if __name__ == "__main__":
    unittest.main()
