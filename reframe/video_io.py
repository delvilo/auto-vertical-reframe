from __future__ import annotations

import codecs
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import deque
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from reframe.config import AppConfig

class StderrTee:
    """Continuously drain a child's stderr to the terminal and keep a bounded error tail."""

    def __init__(self, pipe):
        self.pipe = pipe
        self.chunks = deque(maxlen=64)
        self.stream = sys.stderr
        self.thread = threading.Thread(target=self._pump, name="ffmpeg-stderr", daemon=True)
        self.thread.start()

    def _pump(self):
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while chunk := self.pipe.read1(4096):
                self.chunks.append(chunk)
                self.stream.write(decoder.decode(chunk))
                self.stream.flush()
            self.stream.write(decoder.decode(b"", final=True))
            self.stream.flush()
        finally:
            self.pipe.close()

    def finish(self) -> str:
        self.thread.join()
        return b"".join(self.chunks).decode("utf-8", errors="replace")[-6000:]


def start_ffmpeg(cmd, stdin=subprocess.DEVNULL):
    logging.debug("FFmpeg command: %s", shlex.join(cmd))
    process = subprocess.Popen(cmd, stdin=stdin, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    logging.debug("FFmpeg PID=%s", process.pid)
    return process, StderrTee(process.stderr)


def run_ffmpeg(cmd, timeout=None):
    process, stderr = start_ffmpeg(cmd)
    try:
        code = process.wait(timeout=timeout)
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        tail = stderr.finish()
        logging.debug("FFmpeg exit=%s", process.returncode)
    return subprocess.CompletedProcess(cmd, code, stderr=tail)


def has_ffmpeg_encoder(encoder_name: str) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    try:
        logging.debug("FFmpeg command: ffmpeg -hide_banner -encoders")
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if result.stderr:
            sys.stderr.write(result.stderr)
            sys.stderr.flush()
        if result.returncode:
            logging.warning("FFmpeg encoder listing failed: exit=%s", result.returncode)
            return False
        return encoder_name in result.stdout
    except Exception:
        logging.warning("Could not query FFmpeg encoders", exc_info=True)
        return False


def build_video_filters(post_restore: bool) -> Optional[str]:
    filters = []
    if post_restore:
        filters.append("hqdn3d=1.2:1.2:6:6")
        filters.append("unsharp=5:5:0.6:5:5:0.0")
    if not filters:
        return None
    return ",".join(filters)


def run_ffmpeg_mux(
    silent_video_path: str,
    source_input_path: str,
    final_output_path: str,
    video_encoder: str,
    audio_bitrate: str,
    crf: int,
    preset: str,
    vf: Optional[str],
    nvenc_preset: str = "p5",
    video_bitrate: str = "8M",
    ffmpeg_log_level: str = "warning",
) -> str:
    """Encodes a lossless intermediate with runtime hardware fallback."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")
    supported = {
        "libx264",
        "libx265",
        "h264_nvenc",
        "hevc_nvenc",
        "h264_videotoolbox",
        "hevc_videotoolbox",
    }
    if video_encoder != "auto" and video_encoder not in supported:
        raise ValueError(
            f"Unsupported encoder: {video_encoder}; choose {sorted(supported)}"
        )
    if video_encoder == "auto":
        encoders = [
            e
            for e in ("h264_nvenc", "h264_videotoolbox", "libx264")
            if has_ffmpeg_encoder(e)
        ]
    else:
        encoders = [video_encoder] if has_ffmpeg_encoder(video_encoder) else []
        if not encoders:
            logging.warning("Requested encoder %s is absent; trying libx264 fallback", video_encoder)
        if video_encoder != "libx264":
            encoders.append("libx264")
    encoders = list(dict.fromkeys(encoders))
    if not encoders:
        raise RuntimeError("No usable video encoder in this FFmpeg build")
    final_path = Path(final_output_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{final_path.stem}-",
        suffix=final_path.suffix or ".mp4",
        dir=final_path.parent,
    )
    os.close(fd)
    partial = Path(name)
    errors = []
    try:
        for encoder in encoders:
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                ffmpeg_log_level,
                "-y",
                "-i",
                silent_video_path,
                "-i",
                source_input_path,
                "-map",
                "0:v:0",
                "-map",
                "1:a?",
                "-c:v",
                encoder,
            ]
            if vf:
                cmd += ["-vf", vf]
            if encoder in {"libx264", "libx265"}:
                cmd += ["-crf", str(crf), "-preset", preset]
            elif encoder.endswith("_nvenc"):
                cmd += [
                    "-cq",
                    str(crf),
                    "-preset",
                    nvenc_preset,
                    "-rc",
                    "vbr",
                    "-b:v",
                    "0",
                ]
            else:
                cmd += ["-b:v", video_bitrate]
            cmd += [
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                audio_bitrate,
                "-shortest",
            ]
            if final_path.suffix.lower() in {".mp4", ".mov", ".m4v"}:
                cmd += ["-movflags", "+faststart"]
            cmd += [str(partial)]
            result = run_ffmpeg(cmd)
            if result.returncode == 0:
                os.replace(partial, final_path)
                logging.info("Encoded %s with %s", final_path, encoder)
                return encoder
            errors.append(f"{encoder}: {result.stderr[-4000:]}")
            logging.warning(
                "Encoder %s failed; trying next available encoder. %s",
                encoder,
                result.stderr[-1200:],
            )
        raise RuntimeError("All encoding attempts failed:\n" + "\n".join(errors))
    finally:
        partial.unlink(missing_ok=True)


def iter_video_frames(path: Path):
    """Decode sequentially so scene boundaries can reset trackers BEFORE tracking."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            yield index, frame
    finally:
        cap.release()


class LosslessWriter:
    """BGR frames -> lossless FFV1 temporary file; final encode can safely retry."""

    def __init__(self, path: str, fps: float, size: tuple[int, int], ffmpeg_log_level="warning"):
        self.width, self.height = size
        self.closed = False
        self.process, self.error_output = start_ffmpeg(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                ffmpeg_log_level,
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{self.width}x{self.height}",
                "-r",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "ffv1",
                "-level",
                "3",
                "-pix_fmt",
                "bgr0",
                path,
            ],
            stdin=subprocess.PIPE,
        )

    def isOpened(self):
        return not self.closed and self.process.poll() is None

    def _error(self):
        return self.error_output.finish()

    def write(self, frame):
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError(
                "LosslessWriter expects uint8 BGR frames at output resolution"
            )
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            self.process.wait()
            message = self._error()
            self.abort()
            raise RuntimeError(f"Lossless encoding failed: {message}") from exc

    def release(self):
        if self.closed:
            return
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass  # Drain stderr and report the child's exit status below.
            code = self.process.wait()
            self.error_output.finish()
            logging.debug("Lossless FFmpeg exit=%s", code)
            if code:
                raise RuntimeError(f"Lossless encoding failed: {self._error()}")
        finally:
            self.abort()

    def abort(self):
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        try:
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
        except BrokenPipeError:
            pass
        self.error_output.finish()


def encoder_options(encoder, args):
    if encoder in {"libx264", "libx265"}:
        return ["-crf", str(args.crf), "-preset", args.preset_ffmpeg]
    if encoder.endswith("_nvenc"):
        return [
            "-cq",
            str(args.crf),
            "-preset",
            args.nvenc_preset,
            "-rc",
            "vbr",
            "-b:v",
            "0",
        ]
    return ["-b:v", args.video_bitrate]


def select_live_encoder(args, fps, size, vf):
    """Initializes encoders with a 2-frame test before starting full render."""
    requested = args.video_encoder
    options = (
        ["h264_nvenc", "h264_videotoolbox", "libx264"]
        if requested == "auto"
        else [requested]
    )
    if "libx264" not in options:
        options.append("libx264")
    errors = []
    for encoder in options:
        if not has_ffmpeg_encoder(encoder):
            logging.warning("Encoder %s is absent from this FFmpeg build; trying fallback", encoder)
            continue
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            args.ffmpeg_log_level,
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={size[0]}x{size[1]}:r={fps}",
            "-frames:v",
            "2",
        ]
        if vf:
            cmd += ["-vf", vf]
        cmd += [
            "-c:v",
            encoder,
            *encoder_options(encoder, args),
            "-pix_fmt",
            "yuv420p",
            "-f",
            "null",
            "-",
        ]
        try:
            result = run_ffmpeg(cmd, timeout=30)
            if result.returncode == 0:
                logging.info("Selected live encoder: %s (requested=%s)", encoder, requested)
                return encoder
            errors.append(f"{encoder}: {result.stderr[-1000:]}")
        except subprocess.TimeoutExpired:
            errors.append(f"{encoder}: initialization timed out")
            logging.warning("Encoder %s initialization timed out after 30 seconds", encoder)
        logging.warning("Encoder %s failed initialization; trying fallback", encoder)
    raise RuntimeError("No encoder could initialize:\n" + "\n".join(errors))


class DirectVideoWriter(LosslessWriter):
    """Raw BGR pipe directly into final video encoding and source audio mux in one pass."""

    def __init__(self, destination, source, fps, size, args, vf=None):
        self.width, self.height = size
        self.closed = False
        self.frame_count = 0
        self.destination = Path(destination)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.encoder = select_live_encoder(args, fps, size, vf)
        fd, path = tempfile.mkstemp(
            prefix=f".{self.destination.stem}-",
            suffix=self.destination.suffix or ".mp4",
            dir=self.destination.parent,
        )
        os.close(fd)
        self.partial = Path(path)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            args.ffmpeg_log_level,
            "-y",
            "-thread_queue_size",
            "64",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            self.encoder,
            *encoder_options(self.encoder, args),
            "-pix_fmt",
            "yuv420p",
        ]
        if vf:
            cmd += ["-vf", vf]
        cmd += ["-c:a", "aac", "-b:a", args.audio_bitrate, "-shortest"]
        if self.destination.suffix.lower() in {".mp4", ".mov", ".m4v"}:
            cmd += ["-movflags", "+faststart"]
        cmd += [str(self.partial)]
        try:
            self.process, self.error_output = start_ffmpeg(
                cmd,
                stdin=subprocess.PIPE,
            )
        except BaseException:
            self.partial.unlink(missing_ok=True)
            raise

    def write(self, frame):
        super().write(frame)
        self.frame_count += 1

    def release(self):
        if self.closed:
            return
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass  # Drain stderr and report the child's exit status below.
            result = self.process.wait()
            self.error_output.finish()
            logging.debug("Direct FFmpeg exit=%s encoder=%s", result, self.encoder)
            if result or self.frame_count == 0:
                raise RuntimeError(
                    f"Direct encoding failed: {self._error()}. "
                    "Retry with --video-encoder libx264 or --encode-mode lossless"
                )
            os.replace(self.partial, self.destination)
        finally:
            self.abort()

    def abort(self):
        super().abort()
        self.partial.unlink(missing_ok=True)

