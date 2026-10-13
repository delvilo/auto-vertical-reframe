"""Optional, bounded TorchCodec decoding without silent CUDA-to-CPU fallback.

Frames retain their presentation order and source frame rate.  The rest of the
pipeline currently writes constant-frame-rate video, so irregular presentation
timestamps are rejected rather than silently changing playback speed.
"""
from __future__ import annotations

import math
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .frame_pixels import TensorFrame


def _open_decoder(path: str | Path, device: str):
    # Load native TorchCodec/FFmpeg libraries only when this decoder is opened.
    from torchcodec.decoders import VideoDecoder

    return VideoDecoder(
        str(path), device=device, dimension_order="NCHW", seek_mode="exact",
    )


class TorchCodecReader:
    """Sequential single-pass reader retaining at most one decoded batch.

    A caller retaining a returned TensorFrame can also retain that frame's
    underlying batch.  Consumers should keep only the current/previous frame,
    not accumulate decoded frames.  Timing is host wall time, including any
    synchronization performed internally by the decoder, not a CUDA event time.
    """

    def __init__(self, path: str | Path, device: str = "cpu", batch_size: int = 4):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 8:
            raise ValueError("TorchCodec decode batch size must be an integer from 1 to 8")
        import torch

        requested = torch.device(device)
        if requested.type not in ("cpu", "cuda"):
            raise ValueError("TorchCodec decoding supports explicit cpu or cuda devices only")
        if requested.type == "cpu":
            requested = torch.device("cpu")
        elif requested.index is None:
            requested = torch.device("cuda", torch.cuda.current_device())
        self.device = str(requested)
        self.batch_size = batch_size
        self.pixel_metrics: dict[str, Any] = {}
        self._requested_device = requested
        self._decoder = None
        self._batch = None
        self._batch_offset = 0
        self._next_index = 0
        self._closed = False
        self._frames_decoded = 0
        self._frames_yielded = 0
        self._batches = 0
        self._max_batch = 0
        self._decode_seconds = 0.0
        self._actual_device: str | None = None
        self._fallback: bool | None = None
        self._fallback_known = False
        self._fallback_description = "Not decoded yet"
        started = perf_counter()
        self._decoder = _open_decoder(path, self.device)
        self._initialization_seconds = perf_counter() - started
        self._version = version("torchcodec")
        metadata = self._decoder.metadata
        self.fps = float(metadata.average_fps or 0)
        self.width = int(metadata.width or 0)
        self.height = int(metadata.height or 0)
        self.total_frames = int(metadata.num_frames or 0)
        self._start_seconds = float(metadata.begin_stream_seconds or 0)
        if (
            not math.isfinite(self.fps) or self.fps <= 0
            or self.width <= 0 or self.height <= 0 or self.total_frames <= 0
            or not math.isfinite(self._start_seconds)
        ):
            self.close()
            raise ValueError("TorchCodec returned invalid video dimensions, frame count, FPS or start timestamp")

    def __iter__(self):
        return self

    def __next__(self) -> tuple[int, TensorFrame]:
        if self._closed or self._next_index >= self.total_frames:
            self.close()
            raise StopIteration
        if self._batch is None or self._batch_offset >= self._batch.shape[0]:
            # Drop our prior batch before allocating its replacement.  The
            # caller may still hold a view of the preceding batch.
            self._batch = None
            stop = min(self._next_index + self.batch_size, self.total_frames)
            started = perf_counter()
            try:
                batch = self._decoder.get_frames_in_range(start=self._next_index, stop=stop)
            finally:
                self._decode_seconds += perf_counter() - started
            self._validate_batch(batch, stop - self._next_index)
            self._batch = batch.data
            self._batch_offset = 0
            self._frames_decoded += len(batch.data)
            self._batches += 1
            self._max_batch = max(self._max_batch, len(batch.data))
        from .frame_pixels import TensorFrame

        frame = TensorFrame(self._batch[self._batch_offset], metrics=self.pixel_metrics)
        self._batch_offset += 1
        self._next_index += 1
        self._frames_yielded += 1
        return self._next_index, frame

    def _validate_batch(self, batch, expected_count: int) -> None:
        import torch

        data = batch.data
        if tuple(data.shape) != (expected_count, 3, self.height, self.width) or data.dtype != torch.uint8:
            raise RuntimeError("TorchCodec must return a complete batch of RGB uint8 NCHW frames")
        status = self._decoder.cpu_fallback
        self._fallback_known = bool(status.status_known)
        self._fallback = bool(status) if self._fallback_known else None
        self._fallback_description = str(status)
        self._actual_device = str(data.device)
        if self._requested_device.type == "cuda":
            if not self._fallback_known or self._fallback:
                raise RuntimeError(
                    "CUDA decoding was requested, but hardware decoding was not verified: "
                    + self._fallback_description
                )
        if data.device.type != self._requested_device.type or (
            self._requested_device.index is not None
            and data.device.index != self._requested_device.index
        ):
            raise RuntimeError(
                f"TorchCodec requested {self.device} but returned {data.device} frames"
            )
        # TorchCodec exposes timestamps as tiny CPU float64 tensors.  Accessing
        # them does not copy source video pixels back from CUDA.
        timestamps = batch.pts_seconds.tolist()
        if len(timestamps) != expected_count:
            raise RuntimeError("TorchCodec returned an incomplete timestamp batch")
        for offset, actual in enumerate(timestamps):
            index = self._next_index + offset
            expected = self._start_seconds + index / self.fps
            if not math.isfinite(actual) or abs(actual - expected) > .001:
                raise ValueError(
                    "Variable-frame-rate or discontinuous timestamps are not supported by "
                    f"the constant-FPS writer (frame {index + 1}: PTS {actual}, "
                    f"expected {expected:.9f} at {self.fps:g} FPS)"
                )

    def telemetry(self) -> dict[str, Any]:
        """Return JSON-serializable measurements, available after close()."""
        result = {
            "decoder": "torchcodec", "torchcodec_version": self._version,
            "seek_mode": "exact", "requested_device": self.device,
            "actual_device": self._actual_device,
            "cpu_fallback": self._fallback,
            "cpu_fallback_status_known": self._fallback_known,
            "cpu_fallback_description": self._fallback_description,
            "initialization_wall_seconds": self._initialization_seconds,
            "decode_wall_seconds": self._decode_seconds,
            "frames_decoded": self._frames_decoded,
            "frames_yielded": self._frames_yielded,
            "batches": self._batches, "batch_size": self.batch_size,
            "max_batch_frames": self._max_batch,
            "source_frames": self.total_frames, "source_fps": self.fps,
            "width": self.width, "height": self.height,
            "timestamp_tolerance_seconds": .001,
            "pixels": dict(self.pixel_metrics),
        }
        if self._actual_device is not None and self._actual_device.startswith("cuda"):
            import torch
            result["cuda_allocator_peak"] = {
                "device": self._actual_device,
                "allocated_bytes": torch.cuda.max_memory_allocated(self._actual_device),
                "reserved_bytes": torch.cuda.max_memory_reserved(self._actual_device),
                "scope": "process_lifetime_pytorch_allocator_only",
            }
        return result

    def close(self) -> None:
        """Release reader-owned tensors and the decoder; safe to call twice."""
        self._closed = True
        self._batch = None
        self._decoder = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
