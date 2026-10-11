"""Bounded, per-frame RGB tensor views; CPU copies are explicit and counted.

Tensor interpolation has the same geometry as the reference OpenCV path but can
round pixels differently. Area resizing integrates fractional source pixels;
PyTorch's adaptive-average "area" mode is not equivalent for noninteger scales.
Timers measure host wall time (including downloads),
not CUDA kernel time. A bounded cache retains only resize-coordinate vectors;
frame pixels and full-video tensors are never stored in a global cache.
"""
from __future__ import annotations

from functools import lru_cache
import time


def _area_axis(pixels, length, axis):
    """Separable area-overlap integration, without dense resize matrices."""
    import torch
    source = pixels.shape[axis]
    if source == length:
        return pixels
    # Integrate each piecewise-constant source pixel at destination boundaries.
    # Reduce width first so the second cumulative sum is only thumbnail-width.
    boundaries = torch.arange(length + 1, device=pixels.device, dtype=torch.float32)
    boundaries = boundaries * (source / length)
    lower = boundaries.floor().long().clamp_(0, source)
    fraction = boundaries - lower
    integral = pixels.cumsum(axis)
    shape = [1] * pixels.ndim
    shape[axis] = length + 1
    edges = integral.index_select(axis, (lower - 1).clamp_min(0))
    edges = edges * (lower > 0).reshape(shape)
    edges = edges + pixels.index_select(axis, lower.clamp_max(source - 1)) * fraction.reshape(shape)
    return edges.diff(dim=axis) * (length / source)


def _area_resize(pixels, height, width):
    return _area_axis(_area_axis(pixels, width, -1), height, -2)


@lru_cache(maxsize=16)
def _bilinear_sampling_axes(source_h, source_w, height, width, device):
    """Cache small, read-only coordinate vectors, never source/output pixels."""
    import torch
    # Match interpolate(..., align_corners=False)'s half-pixel coordinates.
    # Work in float32, like the existing float32 interpolation path.
    y = ((torch.arange(height, device=device, dtype=torch.float32) + 0.5)
         * (source_h / height) - 0.5).clamp_min_(0)
    x = ((torch.arange(width, device=device, dtype=torch.float32) + 0.5)
         * (source_w / width) - 0.5).clamp_min_(0)
    y0, x0 = y.long(), x.long()
    y1 = (y0 + 1).clamp_max_(source_h - 1)
    x1 = (x0 + 1).clamp_max_(source_w - 1)
    wy, wx = (y - y0)[None, :, None], (x - x0)[None, None, :]
    return y0, y1, x0, x1, wy, wx, 1 - wy, 1 - wx


def _bilinear_sampled_resize(rgb, height, width):
    """Gather four uint8 neighbors before allocating small float32 tensors.

    This has the same sampling support as non-antialiased bilinear interpolation.
    It accepts strided precrop/decoder views without making a full-frame copy.
    At most three destination-sized float32 planes (per channel) and one uint8
    gather result coexist. Normal float32 rounding may differ by one byte from
    the native float32 resize after quantization.
    """
    source_h, source_w = rgb.shape[-2:]
    y0, y1, x0, x1, wy, wx, wy0, wx0 = _bilinear_sampling_axes(
        source_h, source_w, height, width, rgb.device)
    top = rgb[:, y0[:, None], x0[None, :]].float().mul_(wx0)
    top.addcmul_(rgb[:, y0[:, None], x1[None, :]].float(), wx)
    bottom = rgb[:, y1[:, None], x0[None, :]].float().mul_(wx0)
    bottom.addcmul_(rgb[:, y1[:, None], x1[None, :]].float(), wx)
    return top.mul_(wy0).addcmul_(bottom, wy)


class TensorFrame:
    def __init__(self, rgb, metrics=None):
        import torch
        if not isinstance(rgb, torch.Tensor) or rgb.dtype != torch.uint8:
            raise TypeError("TensorFrame requires a uint8 torch tensor")
        if rgb.ndim != 3 or rgb.shape[0] != 3 or min(rgb.shape[1:]) < 1:
            raise ValueError("TensorFrame requires nonempty RGB CHW pixels")
        self.rgb = rgb
        self.metrics = metrics if metrics is not None else {}
        self._cache = {}

    @property
    def shape(self):
        return (int(self.rgb.shape[1]), int(self.rgb.shape[2]), 3)

    def _add(self, key, amount=1):
        self.metrics[key] = self.metrics.get(key, 0) + amount

    def _cached(self, key, operation):
        if key in self._cache:
            self._add("cache_hits")
            return self._cache[key]
        value = operation()
        if len(self._cache) >= 8:
            del self._cache[next(iter(self._cache))]
        self._cache[key] = value
        return value

    def __getitem__(self, slices):
        if (not isinstance(slices, tuple) or len(slices) != 2
                or any(not isinstance(s, slice) or s.step not in (None, 1) for s in slices)):
            raise TypeError("TensorFrame supports two spatial slices with step 1")
        return TensorFrame(self.rgb[:, slices[0], slices[1]], self.metrics)

    def _download(self, value):
        started = time.perf_counter()
        result = value.contiguous().cpu().numpy()
        self._add("cpu_materializations")
        self._add("cpu_materialized_bytes", result.nbytes)
        if value.device.type == "cuda":
            self._add("device_to_host_calls")
            self._add("device_to_host_bytes", result.nbytes)
        self._add("cpu_materialization_host_seconds", time.perf_counter() - started)
        return result

    def bgr(self):
        return self._cached("bgr", lambda: self._download(self.rgb.flip(0).permute(1, 2, 0)))

    def resized_rgb(self, size, interpolation="area"):
        import torch.nn.functional as F
        width, height = map(int, size)
        if min(width, height) < 1 or interpolation not in {"area", "bilinear"}:
            raise ValueError("Resize requires positive dimensions and area or bilinear interpolation")
        def resize():
            if (width, height) == (self.shape[1], self.shape[0]):
                return self.rgb
            started = time.perf_counter()
            if interpolation == "area":
                pixels = _area_resize(self.rgb.float(), height, width)
            elif (self.rgb.device.type == "cuda"
                  and height <= self.shape[0] and width <= self.shape[1]
                  and height * width * 8 <= self.shape[0] * self.shape[1]
                  and max(self.shape[:2]) <= 8192):
                # Large GPU downscales do not need a full float32 source frame.
                # Keep native interpolation for rendering/upscales and on CPU:
                # CPU gather overhead can outweigh the saved conversion cost.
                pixels = _bilinear_sampled_resize(self.rgb, height, width)
                self._add("bilinear_sampled_calls")
            else:
                pixels = F.interpolate(self.rgb[None].float(), size=(height, width),
                                       mode="bilinear", align_corners=False)[0]
            self._add("resize_calls")
            self._add("resize_host_seconds", time.perf_counter() - started)
            return pixels.round().clamp_(0, 255).to(self.rgb.dtype)
        return self._cached(("rgb", width, height, interpolation), resize)

    def resized_bgr(self, size, interpolation="area"):
        return self._cached(("bgr", tuple(size), interpolation), lambda: self._download(
            self.resized_rgb(size, interpolation).flip(0).permute(1, 2, 0)))

    def resized(self, size, interpolation="area"):
        return self._cached(("frame", tuple(size), interpolation), lambda: TensorFrame(
            self.resized_rgb(size, interpolation), self.metrics))

    def gray_thumbnail(self, size, method="gray-area"):
        import cv2
        if len(size) != 2 or min(size) < 1 or any(int(n) != n for n in size):
            raise ValueError("Thumbnail dimensions must be positive integers")
        if method not in {"gray-area", "bgr-area"}:
            raise ValueError("Unknown grayscale thumbnail method")
        def thumbnail():
            if method == "bgr-area":
                return cv2.cvtColor(self.resized_bgr(size), cv2.COLOR_BGR2GRAY)
            # Keep only one float grayscale source plane, rather than three.
            started = time.perf_counter()
            rgb = self.rgb
            gray = (rgb[0].float() * 0.299 + rgb[1].float() * 0.587
                    + rgb[2].float() * 0.114).round()
            gray = _area_resize(gray, size[1], size[0])
            self._add("resize_calls")
            self._add("resize_host_seconds", time.perf_counter() - started)
            return self._download(gray.round().clamp_(0, 255).to(rgb.dtype))
        return self._cached(("gray", tuple(size), method), thumbnail)

    def letterbox(self, size=(640, 640), auto=False, stride=32):
        import torch.nn.functional as F
        height, width = self.shape[:2]
        target_h, target_w = size
        if min(target_h, target_w, stride) < 1:
            raise ValueError("Letterbox dimensions and stride must be positive")
        gain = min(target_h / height, target_w / width)
        resized_w, resized_h = round(width * gain), round(height * gain)
        dw, dh = target_w - resized_w, target_h - resized_h
        if auto:
            dw, dh = dw % stride, dh % stride
        left, right = round(dw / 2 - 0.1), round(dw / 2 + 0.1)
        top, bottom = round(dh / 2 - 0.1), round(dh / 2 + 0.1)
        pixels = self.resized_rgb((resized_w, resized_h), "bilinear")
        pixels = F.pad(pixels, (left, right, top, bottom), value=114)
        return pixels[None].float().div_(255), {
            "gain": gain, "padding": (left, top, right, bottom),
            "original_shape": (height, width), "resized_shape": (resized_h, resized_w),
        }
