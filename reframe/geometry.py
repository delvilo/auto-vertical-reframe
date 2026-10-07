from __future__ import annotations

import math

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def lerp(a: float, b: float, alpha: float) -> float:
    return a + (b - a) * alpha


def compute_base_crop(
    frame_w: int,
    frame_h: int,
    aspect_w: int = 9,
    aspect_h: int = 16,
) -> tuple[int, int]:
    target_ratio = aspect_w / aspect_h
    frame_ratio = frame_w / frame_h
    if frame_ratio >= target_ratio:
        crop_h = frame_h
        crop_w = int(round(crop_h * target_ratio))
    else:
        crop_w = frame_w
        crop_h = int(round(crop_w / target_ratio))
    return min(crop_w, frame_w), min(crop_h, frame_h)


def current_crop_size(
    base_crop_w: int,
    base_crop_h: int,
    zoom: float,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int]:
    safe_zoom = max(0.1, zoom)
    crop_w = int(round(base_crop_w / safe_zoom))
    crop_h = int(round(base_crop_h / safe_zoom))
    crop_w = max(64, min(crop_w, frame_w))
    crop_h = max(64, min(crop_h, frame_h))
    return crop_w, crop_h


def box_iou(a, b) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1.0)


def critically_damped_step(current, target, velocity, dt, tau, max_speed):
    """Exact critical spring solution for a fixed target during dt seconds."""
    omega = 2.0 / max(tau, 1e-4)
    offset = current - target
    term = velocity + omega * offset
    decay = math.exp(-omega * dt)
    value = target + (offset + term * dt) * decay
    new_velocity = (velocity - omega * term * dt) * decay
    delta = clamp(value - current, -max_speed * dt, max_speed * dt)
    value = current + delta
    if (target - current) * (target - value) <= 0:
        return target, 0.0
    return value, clamp(new_velocity, -max_speed, max_speed)
