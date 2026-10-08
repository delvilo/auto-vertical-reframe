from __future__ import annotations

import math
import numpy as np
from typing import Optional
from reframe.config import AppConfig
from reframe.contracts import Candidate, CameraObservation, CameraState
from reframe.geometry import clamp, lerp, current_crop_size, critically_damped_step
from reframe.saliency.regions import extract_saliency_region

def derive_candidate_focus_bounds(
    subject: Candidate,
) -> tuple[float, float, float, float]:
    """Derives visual focus bounding box with proper vertical headroom."""
    if (
        subject.cls_name == "person"
        and subject.body_min_x is not None
        and subject.body_max_x is not None
        and subject.body_top_y is not None
        and subject.body_bottom_y is not None
    ):
        left = float(subject.body_min_x)
        right = float(subject.body_max_x)
        top = float(subject.body_top_y)
        bottom = float(subject.body_bottom_y)

        left = min(left, subject.x1)
        right = max(right, subject.x2)
        if not subject.body_bottom_confident:
            bottom = max(bottom, subject.y2)

        height = max(1.0, bottom - top)
        if subject.eye_y is not None:
            headroom = max(0.0, float(subject.eye_y) - top)
            top -= max(height * 0.12, headroom * 1.35)
        else:
            top -= height * 0.15

        if subject.body_bottom_confident:
            bottom += height * 0.08
        else:
            bottom += height * 0.18
    elif subject.cls_name == "person" and subject.head_box is not None:
        fx1, fy1, fx2, fy2 = subject.head_box
        head_w = max(1.0, fx2 - fx1)
        head_h = max(1.0, fy2 - fy1)
        left = min(subject.x1, fx1 - head_w * 0.15)
        right = max(subject.x2, fx2 + head_w * 0.15)
        top = min(subject.y1, fy1 - head_h * 0.15)
        bottom = subject.y2 + subject.height * 0.08
    else:
        left = subject.x1
        right = subject.x2
        top = subject.mask_top_y
        bottom = subject.y2
        height = max(1.0, bottom - top)
        width = max(1.0, right - left)
        left -= width * 0.14
        right += width * 0.14
        top -= height * 0.14
        bottom += height * 0.16

    if (
        subject.salient_x1 is not None
        and subject.salient_y1 is not None
        and subject.salient_x2 is not None
        and subject.salient_y2 is not None
    ):
        saliency_pad_x = max(8.0, (subject.salient_x2 - subject.salient_x1) * 0.18)
        saliency_pad_y = max(8.0, (subject.salient_y2 - subject.salient_y1) * 0.22)
        left = min(left, subject.salient_x1 - saliency_pad_x)
        top = min(top, subject.salient_y1 - saliency_pad_y)
        right = max(right, subject.salient_x2 + saliency_pad_x)
        bottom = max(bottom, subject.salient_y2 + saliency_pad_y)

    return left, top, right, bottom


def compute_observation_from_bounds(
    bounds: tuple[float, float, float, float],
    confidence: float,
    base_crop_w: int,
    base_crop_h: int,
    frame_w: int,
    frame_h: int,
    min_zoom: float,
    max_zoom: float,
    anchor_y: Optional[float] = None,
) -> CameraObservation:
    left, top, right, bottom = bounds
    left = clamp(left, 0.0, frame_w - 1.0)
    top = clamp(top, 0.0, frame_h - 1.0)
    right = clamp(right, left + 1.0, float(frame_w))
    bottom = clamp(bottom, top + 1.0, float(frame_h))

    region_w = max(1.0, right - left)
    region_h = max(1.0, bottom - top)
    pad_x = max(region_w * 0.12, 16.0)
    pad_y = max(region_h * 0.16, 20.0)

    fit_w = min(frame_w, region_w + pad_x * 2.0)
    fit_h = min(frame_h, region_h + pad_y * 2.0)
    zoom_w = base_crop_w / max(1.0, fit_w)
    zoom_h = base_crop_h / max(1.0, fit_h)
    zoom = clamp(min(zoom_w, zoom_h), min_zoom, max_zoom)

    crop_w, crop_h = current_crop_size(
        base_crop_w,
        base_crop_h,
        zoom,
        frame_w,
        frame_h,
    )

    center_x = clamp((left + right) / 2.0, crop_w / 2.0, frame_w - crop_w / 2.0)

    if anchor_y is not None:
        target_center_y = anchor_y + crop_h * 0.15
        center_y = clamp(target_center_y, crop_h / 2.0, frame_h - crop_h / 2.0)
    else:
        center_y = clamp((top + bottom) / 2.0, crop_h / 2.0, frame_h - crop_h / 2.0)

    return CameraObservation(
        center_x=center_x,
        center_y=center_y,
        zoom=zoom,
        confidence=confidence,
    )


def build_single_subject_observation(
    subject: Candidate,
    base_crop_w: int,
    base_crop_h: int,
    frame_w: int,
    frame_h: int,
    min_zoom: float,
    max_zoom: float,
) -> CameraObservation:
    bounds = derive_candidate_focus_bounds(subject)
    confidence = clamp(
        subject.conf * 0.60
        + subject.rank_confidence * 0.20
        + subject.saliency_confidence * 0.20,
        0.2,
        1.0,
    )

    anchor_y = None
    if subject.eye_y is not None:
        anchor_y = subject.eye_y
    elif subject.head_box is not None:
        anchor_y = (
            subject.head_box[1] + (subject.head_box[3] - subject.head_box[1]) * 0.45
        )

    return compute_observation_from_bounds(
        bounds=bounds,
        confidence=confidence,
        base_crop_w=base_crop_w,
        base_crop_h=base_crop_h,
        frame_w=frame_w,
        frame_h=frame_h,
        min_zoom=min_zoom,
        max_zoom=max_zoom,
        anchor_y=anchor_y,
    )


def build_pair_observation(
    first: Candidate,
    second: Candidate,
    base_crop_w: int,
    base_crop_h: int,
    frame_w: int,
    frame_h: int,
    min_zoom: float,
    max_zoom: float,
) -> CameraObservation:
    first_bounds = derive_candidate_focus_bounds(first)
    second_bounds = derive_candidate_focus_bounds(second)
    union_bounds = (
        min(first_bounds[0], second_bounds[0]),
        min(first_bounds[1], second_bounds[1]),
        max(first_bounds[2], second_bounds[2]),
        max(first_bounds[3], second_bounds[3]),
    )
    confidence = clamp((first.conf + second.conf) / 2.0, 0.25, 1.0)
    eyes = [y for y in (first.eye_y, second.eye_y) if y is not None]
    anchor_y = (sum(eyes) / len(eyes)) if eyes else None

    return compute_observation_from_bounds(
        bounds=union_bounds,
        confidence=confidence,
        base_crop_w=base_crop_w,
        base_crop_h=base_crop_h,
        frame_w=frame_w,
        frame_h=frame_h,
        min_zoom=min_zoom,
        max_zoom=max_zoom,
        anchor_y=anchor_y,
    )


def build_global_saliency_observation(
    saliency_map: np.ndarray | None,
    base_crop_w: int,
    base_crop_h: int,
    frame_w: int,
    frame_h: int,
    min_zoom: float,
    max_zoom: float,
    saliency_bounds: tuple[int, int, int, int] | None = None,
) -> Optional[CameraObservation]:
    region = extract_saliency_region(
        saliency_map, (0, 0, frame_w, frame_h), frame_shape=(frame_h, frame_w),
        map_bounds=saliency_bounds,
    )
    if region is None or region[6] < 0.02:
        return None
    return compute_observation_from_bounds(
        bounds=(region[2], region[3], region[4], region[5]),
        confidence=clamp(region[6], 0.12, 0.55),
        base_crop_w=base_crop_w,
        base_crop_h=base_crop_h,
        frame_w=frame_w,
        frame_h=frame_h,
        min_zoom=min_zoom,
        max_zoom=max_zoom,
    )


def apply_camera_motion(
    state: CameraState,
    observation: CameraObservation,
    args: AppConfig,
    base_crop_w: int,
    base_crop_h: int,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int]:
    dt = 1.0 / args.runtime_fps
    desired_zoom = clamp(
        args.fixed_zoom if args.fixed_zoom is not None else observation.zoom,
        args.min_zoom,
        args.max_zoom,
    )
    if state.is_scene_cut:
        state.zoom = state.target_zoom = desired_zoom
        cw, ch = current_crop_size(
            base_crop_w, base_crop_h, state.zoom, frame_w, frame_h
        )
        state.crop_center_x = clamp(observation.center_x, cw / 2.0, frame_w - cw / 2.0)
        state.crop_center_y = clamp(observation.center_y, ch / 2.0, frame_h - ch / 2.0)
        state.target_center_x, state.target_center_y = (
            state.crop_center_x,
            state.crop_center_y,
        )
        state.velocity_x = state.velocity_y = state.zoom_velocity = 0.0
        state.zoom_pending_seconds = 0.0
        state.is_scene_cut = False
        return cw, ch

    cw, ch = current_crop_size(base_crop_w, base_crop_h, state.zoom, frame_w, frame_h)
    confidence = clamp(observation.confidence, 0.15, 1.0)
    alpha = 1.0 - math.exp(-dt / (args.target_time / (0.55 + 0.65 * confidence)))
    x, y = observation.center_x, observation.center_y

    if abs(x - state.crop_center_x) <= cw * args.dead_zone:
        x = state.crop_center_x
        state.target_center_x = x
        state.velocity_x = 0.0
    if abs(y - state.crop_center_y) <= ch * args.dead_zone:
        y = state.crop_center_y
        state.target_center_y = y
        state.velocity_y = 0.0

    if abs(desired_zoom - state.zoom) <= args.zoom_dead_zone:
        state.zoom_pending_seconds = 0.0
    else:
        state.zoom_pending_seconds += dt

    if args.fixed_zoom is not None:
        state.zoom = state.target_zoom = desired_zoom
        state.zoom_velocity = 0.0
    elif state.zoom_pending_seconds + 1e-9 >= args.zoom_hold_seconds:
        state.target_zoom = lerp(state.target_zoom, desired_zoom, alpha)
        state.zoom, state.zoom_velocity = critically_damped_step(
            state.zoom,
            state.target_zoom,
            state.zoom_velocity,
            dt,
            args.zoom_time,
            args.zoom_speed,
        )
    else:
        state.target_zoom = state.zoom
        state.zoom_velocity = 0.0

    bounded = clamp(state.zoom, args.min_zoom, args.max_zoom)
    if bounded != state.zoom:
        state.zoom_velocity = 0.0
        state.zoom = bounded

    cw, ch = current_crop_size(base_crop_w, base_crop_h, state.zoom, frame_w, frame_h)
    state.target_center_x = clamp(
        lerp(state.target_center_x, x, alpha), cw / 2.0, frame_w - cw / 2.0
    )
    state.target_center_y = clamp(
        lerp(state.target_center_y, y, alpha), ch / 2.0, frame_h - ch / 2.0
    )

    nx, state.velocity_x = critically_damped_step(
        state.crop_center_x,
        state.target_center_x,
        state.velocity_x,
        dt,
        args.pan_time,
        frame_w * args.pan_speed_x * (0.35 + 0.65 * confidence),
    )
    ny, state.velocity_y = critically_damped_step(
        state.crop_center_y,
        state.target_center_y,
        state.velocity_y,
        dt,
        args.pan_time,
        frame_h * args.pan_speed_y * (0.35 + 0.65 * confidence),
    )
    state.crop_center_x = clamp(nx, cw / 2.0, frame_w - cw / 2.0)
    state.crop_center_y = clamp(ny, ch / 2.0, frame_h - ch / 2.0)
    if nx != state.crop_center_x:
        state.velocity_x = 0.0
    if ny != state.crop_center_y:
        state.velocity_y = 0.0
    return cw, ch


def crop_frame(
    frame: np.ndarray,
    center_x: float,
    center_y: float,
    crop_w: int,
    crop_h: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    frame_h, frame_w = frame.shape[:2]
    left = int(round(center_x - crop_w / 2.0))
    top = int(round(center_y - crop_h / 2.0))

    left = int(clamp(left, 0, frame_w - crop_w))
    top = int(clamp(top, 0, frame_h - crop_h))
    right = left + crop_w
    bottom = top + crop_h
    return frame[top:bottom, left:right], (left, top, right, bottom)


def reset_for_new_scene(
    state: CameraState,
    frame_w: int,
    frame_h: int,
    min_zoom: float,
) -> None:
    state.crop_center_x = frame_w / 2.0
    state.crop_center_y = frame_h / 2.0
    state.zoom = min_zoom
    state.target_center_x = frame_w / 2.0
    state.target_center_y = frame_h / 2.0
    state.target_zoom = min_zoom
    state.velocity_x = 0.0
    state.velocity_y = 0.0
    state.zoom_velocity = 0.0
    state.tracked_id = None
    state.tracked_cls_id = None
    state.missed_frames = 0
    state.lock_track_id = None
    state.lock_cls_id = None
    state.frames_since_subject_switch = 0
    state.last_framing_cx = None
    state.last_framing_cy = None
    state.last_subject_key = None
    state.framing_vx = 0.0
    state.framing_vy = 0.0
    state.is_scene_cut = True
    state.two_person_active_frames = 0
    state.zoom_pending_frames = 0
    state.zoom_pending_seconds = 0.0
    state.motion_history.clear()


def pair_fits(pair, base_w: int, base_h: int, zoom: float) -> bool:
    """Reject pairs whose padded focus regions cannot fit in the available crop."""
    a, b = (derive_candidate_focus_bounds(c) for c in pair)
    width = max(a[2], b[2]) - min(a[0], b[0])
    height = max(a[3], b[3]) - min(a[1], b[1])
    return (
        width + 2 * max(width * 0.12, 16) <= base_w / zoom
        and height + 2 * max(height * 0.16, 20) <= base_h / zoom
    )


def regression_velocity(history):
    """Robust multi-frame velocity in pixels/second using median pairwise slopes."""
    if len(history) < 3:
        return 0.0, 0.0
    a = np.asarray(history, dtype=np.float64)
    # Use k=1 keyword arg for upper triangle offset (second positional arg in np.triu_indices is m, not k)
    i, j = np.triu_indices(len(a), k=1)
    dt = a[j, 0] - a[i, 0]
    valid = dt > 1e-6
    if not valid.any():
        return 0.0, 0.0
    slopes = (a[j[valid], 1:] - a[i[valid], 1:]) / dt[valid, None]
    return tuple(np.median(slopes, axis=0))
