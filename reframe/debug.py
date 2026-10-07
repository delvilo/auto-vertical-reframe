from __future__ import annotations

from typing import Optional
import cv2
import numpy as np
from reframe.contracts import Candidate, CameraState

def draw_debug(
    frame: np.ndarray,
    crop_rect: tuple[int, int, int, int],
    candidates: list[Candidate],
    subject: Optional[Candidate],
    pair: Optional[tuple[Candidate, Candidate]],
    state: CameraState,
    frame_idx: int,
    total_frames: int,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """Renders diagnostic overlay maintaining clean letterboxed aspect ratio."""
    left, top, right, bottom = crop_rect
    cw = max(1, right - left)
    ch = max(1, bottom - top)

    reframed_crop = cv2.resize(
        frame[top:bottom, left:right],
        (output_width, output_height),
        interpolation=cv2.INTER_LINEAR,
    )

    scale_x = output_width / float(cw)
    scale_y = output_height / float(ch)

    for candidate in candidates[:6]:
        color = (120, 120, 120)
        thickness = 1
        is_sub = (
            subject is not None
            and candidate.track_id == subject.track_id
            and candidate.cls_id == subject.cls_id
        )
        if is_sub:
            color = (0, 255, 0)
            thickness = 2

        vx1 = int((candidate.x1 - left) * scale_x)
        vy1 = int((candidate.y1 - top) * scale_y)
        vx2 = int((candidate.x2 - left) * scale_x)
        vy2 = int((candidate.y2 - top) * scale_y)
        cv2.rectangle(reframed_crop, (vx1, vy1), (vx2, vy2), color, thickness)

        if candidate.head_box is not None:
            fx1, fy1, fx2, fy2 = candidate.head_box
            vfx1 = int((fx1 - left) * scale_x)
            vfy1 = int((fy1 - top) * scale_y)
            vfx2 = int((fx2 - left) * scale_x)
            vfy2 = int((fy2 - top) * scale_y)
            cv2.rectangle(reframed_crop, (vfx1, vfy1), (vfx2, vfy2), (255, 128, 0), 2)

    if pair is not None:
        for candidate in pair:
            pcx = int((candidate.framing_cx - left) * scale_x)
            pcy = int((candidate.framing_cy - top) * scale_y)
            cv2.circle(reframed_crop, (pcx, pcy), 8, (255, 0, 255), -1)

    hud_sub = reframed_crop[10:161, 10:451]
    hud_bg = hud_sub.copy()
    cv2.rectangle(hud_bg, (0, 0), (440, 150), (20, 20, 20), -1)
    cv2.addWeighted(hud_bg, 0.75, hud_sub, 0.25, 0, hud_sub)

    lines = [
        f"Frame: {frame_idx}/{total_frames if total_frames > 0 else '?'}",
        f"Scene: {state.current_scene_index} | Zoom: {state.zoom:.2f}x",
        f"Tracked ID: {state.tracked_id} ({state.tracked_cls_id})",
        f"Hold Count: {state.frames_since_subject_switch} | Missed: {state.missed_frames}",
    ]
    y = 38
    for line in lines:
        cv2.putText(
            reframed_crop,
            line,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        y += 28

    return reframed_crop


