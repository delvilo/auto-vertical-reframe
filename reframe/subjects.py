from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional
import numpy as np
from reframe.contracts import Candidate, CameraState, FrameObservations
from reframe.geometry import clamp
from reframe.saliency.regions import extract_saliency_region

class SubjectRankingModel:
    """Prioritizes candidates using detection, saliency, pose, and track continuity."""

    def __init__(self) -> None:
        self.class_bias = {
            "person": 0.22,
            "dog": 0.12,
            "cat": 0.10,
            "car": 0.06,
            "bicycle": 0.02,
            "motorcycle": 0.02,
            "bus": 0.01,
            "truck": 0.01,
        }
        self.feature_weights = {
            "det_conf": 1.35,
            "mask_presence": 0.95,
            "center_affinity": 0.55,
            "head_presence": 0.24,
            "pose_presence": 0.34,
            "saliency_presence": 0.72,
            "saliency_conf": 0.78,
            "tracking_match": 1.05,
            "lock_match": 1.30,
            "speaker_active": 0.22,
            "size_logit": 0.26,
        }

    def predict(
        self,
        *,
        cls_name: str,
        conf: float,
        mask_area: float,
        frame_area: float,
        dist_center: float,
        frame_diag: float,
        has_head: bool,
        has_pose: bool,
        saliency_confidence: float,
        tracking_match: bool,
        lock_match: bool,
        speaker_active: bool,
    ) -> float:
        norm_area = clamp(mask_area / max(frame_area, 1.0), 0.0, 1.0)
        center_affinity = 1.0 - clamp(dist_center / max(frame_diag, 1.0), 0.0, 1.0)
        size_logit = math.log1p(norm_area * 250.0)

        weights = self.feature_weights
        score = (
            self.class_bias.get(cls_name, 0.0)
            + weights["det_conf"] * clamp(conf, 0.0, 1.0)
            + weights["mask_presence"] * math.sqrt(norm_area)
            + weights["center_affinity"] * center_affinity
            + (weights["head_presence"] if has_head else 0.0)
            + (weights["pose_presence"] if has_pose else 0.0)
            + (weights["saliency_presence"] if saliency_confidence > 0.0 else 0.0)
            + weights["saliency_conf"] * clamp(saliency_confidence, 0.0, 1.0)
            + (weights["tracking_match"] if tracking_match else 0.0)
            + (weights["lock_match"] if lock_match else 0.0)
            + (weights["speaker_active"] if speaker_active else 0.0)
            + weights["size_logit"] * size_logit
        )

        return score


def load_speaker_segments(path: str) -> list[dict]:
    if not path:
        return []
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "segments" in data:
            return data["segments"]
        if isinstance(data, list):
            return data
    except Exception as exc:
        logging.warning("Failed to load speaker diarization JSON: %s", exc)
    return []


def active_speaker_bonus(
    frame_idx: int,
    fps: float,
    segments: list[dict],
    track_id: Optional[int],
    scene_index: int,
) -> float:
    if track_id is None:
        return 1.0
    t = (frame_idx - 1) / max(fps, 1e-6)
    for seg in segments:
        if (
            seg.get("track_id") == track_id
            and seg.get("scene_index") == scene_index
            and seg.get("start", -1) <= t < seg.get("end", -1)
        ):
            return 1.08
    return 1.0


def choose_subject(
    candidates: list[Candidate],
    state: CameraState,
    lock_first_subject: bool,
    min_subject_hold_frames: int,
    switch_score_threshold: float,
    max_missed_frames: int = 0,
) -> Optional[Candidate]:
    if not candidates:
        return None

    if lock_first_subject and state.lock_track_id is not None:
        for candidate in candidates:
            if (
                candidate.track_id == state.lock_track_id
                and candidate.cls_id == state.lock_cls_id
            ):
                return candidate

    best = candidates[0]
    previous = None
    for candidate in candidates:
        if (
            candidate.track_id == state.tracked_id
            and candidate.cls_id == state.tracked_cls_id
        ):
            previous = candidate
            break

    if (
        previous is None
        and state.tracked_id is not None
        and state.missed_frames < max_missed_frames
    ):
        return None

    if previous is not None:
        if previous.track_id == best.track_id and previous.cls_id == best.cls_id:
            return previous

        if state.frames_since_subject_switch < min_subject_hold_frames:
            if previous.score * switch_score_threshold >= best.score:
                return previous

        if previous.score >= best.score * 0.92:
            return previous

    return best


def choose_two_person_pair(
    candidates: list[Candidate],
    threshold: float,
    currently_active: bool,
) -> Optional[tuple[Candidate, Candidate]]:
    """Evaluates two-person group shot with hysteresis to prevent rapid mode flipping."""
    persons = [c for c in candidates if c.cls_name == "person"]
    if len(persons) < 2:
        return None

    first, second = persons[0], persons[1]
    effective_threshold = (threshold - 0.12) if currently_active else threshold
    if second.score < first.score * effective_threshold:
        return None
    return first, second

def build_candidates(
    observations: FrameObservations,
    saliency_map: np.ndarray | None,
    ranking_model: SubjectRankingModel,
    class_names: dict[int, str],
    state: CameraState,
    fps: float,
    speaker_segments: list[dict],
    saliency_bounds: tuple[int, int, int, int] | None = None,
    tracking_max_age: float = 0.1,
) -> list[Candidate]:
    h, w = observations.frame.height, observations.frame.width
    frame_idx = observations.frame.frame_index
    frame_cx, frame_cy = w / 2.0, h / 2.0
    frame_area, frame_diag = float(h * w), math.hypot(w, h)
    candidates: list[Candidate] = []
    for track in observations.tracks:
        if not track.reliable_at(observations.frame, tracking_max_age):
            continue
        i, cls_id, conf, track_id = track.row_index, track.cls_id, track.confidence, track.track_id
        prediction_quality = track.prediction_confidence if track.source == "predicted" else 1.0
        conf *= prediction_quality
        x1, y1, x2, y2 = track.box
        try:
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            area = width * height
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            cls_name = class_names.get(cls_id, str(cls_id))

            mask_area, mask_cx, mask_cy, mask_top_y = (track.mask_statistics or (area, cx, cy, y1))

            framing_cx = mask_cx
            framing_cy = mask_cy
            salient_box = extract_saliency_region(
                saliency_map, (int(x1), int(y1), int(x2), int(y2)), frame_shape=(h, w),
                map_bounds=saliency_bounds,
            )
            pose_observation = observations.poses.get(i)
            pose_data = pose_observation.cues if pose_observation is not None else None
            head_box = pose_data.get("head_box") if pose_data else None

            if cls_name == "person":
                if head_box is not None:
                    fx1, fy1, fx2, fy2 = head_box
                    framing_cx = (fx1 + fx2) / 2.0
                    framing_cy = fy1 + (fy2 - fy1) * 0.42
                elif pose_data is not None and pose_data.get("body_cx") is not None:
                    framing_cx = float(pose_data["body_cx"])
                    eye_val = pose_data.get("eye_y")
                    shoulder_val = pose_data.get("shoulder_y")
                    if eye_val is not None and shoulder_val is not None:
                        framing_cy = (float(eye_val) + float(shoulder_val)) / 2.0
                    elif eye_val is not None:
                        framing_cy = float(eye_val)
                    elif shoulder_val is not None:
                        framing_cy = float(shoulder_val)
                    else:
                        framing_cy = mask_cy
                elif salient_box is not None:
                    framing_cx = salient_box[0]
                    framing_cy = salient_box[1]
                else:
                    framing_cx = mask_cx
                    framing_cy = mask_top_y + height * 0.28
            elif salient_box is not None:
                framing_cx = salient_box[0]
                framing_cy = salient_box[1]

            dist_center = math.hypot(
                framing_cx - frame_cx,
                framing_cy - frame_cy,
            )

            speaker_is_active = (
                active_speaker_bonus(
                    frame_idx,
                    fps,
                    speaker_segments,
                    track_id,
                    state.current_scene_index,
                )
                > 1.0
                if cls_name == "person"
                else False
            )
            tracking_match = (
                state.tracked_id is not None
                and track_id == state.tracked_id
                and cls_id == state.tracked_cls_id
            )
            lock_match = (
                state.lock_track_id is not None
                and track_id == state.lock_track_id
                and cls_id == state.lock_cls_id
            )
            score = ranking_model.predict(
                cls_name=cls_name,
                conf=conf,
                mask_area=mask_area * prediction_quality,
                frame_area=frame_area,
                dist_center=dist_center,
                frame_diag=frame_diag,
                has_head=head_box is not None,
                has_pose=bool(pose_data and pose_data.get("has_pose")),
                saliency_confidence=(
                    salient_box[6] if salient_box is not None else 0.0
                ),
                tracking_match=tracking_match,
                lock_match=lock_match,
                speaker_active=speaker_is_active,
            )

            eye_y_val = None
            body_top_val = None
            body_bottom_val = None
            body_cx_val = None
            shoulder_span_val = None
            body_bottom_confident_val = False
            body_min_x_val = None
            body_max_x_val = None
            if pose_data is not None:
                eye_y_val = pose_data.get("eye_y")
                body_top_val = pose_data.get("body_top_y")
                body_bottom_val = pose_data.get("body_bottom_y")
                body_cx_val = pose_data.get("body_cx")
                shoulder_span_val = pose_data.get("shoulder_span")
                body_bottom_confident_val = bool(
                    pose_data.get("body_bottom_confident", False)
                )
                body_min_x_val = pose_data.get("body_min_x")
                body_max_x_val = pose_data.get("body_max_x")

            candidates.append(
                Candidate(
                    cls_id=cls_id,
                    cls_name=cls_name,
                    track_id=track_id,
                    conf=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    cx=cx,
                    cy=cy,
                    width=width,
                    height=height,
                    area=area,
                    mask_area=mask_area,
                    mask_cx=mask_cx,
                    mask_cy=mask_cy,
                    mask_top_y=mask_top_y,
                    framing_cx=framing_cx,
                    framing_cy=framing_cy,
                    head_box=head_box,
                    has_pose=bool(pose_data and pose_data.get("has_pose")),
                    score=score,
                    eye_y=eye_y_val,
                    body_top_y=body_top_val,
                    body_bottom_y=body_bottom_val,
                    body_cx=body_cx_val,
                    shoulder_span=shoulder_span_val,
                    body_bottom_confident=body_bottom_confident_val,
                    body_min_x=body_min_x_val,
                    body_max_x=body_max_x_val,
                    salient_x1=(salient_box[2] if salient_box is not None else None),
                    salient_y1=(salient_box[3] if salient_box is not None else None),
                    salient_x2=(salient_box[4] if salient_box is not None else None),
                    salient_y2=(salient_box[5] if salient_box is not None else None),
                    saliency_confidence=(
                        salient_box[6] if salient_box is not None else 0.0
                    ),
                )
            )
        except Exception:
            logging.debug("Candidate cue extraction failed", exc_info=True)
            continue

    candidates.sort(key=lambda c: c.score, reverse=True)
    if candidates:
        scores = np.asarray([c.score for c in candidates], dtype=np.float64)
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        for candidate, relative in zip(candidates, weights):
            absolute = 1.0 - math.exp(-max(0.0, candidate.score) / 4.0)
            candidate.rank_confidence = float(0.5 * absolute + 0.5 * relative)
    return candidates
