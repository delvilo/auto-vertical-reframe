import argparse
import json
import math
import numpy as np
import pytest
from typing import Any, Optional

import auto_reframe
from auto_reframe import (
    Candidate,
    CameraObservation,
    CameraState,
    HandcraftedSaliencyHelper,
    InlineSceneDetector,
    SubjectRankingModel,
    active_speaker_bonus,
    apply_camera_motion,
    apply_preset,
    build_pair_observation,
    build_single_subject_observation,
    build_video_filters,
    clamp,
    choose_subject,
    choose_two_person_pair,
    compute_base_crop,
    compute_observation_from_bounds,
    critically_damped_step,
    crop_frame,
    current_crop_size,
    derive_candidate_focus_bounds,
    get_track_id,
    lerp,
    load_speaker_segments,
    mask_stats_from_binary_mask,
    pair_fits,
    regression_velocity,
    reset_for_new_scene,
)


def make_candidate(
    cls_id: int = 0,
    cls_name: str = "person",
    track_id: Optional[int] = None,
    conf: float = 0.9,
    x1: float = 100.0,
    y1: float = 100.0,
    x2: float = 200.0,
    y2: float = 500.0,
    cx: float = 150.0,
    cy: float = 300.0,
    width: float = 100.0,
    height: float = 400.0,
    area: float = 40000.0,
    mask_area: float = 40000.0,
    mask_cx: float = 150.0,
    mask_cy: float = 300.0,
    mask_top_y: float = 100.0,
    framing_cx: float = 150.0,
    framing_cy: float = 300.0,
    face_box: Optional[tuple[float, float, float, float]] = None,
    score: float = 0.0,
    rank_confidence: float = 0.0,
    eye_y: Optional[float] = None,
    chin_y: Optional[float] = None,
    body_top_y: Optional[float] = None,
    body_bottom_y: Optional[float] = None,
    body_cx: Optional[float] = None,
    shoulder_span: Optional[float] = None,
    body_bottom_confident: bool = False,
    body_min_x: Optional[float] = None,
    body_max_x: Optional[float] = None,
    salient_x1: Optional[float] = None,
    salient_y1: Optional[float] = None,
    salient_x2: Optional[float] = None,
    salient_y2: Optional[float] = None,
    saliency_confidence: float = 0.0,
) -> Candidate:
    return Candidate(
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
        face_box=face_box,
        score=score,
        rank_confidence=rank_confidence,
        eye_y=eye_y,
        chin_y=chin_y,
        body_top_y=body_top_y,
        body_bottom_y=body_bottom_y,
        body_cx=body_cx,
        shoulder_span=shoulder_span,
        body_bottom_confident=body_bottom_confident,
        body_min_x=body_min_x,
        body_max_x=body_max_x,
        salient_x1=salient_x1,
        salient_y1=salient_y1,
        salient_x2=salient_x2,
        salient_y2=salient_y2,
        saliency_confidence=saliency_confidence,
    )


def make_camera_state(
    crop_center_x: float = 960.0,
    crop_center_y: float = 540.0,
    zoom: float = 1.0,
    target_center_x: float = 960.0,
    target_center_y: float = 540.0,
    target_zoom: float = 1.0,
) -> CameraState:
    return CameraState(
        crop_center_x=crop_center_x,
        crop_center_y=crop_center_y,
        zoom=zoom,
        target_center_x=target_center_x,
        target_center_y=target_center_y,
        target_zoom=target_zoom,
    )


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-5, 0, 10) == 0
    assert clamp(15, 0, 10) == 10
    assert clamp(3.14, 0.0, 5.0) == 3.14
    assert clamp(0, 0, 0) == 0


def test_lerp():
    assert lerp(10.0, 20.0, 0.0) == 10.0
    assert lerp(10.0, 20.0, 1.0) == 20.0
    assert lerp(10.0, 20.0, 0.5) == 15.0
    assert lerp(0.0, 100.0, -0.1) == pytest.approx(-10.0)
    assert lerp(0.0, 100.0, 1.2) == pytest.approx(120.0)


def test_compute_base_crop():
    # 16:9 source (1920x1080) reframed to 9:16 target aspect
    crop_w, crop_h = compute_base_crop(1920, 1080, aspect_w=9, aspect_h=16)
    assert crop_h == 1080
    assert crop_w == 608

    # Taller source reframed to 16:9 target aspect
    crop_w2, crop_h2 = compute_base_crop(1080, 1920, aspect_w=16, aspect_h=9)
    assert crop_w2 == 1080
    assert crop_h2 == 608

    # Equal aspect ratio
    crop_w3, crop_h3 = compute_base_crop(1000, 1000, aspect_w=1, aspect_h=1)
    assert crop_w3 == 1000
    assert crop_h3 == 1000


def test_current_crop_size():
    cw, ch = current_crop_size(608, 1080, zoom=1.0, frame_w=1920, frame_h=1080)
    assert cw % 2 == 0
    assert ch % 2 == 0
    assert cw == 608
    assert ch == 1080

    # Zoomed in 2x
    cw_z, ch_z = current_crop_size(600, 1000, zoom=2.0, frame_w=1920, frame_h=1080)
    assert cw_z == 300
    assert ch_z == 500

    # Frame bounds constraint
    cw_clamped, ch_clamped = current_crop_size(1000, 1000, zoom=0.5, frame_w=800, frame_h=800)
    assert cw_clamped <= 800
    assert ch_clamped <= 800


def test_critically_damped_step():
    # Zero delta time -> no movement
    val, vel = critically_damped_step(current=0.0, target=100.0, velocity=0.0, dt=0.0, tau=0.5, max_speed=500.0)
    assert val == 0.0
    assert vel == 0.0

    # Step towards target
    val, vel = critically_damped_step(current=0.0, target=100.0, velocity=0.0, dt=0.1, tau=0.5, max_speed=500.0)
    assert val > 0.0
    assert val < 100.0

    # Max speed limiting check
    val_limited, vel_limited = critically_damped_step(current=0.0, target=10000.0, velocity=0.0, dt=0.1, tau=0.1, max_speed=10.0)
    assert abs(vel_limited) <= 10.0


def test_regression_velocity():
    # Insufficient history
    assert regression_velocity([]) == (0.0, 0.0)
    assert regression_velocity([(0.0, 10.0, 20.0)]) == (0.0, 0.0)

    # Constant velocity sequence: positions at t=0, t=1, t=2
    # Moving at (10.0, 5.0) per second
    history = [
        (0.0, 0.0, 0.0),
        (1.0, 10.0, 5.0),
        (2.0, 20.0, 10.0),
    ]
    vx, vy = regression_velocity(history)
    assert vx == pytest.approx(10.0)
    assert vy == pytest.approx(5.0)


def test_get_track_id():
    class MockTensorItem:
        def __init__(self, val: int):
            self.val = val

        def item(self) -> int:
            return self.val

    class DummyBox:
        def __init__(self, box_id: Any):
            self.id = box_id

    assert get_track_id(DummyBox([MockTensorItem(42)])) == 42
    assert get_track_id(DummyBox(None)) is None
    assert get_track_id(DummyBox([])) is None


def test_mask_stats_from_binary_mask():
    # Empty mask
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    area, cx, cy, top_y = mask_stats_from_binary_mask(empty_mask)
    assert area == 0.0

    # Non-empty mask: 20x20 filled box from y: 10..30, x: 20..40
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 20:40] = 1
    area, cx, cy, top_y = mask_stats_from_binary_mask(mask)
    assert area == 400.0
    assert cx == pytest.approx(29.5)
    assert cy == pytest.approx(19.5)
    assert top_y == 10.0


def test_pair_fits():
    cand1 = make_candidate(x1=100.0, y1=100.0, x2=200.0, y2=300.0, mask_top_y=100.0)
    cand2 = make_candidate(x1=210.0, y1=110.0, x2=300.0, y2=310.0, mask_top_y=110.0)

    # Two close candidates fit in base_w=608, base_h=1080
    assert pair_fits((cand1, cand2), base_w=608, base_h=1080, zoom=1.0) is True

    # Candidates too far apart
    cand_far = make_candidate(x1=800.0, y1=100.0, x2=900.0, y2=300.0, mask_top_y=100.0)
    assert pair_fits((cand1, cand_far), base_w=608, base_h=1080, zoom=1.0) is False


def test_derive_candidate_focus_bounds():
    # Non-person candidate
    cand_car = make_candidate(
        cls_id=2, cls_name="car", conf=0.9, x1=100.0, y1=200.0, x2=300.0, y2=400.0, mask_top_y=200.0
    )
    left, top, right, bottom = derive_candidate_focus_bounds(cand_car)
    assert left < cand_car.x1
    assert right > cand_car.x2
    assert top < cand_car.y1
    assert bottom > cand_car.y2

    # Person candidate with body points
    cand_person = make_candidate(
        cls_id=0,
        cls_name="person",
        conf=0.95,
        x1=100.0,
        y1=100.0,
        x2=200.0,
        y2=500.0,
        body_min_x=110.0,
        body_max_x=190.0,
        body_top_y=100.0,
        body_bottom_y=480.0,
        body_bottom_confident=True,
        eye_y=130.0,
    )
    left, top, right, bottom = derive_candidate_focus_bounds(cand_person)
    assert left <= cand_person.x1
    assert right >= cand_person.x2
    assert top < 100.0  # Headroom added


def test_compute_observation_from_bounds():
    bounds = (200.0, 100.0, 400.0, 500.0)
    obs = compute_observation_from_bounds(
        bounds=bounds,
        confidence=0.8,
        base_crop_w=608,
        base_crop_h=1080,
        frame_w=1920,
        frame_h=1080,
        min_zoom=1.0,
        max_zoom=1.8,
    )
    assert isinstance(obs, CameraObservation)
    assert obs.confidence == 0.8
    assert 1.0 <= obs.zoom <= 1.8
    assert obs.center_x == pytest.approx(300.0)


def test_build_single_subject_observation():
    cand = make_candidate(
        cls_id=0,
        cls_name="person",
        conf=0.9,
        x1=500.0,
        y1=200.0,
        x2=700.0,
        y2=800.0,
        eye_y=250.0,
    )
    obs = build_single_subject_observation(
        subject=cand,
        base_crop_w=608,
        base_crop_h=1080,
        frame_w=1920,
        frame_h=1080,
        min_zoom=1.0,
        max_zoom=1.8,
    )
    assert obs.confidence >= 0.2
    assert obs.confidence <= 1.0


def test_build_pair_observation():
    cand1 = make_candidate(
        cls_id=0,
        cls_name="person",
        conf=0.9,
        x1=300.0,
        y1=200.0,
        x2=450.0,
        y2=800.0,
        eye_y=240.0,
    )
    cand2 = make_candidate(
        cls_id=0,
        cls_name="person",
        conf=0.85,
        x1=600.0,
        y1=220.0,
        x2=750.0,
        y2=810.0,
        eye_y=260.0,
    )
    obs = build_pair_observation(
        first=cand1,
        second=cand2,
        base_crop_w=608,
        base_crop_h=1080,
        frame_w=1920,
        frame_h=1080,
        min_zoom=1.0,
        max_zoom=1.8,
    )
    assert obs.confidence == pytest.approx((0.9 + 0.85) / 2.0)


def test_crop_frame():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cropped, (l, t, r, b) = crop_frame(frame, center_x=960.0, center_y=540.0, crop_w=608, crop_h=1080)
    assert cropped.shape == (1080, 608, 3)
    assert l >= 0 and r <= 1920
    assert t >= 0 and b <= 1080


def test_reset_for_new_scene():
    state = make_camera_state(crop_center_x=100.0, crop_center_y=100.0, zoom=1.5)
    state.tracked_id = 5
    state.missed_frames = 10
    reset_for_new_scene(state, frame_w=1920, frame_h=1080, min_zoom=1.0)

    assert state.crop_center_x == 960.0
    assert state.crop_center_y == 540.0
    assert state.zoom == 1.0
    assert state.tracked_id is None
    assert state.missed_frames == 0
    assert state.is_scene_cut is True


def test_apply_camera_motion():
    args = argparse.Namespace(
        runtime_fps=30.0,
        fixed_zoom=None,
        min_zoom=1.0,
        max_zoom=1.8,
        target_time=0.3,
        dead_zone=0.06,
        zoom_dead_zone=0.04,
        zoom_hold_seconds=0.4,
        zoom_time=0.3,
        zoom_speed=1.0,
        pan_time=0.3,
        pan_speed_x=1.0,
        pan_speed_y=1.0,
    )
    state = make_camera_state(crop_center_x=960.0, crop_center_y=540.0, zoom=1.0)
    obs = CameraObservation(center_x=1000.0, center_y=550.0, zoom=1.2, confidence=0.9)

    # Initial cut test -> snaps immediately
    state.is_scene_cut = True
    cw, ch = apply_camera_motion(state, obs, args, base_crop_w=608, base_crop_h=1080, frame_w=1920, frame_h=1080)
    expected_cw, expected_ch = current_crop_size(608, 1080, zoom=1.2, frame_w=1920, frame_h=1080)
    assert cw == expected_cw
    assert ch == expected_ch
    assert state.crop_center_x == 1000.0
    assert state.is_scene_cut is False

    # Second step -> dead zone test
    obs_small_move = CameraObservation(center_x=1005.0, center_y=550.0, zoom=1.2, confidence=0.9)
    apply_camera_motion(state, obs_small_move, args, base_crop_w=608, base_crop_h=1080, frame_w=1920, frame_h=1080)
    assert state.velocity_x == 0.0


def test_subject_ranking_model():
    model = SubjectRankingModel()
    score_person = model.predict(
        cls_name="person",
        conf=0.9,
        mask_area=50000.0,
        frame_area=2000000.0,
        dist_center=100.0,
        frame_diag=2200.0,
        has_face=True,
        has_pose=True,
        saliency_confidence=0.8,
        tracking_match=True,
        lock_match=False,
        speaker_active=True,
    )
    score_car = model.predict(
        cls_name="car",
        conf=0.5,
        mask_area=10000.0,
        frame_area=2000000.0,
        dist_center=500.0,
        frame_diag=2200.0,
        has_face=False,
        has_pose=False,
        saliency_confidence=0.0,
        tracking_match=False,
        lock_match=False,
        speaker_active=False,
    )
    assert score_person > score_car


def test_choose_subject():
    cand1 = make_candidate(cls_id=0, cls_name="person", conf=0.9, track_id=1, score=10.0)
    cand2 = make_candidate(cls_id=0, cls_name="person", conf=0.8, track_id=2, score=12.0)

    # candidates are sorted descending by score
    sorted_candidates = [cand2, cand1]

    state = make_camera_state()

    # Empty candidates
    assert choose_subject([], state, lock_first_subject=False, min_subject_hold_frames=10, switch_score_threshold=1.2) is None

    # Locked mode test
    state.lock_track_id = 1
    state.lock_cls_id = 0
    chosen_locked = choose_subject(sorted_candidates, state, lock_first_subject=True, min_subject_hold_frames=10, switch_score_threshold=1.2)
    assert chosen_locked == cand1

    # Normal mode select best score
    state.lock_track_id = None
    chosen = choose_subject(sorted_candidates, state, lock_first_subject=False, min_subject_hold_frames=10, switch_score_threshold=1.2)
    assert chosen == cand2


def test_choose_two_person_pair():
    p1 = make_candidate(cls_id=0, cls_name="person", conf=0.9, x1=0, y1=0, x2=10, y2=10, score=10.0)
    p2_strong = make_candidate(cls_id=0, cls_name="person", conf=0.85, x1=20, y1=0, x2=30, y2=10, score=9.0)
    p2_weak = make_candidate(cls_id=0, cls_name="person", conf=0.3, x1=20, y1=0, x2=30, y2=10, score=2.0)

    # Strong second person -> pair selected
    pair = choose_two_person_pair([p1, p2_strong], threshold=0.7, currently_active=False)
    assert pair == (p1, p2_strong)

    # Weak second person -> pair rejected
    pair_none = choose_two_person_pair([p1, p2_weak], threshold=0.7, currently_active=False)
    assert pair_none is None


def test_apply_preset():
    args = argparse.Namespace(
        preset="talking_head",
        classes=None,
        min_zoom=None,
        max_zoom=None,
        fixed_zoom=None,
        dead_zone=None,
        max_step_x=None,
        max_step_y=None,
        pan_time=None,
        zoom_time=None,
        target_time=None,
        pan_speed_x=None,
        pan_speed_y=None,
        zoom_speed=None,
        motion_response=0.08,
        motion_damping=0.92,
        zoom_response=0.05,
        zoom_damping=0.90,
        target_alpha=0.2,
        zoom_alpha=0.1,
    )
    res = apply_preset(args)
    assert res.classes == ["person"]
    assert res.min_zoom == 1.00
    assert res.max_zoom == 1.85


def test_speaker_segments_and_bonus(tmp_path):
    # Nonexistent file returns []
    assert load_speaker_segments(str(tmp_path / "nonexistent.json")) == []

    # Valid JSON
    json_content = {
        "segments": [
            {"track_id": 1, "scene_index": 0, "start": 0.0, "end": 2.0}
        ]
    }
    json_file = tmp_path / "diarization.json"
    json_file.write_text(json.dumps(json_content), encoding="utf-8")

    segments = load_speaker_segments(str(json_file))
    assert len(segments) == 1

    # Active speaker bonus check
    # frame_idx = 30, fps = 30 -> timestamp = 29/30 ~ 0.967s (within 0.0 .. 2.0)
    bonus_active = active_speaker_bonus(frame_idx=30, fps=30.0, segments=segments, track_id=1, scene_index=0)
    assert bonus_active == 1.08

    bonus_inactive = active_speaker_bonus(frame_idx=100, fps=30.0, segments=segments, track_id=1, scene_index=0)
    assert bonus_inactive == 1.0


def test_build_video_filters():
    assert build_video_filters(post_restore=False) is None
    filters = build_video_filters(post_restore=True)
    assert "hqdn3d" in filters
    assert "unsharp" in filters


def test_inline_scene_detector():
    detector = InlineSceneDetector(min_frames=5, threshold=3.0)

    # First frame always registers as cut
    f1 = np.full((1080, 1920, 3), 100, dtype=np.uint8)
    assert detector.update(f1, frame_idx=0) is True

    # Same frame soon after -> no cut
    f2 = np.full((1080, 1920, 3), 100, dtype=np.uint8)
    assert detector.update(f2, frame_idx=1) is False


def test_handcrafted_saliency_helper():
    helper = HandcraftedSaliencyHelper()
    frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    saliency_map = helper.compute_map(frame)
    assert saliency_map.shape == (100, 100)
    assert saliency_map.dtype == np.float32
    assert helper.get_telemetry()["active_backend"] == "handcrafted"
