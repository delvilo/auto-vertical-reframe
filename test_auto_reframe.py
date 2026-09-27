import math
import pytest
from auto_reframe import Candidate, derive_candidate_focus_bounds, pair_fits


def make_candidate(
    cls_id: int = 0,
    cls_name: str = "person",
    track_id: int = 1,
    conf: float = 0.9,
    x1: float = 100.0,
    y1: float = 100.0,
    x2: float = 200.0,
    y2: float = 300.0,
    mask_top_y: float = 100.0,
    face_box=None,
    eye_y=None,
    body_top_y=None,
    body_bottom_y=None,
    body_min_x=None,
    body_max_x=None,
    body_bottom_confident=False,
    salient_box=None,
) -> Candidate:
    width = x2 - x1
    height = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    area = width * height

    salient_x1, salient_y1, salient_x2, salient_y2 = (
        salient_box if salient_box else (None, None, None, None)
    )

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
        mask_area=area,
        mask_cx=cx,
        mask_cy=cy,
        mask_top_y=mask_top_y,
        framing_cx=cx,
        framing_cy=cy,
        face_box=face_box,
        eye_y=eye_y,
        body_top_y=body_top_y,
        body_bottom_y=body_bottom_y,
        body_min_x=body_min_x,
        body_max_x=body_max_x,
        body_bottom_confident=body_bottom_confident,
        salient_x1=salient_x1,
        salient_y1=salient_y1,
        salient_x2=salient_x2,
        salient_y2=salient_y2,
    )


def test_pair_fits_compact_pair_fits():
    """Two compact candidates close together should fit within 1080x1920 at 1.0x zoom."""
    c1 = make_candidate(track_id=1, x1=200, y1=200, x2=350, y2=500)
    c2 = make_candidate(track_id=2, x1=400, y1=200, x2=550, y2=500)
    pair = (c1, c2)

    assert pair_fits(pair, base_w=1080, base_h=1920, zoom=1.0) is True


def test_pair_fits_horizontal_exceeded():
    """Two candidates separated widely horizontally should fail to fit."""
    c1 = make_candidate(track_id=1, x1=50, y1=200, x2=150, y2=500)
    c2 = make_candidate(track_id=2, x1=1000, y1=200, x2=1100, y2=500)
    pair = (c1, c2)

    assert pair_fits(pair, base_w=1080, base_h=1920, zoom=1.0) is False


def test_pair_fits_vertical_exceeded():
    """Two candidates separated widely vertically should fail to fit."""
    c1 = make_candidate(track_id=1, x1=200, y1=50, x2=400, y2=150)
    c2 = make_candidate(track_id=2, x1=200, y1=1700, x2=400, y2=1800)
    pair = (c1, c2)

    assert pair_fits(pair, base_w=1080, base_h=1920, zoom=1.0) is False


def test_pair_fits_zoom_effect():
    """A pair that fits at zoom 1.0x should fail when zoom is increased to 2.0x."""
    c1 = make_candidate(track_id=1, x1=200, y1=200, x2=400, y2=600)
    c2 = make_candidate(track_id=2, x1=450, y1=200, x2=650, y2=600)
    pair = (c1, c2)

    # Base w/zoom = 1080/1.0 = 1080; width ~ 450 + padding <= 1080
    assert pair_fits(pair, base_w=1080, base_h=1920, zoom=1.0) is True

    # Base w/zoom = 1080/2.0 = 540; width ~ 450 + padding (~558) > 540
    assert pair_fits(pair, base_w=1080, base_h=1920, zoom=2.0) is False


def test_pair_fits_minimum_padding():
    """Verify that minimum padding fallbacks (16px x, 20px y) are applied for small bounding boxes."""
    # Tiny boxes: width = 10px, height = 10px
    c1 = make_candidate(cls_name="dog", x1=100, y1=100, x2=110, y2=110, mask_top_y=100)
    c2 = make_candidate(cls_name="dog", x1=100, y1=100, x2=110, y2=110, mask_top_y=100)
    pair = (c1, c2)

    a, b = (derive_candidate_focus_bounds(c) for c in pair)
    union_w = max(a[2], b[2]) - min(a[0], b[0])
    union_h = max(a[3], b[3]) - min(a[1], b[1])

    required_w = union_w + 2 * 16
    required_h = union_h + 2 * 20

    base_w = math.ceil(required_w)
    base_h = math.ceil(required_h)

    assert pair_fits(pair, base_w=base_w, base_h=base_h, zoom=1.0) is True
    assert pair_fits(pair, base_w=base_w - 1, base_h=base_h, zoom=1.0) is False


def test_pair_fits_with_face_boxes():
    """Candidates with face boxes expand focus bounds appropriately and affect pair_fits."""
    c1 = make_candidate(track_id=1, x1=200, y1=200, x2=400, y2=600, face_box=(250, 220, 350, 320))
    c2 = make_candidate(track_id=2, x1=500, y1=200, x2=700, y2=600, face_box=(550, 220, 650, 320))
    pair = (c1, c2)

    assert pair_fits(pair, base_w=1080, base_h=1920, zoom=1.0) is True


def test_pair_fits_with_body_pose():
    """Candidates with body pose landmarks are accurately evaluated by pair_fits."""
    c1 = make_candidate(
        track_id=1,
        x1=150,
        y1=100,
        x2=350,
        y2=800,
        body_min_x=160,
        body_max_x=340,
        body_top_y=120,
        body_bottom_y=780,
        eye_y=150,
        body_bottom_confident=True,
    )
    c2 = make_candidate(
        track_id=2,
        x1=400,
        y1=100,
        x2=600,
        y2=800,
        body_min_x=410,
        body_max_x=590,
        body_top_y=120,
        body_bottom_y=780,
        eye_y=150,
        body_bottom_confident=True,
    )
    pair = (c1, c2)

    assert pair_fits(pair, base_w=1080, base_h=1920, zoom=1.0) is True


def test_pair_fits_with_saliency_boxes():
    """Candidates with saliency regions expand focus bounds and affect pair_fits result."""
    c1 = make_candidate(
        track_id=1,
        x1=100,
        y1=100,
        x2=200,
        y2=300,
        salient_box=(50, 50, 250, 350),
    )
    c2 = make_candidate(
        track_id=2,
        x1=700,
        y1=100,
        x2=800,
        y2=300,
        salient_box=(650, 50, 850, 350),
    )
    pair = (c1, c2)

    # Fits when base width is 1200, fails when base width is 900
    assert pair_fits(pair, base_w=1200, base_h=1920, zoom=1.0) is True
    assert pair_fits(pair, base_w=900, base_h=1920, zoom=1.0) is False


def test_pair_fits_non_person_classes():
    """Pair fitting works correctly for non-person classes (e.g. pets or cars)."""
    cat = make_candidate(cls_name="cat", x1=100, y1=500, x2=250, y2=650)
    dog = make_candidate(cls_name="dog", x1=300, y1=500, x2=500, y2=700)
    pair = (cat, dog)

    assert pair_fits(pair, base_w=1080, base_h=1920, zoom=1.0) is True


def test_pair_fits_exact_boundary():
    """Verify behavior right at the boundary threshold."""
    c1 = make_candidate(cls_name="dog", x1=100, y1=100, x2=300, y2=500)
    c2 = make_candidate(cls_name="dog", x1=300, y1=100, x2=500, y2=500)
    pair = (c1, c2)

    a, b = (derive_candidate_focus_bounds(c) for c in pair)
    width = max(a[2], b[2]) - min(a[0], b[0])
    height = max(a[3], b[3]) - min(a[1], b[1])

    padded_w = width + 2 * max(width * 0.12, 16)
    padded_h = height + 2 * max(height * 0.16, 20)

    base_w = math.ceil(padded_w)
    base_h = math.ceil(padded_h)

    # Exactly required dimension -> True
    assert pair_fits(pair, base_w=base_w, base_h=base_h, zoom=1.0) is True
    # 1 pixel smaller than required width -> False
    assert pair_fits(pair, base_w=base_w - 1, base_h=base_h, zoom=1.0) is False
