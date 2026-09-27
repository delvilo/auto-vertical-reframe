"""Unit tests for auto_reframe.py module."""

import pytest
from auto_reframe import compute_base_crop


def test_compute_base_crop_default_aspect_ratio_landscape():
    """Test compute_base_crop with default 9:16 aspect ratio on landscape inputs."""
    # 1080p widescreen (16:9): crop height is full frame height (1080),
    # crop width is round(1080 * 9 / 16) = round(607.5) = 608.
    assert compute_base_crop(1920, 1080) == (608, 1080)

    # 720p widescreen (16:9): crop height = 720, crop width = round(720 * 9 / 16) = 405.
    assert compute_base_crop(1280, 720) == (405, 720)

    # 4K UHD widescreen (16:9): crop height = 2160, crop width = round(2160 * 9 / 16) = 1215.
    assert compute_base_crop(3840, 2160) == (1215, 2160)


def test_compute_base_crop_default_aspect_ratio_square():
    """Test compute_base_crop with default 9:16 aspect ratio on square inputs."""
    # 1080x1080 square source: frame_ratio (1.0) >= target_ratio (9/16 = 0.5625)
    # crop_h = 1080, crop_w = round(1080 * 9 / 16) = 608.
    assert compute_base_crop(1080, 1080) == (608, 1080)

    # 500x500 square source: crop_h = 500, crop_w = round(500 * 9 / 16) = 281.
    assert compute_base_crop(500, 500) == (281, 500)


def test_compute_base_crop_default_aspect_ratio_matching_vertical():
    """Test compute_base_crop when source frame ratio matches target ratio (9:16)."""
    assert compute_base_crop(1080, 1920) == (1080, 1920)
    assert compute_base_crop(720, 1280) == (720, 1280)


def test_compute_base_crop_default_aspect_ratio_ultra_tall_vertical():
    """Test compute_base_crop on ultra-tall vertical source frames (frame_ratio < target_ratio)."""
    # 360x1080 source (1:3 ratio): frame_ratio (0.333) < target_ratio (0.5625)
    # crop_w = 360, crop_h = round(360 / (9 / 16)) = round(640.0) = 640.
    assert compute_base_crop(360, 1080) == (360, 640)

    # 400x1600 source (1:4 ratio):
    # crop_w = 400, crop_h = round(400 * 16 / 9) = round(711.111...) = 711.
    assert compute_base_crop(400, 1600) == (400, 711)


def test_compute_base_crop_custom_aspect_ratios():
    """Test compute_base_crop with custom aspect ratio parameters."""
    # 1:1 square target on 1920x1080 widescreen frame
    assert compute_base_crop(1920, 1080, aspect_w=1, aspect_h=1) == (1080, 1080)

    # 1:1 square target on 1080x1920 vertical frame
    assert compute_base_crop(1080, 1920, aspect_w=1, aspect_h=1) == (1080, 1080)

    # 16:9 landscape target on 1080x1920 vertical frame
    # frame_ratio (1080/1920 = 0.5625) < target_ratio (16/9 = 1.7777...)
    # crop_w = 1080, crop_h = round(1080 / (16/9)) = round(607.5) = 608.
    assert compute_base_crop(1080, 1920, aspect_w=16, aspect_h=9) == (1080, 608)

    # 4:3 target ratio on 1920x1080 frame
    # frame_ratio (1.7777...) >= target_ratio (1.3333...)
    # crop_h = 1080, crop_w = round(1080 * 4 / 3) = 1440.
    assert compute_base_crop(1920, 1080, aspect_w=4, aspect_h=3) == (1440, 1080)


@pytest.mark.parametrize(
    "frame_w, frame_h, aspect_w, aspect_h",
    [
        (1920, 1080, 9, 16),
        (1080, 1920, 9, 16),
        (1080, 1080, 9, 16),
        (360, 1080, 9, 16),
        (3840, 2160, 9, 16),
        (1920, 1080, 1, 1),
        (1080, 1920, 16, 9),
        (1920, 1080, 4, 3),
        (800, 600, 16, 9),
        (100, 100, 9, 16),
    ],
)
def test_compute_base_crop_invariants(frame_w, frame_h, aspect_w, aspect_h):
    """Test mathematical and type invariants for compute_base_crop outputs."""
    crop_w, crop_h = compute_base_crop(frame_w, frame_h, aspect_w, aspect_h)

    # Types must be int
    assert isinstance(crop_w, int)
    assert isinstance(crop_h, int)

    # Crop size must be positive and fit within frame dimensions
    assert 0 < crop_w <= frame_w
    assert 0 < crop_h <= frame_h

    # Ratio of crop dimensions should match target ratio within discrete pixel rounding tolerance
    target_ratio = aspect_w / aspect_h
    crop_ratio = crop_w / crop_h
    tolerance = 1.0 / min(crop_w, crop_h)
    assert abs(crop_ratio - target_ratio) <= tolerance + 1e-6
