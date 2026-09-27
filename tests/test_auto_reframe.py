import pytest
from auto_reframe import current_crop_size


class TestCurrentCropSize:
    def test_standard_zoom_unconstrained(self):
        """Test standard zoom factors where results stay well within bounds."""
        # base crop 1080x1920, frame 3840x2160, zoom 1.0
        assert current_crop_size(1080, 1920, 1.0, 3840, 2160) == (1080, 1920)

        # zoom = 2.0 (zoomed in -> crop box shrinks to half)
        assert current_crop_size(1080, 1920, 2.0, 3840, 2160) == (540, 960)

        # zoom = 1.5 (1080 / 1.5 = 720, 1920 / 1.5 = 1280)
        assert current_crop_size(1080, 1920, 1.5, 3840, 2160) == (720, 1280)

    def test_rounding_behavior(self):
        """Test rounding logic for non-integer division results."""
        # 1000 / 1.5 = 666.666... -> round to 667
        # 500 / 1.5 = 333.333... -> round to 333
        assert current_crop_size(1000, 500, 1.5, 3840, 2160) == (667, 333)

    def test_safe_zoom_lower_bound(self):
        """Test zoom values <= 0.1 are clamped to safe_zoom = 0.1."""
        # base 100x200, frame 2000x2000
        # at zoom = 0.1 -> 100 / 0.1 = 1000, 200 / 0.1 = 2000
        expected = current_crop_size(100, 200, 0.1, 2000, 2000)

        # Any zoom below 0.1 (0.05, 0.0, -1.0) should produce identical result as zoom 0.1
        assert current_crop_size(100, 200, 0.05, 2000, 2000) == expected
        assert current_crop_size(100, 200, 0.0, 2000, 2000) == expected
        assert current_crop_size(100, 200, -2.5, 2000, 2000) == expected

    def test_min_dimension_clamping(self):
        """Test crop dimensions are clamped to minimum 64 pixels when heavily zoomed in."""
        # base 100x100, zoom = 10.0 -> 100 / 10 = 10 -> clamped to 64
        assert current_crop_size(100, 100, 10.0, 1920, 1080) == (64, 64)

        # base 500x50, zoom = 2.0 -> 500/2=250, 50/2=25 -> crop_h clamped to 64
        assert current_crop_size(500, 50, 2.0, 1920, 1080) == (250, 64)

    def test_max_dimension_clamping(self):
        """Test crop dimensions are clamped to frame_w and frame_h when zoomed out."""
        # base 1080x1920, zoom = 0.5 -> 2160 x 3840, but frame is 1920x1080
        # crop_w max clamped to frame_w (1920), crop_h max clamped to frame_h (1080)
        assert current_crop_size(1080, 1920, 0.5, 1920, 1080) == (1920, 1080)

    def test_return_types(self):
        """Test return type guarantees (tuple of int, int)."""
        res = current_crop_size(1080, 1920, 1.25, 1920, 1080)
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert isinstance(res[0], int)
        assert isinstance(res[1], int)
