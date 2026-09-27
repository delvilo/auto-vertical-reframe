import pytest
import numpy as np

# Import regression_velocity directly without triggering full auto_reframe dependencies if possible,
# or import from auto_reframe.
# Since auto_reframe imports cv2, mediapipe, ultralytics, etc., at module level,
# we can mock cv2 or other missing heavy modules if necessary, or import regression_velocity directly.
try:
    from auto_reframe import regression_velocity
except ModuleNotFoundError:
    import sys
    from unittest.mock import MagicMock
    sys.modules['cv2'] = MagicMock()
    sys.modules['scenedetect'] = MagicMock()
    sys.modules['ultralytics'] = MagicMock()
    sys.modules['mediapipe'] = MagicMock()
    from auto_reframe import regression_velocity


def test_regression_velocity_insufficient_points():
    """Test history with fewer than 3 points returns (0.0, 0.0)."""
    assert regression_velocity([]) == (0.0, 0.0)
    assert regression_velocity([(0.0, 10.0, 20.0)]) == (0.0, 0.0)
    assert regression_velocity([(0.0, 10.0, 20.0), (1.0, 12.0, 25.0)]) == (0.0, 0.0)


def test_regression_velocity_invalid_time_deltas():
    """Test history with identical or near-zero time steps returns (0.0, 0.0)."""
    history = [
        (1.0, 10.0, 20.0),
        (1.0, 15.0, 25.0),
        (1.0, 20.0, 30.0),
    ]
    assert regression_velocity(history) == (0.0, 0.0)


def test_regression_velocity_constant_motion():
    """Test history with constant velocity returns expected vx and vy slopes."""
    # x(t) = 10 + 5*t -> vx = 5.0
    # y(t) = 20 - 3*t -> vy = -3.0
    history = [
        (0.0, 10.0, 20.0),
        (1.0, 15.0, 17.0),
        (2.0, 20.0, 14.0),
        (3.0, 25.0, 11.0),
    ]
    vx, vy = regression_velocity(history)
    assert pytest.approx(vx) == 5.0
    assert pytest.approx(vy) == -3.0


def test_regression_velocity_robustness_to_outliers():
    """Test that median pairwise slope estimation is robust to single-frame outliers (Theil-Sen)."""
    # Normal motion: vx = 10.0, vy = 20.0
    # Add an outlier at t = 2.0
    history = [
        (0.0, 0.0, 0.0),
        (1.0, 10.0, 20.0),
        (2.0, 100.0, -500.0),  # Outlier frame (e.g. tracking glitch)
        (3.0, 30.0, 60.0),
        (4.0, 40.0, 80.0),
    ]
    vx, vy = regression_velocity(history)
    assert pytest.approx(vx) == 10.0
    assert pytest.approx(vy) == 20.0


def test_regression_velocity_stationary():
    """Test history with zero motion returns (0.0, 0.0)."""
    history = [
        (0.0, 50.0, 50.0),
        (1.0, 50.0, 50.0),
        (2.0, 50.0, 50.0),
    ]
    vx, vy = regression_velocity(history)
    assert pytest.approx(vx) == 0.0
    assert pytest.approx(vy) == 0.0
