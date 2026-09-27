import sys
from unittest.mock import MagicMock

# Mock video libraries before importing auto_reframe
for mod in ["cv2", "scenedetect", "ultralytics", "mediapipe"]:
    sys.modules[mod] = MagicMock()

import math
import pytest
from auto_reframe import critically_damped_step


class TestCriticallyDampedStep:
    def test_convergence_over_time(self):
        """Test that the position smooths towards the target over multiple steps."""
        current = 0.0
        target = 100.0
        velocity = 0.0
        dt = 0.1
        tau = 0.5
        max_speed = 1000.0

        positions = [current]
        for _ in range(50):
            current, velocity = critically_damped_step(
                current, target, velocity, dt, tau, max_speed
            )
            positions.append(current)

        # Should move closer to target with each step initially
        assert positions[1] > positions[0]
        # Should eventually converge close to target
        assert abs(positions[-1] - target) < 1e-3
        assert velocity == pytest.approx(0.0, abs=1e-3)

    def test_already_at_target_zero_velocity(self):
        """Test that being at target with zero velocity remains stationary."""
        current = 50.0
        target = 50.0
        velocity = 0.0
        dt = 0.1
        tau = 0.5
        max_speed = 100.0

        new_pos, new_vel = critically_damped_step(
            current, target, velocity, dt, tau, max_speed
        )
        assert new_pos == 50.0
        assert new_vel == 0.0

    def test_max_speed_delta_clamping(self):
        """Test that position delta per step is clamped by max_speed * dt."""
        current = 0.0
        target = 1000.0
        velocity = 0.0
        dt = 0.1
        tau = 0.1  # Very fast convergence time constant
        max_speed = 50.0  # Max delta allowed = 50.0 * 0.1 = 5.0

        new_pos, new_vel = critically_damped_step(
            current, target, velocity, dt, tau, max_speed
        )
        delta = new_pos - current
        assert delta == pytest.approx(5.0)

    def test_velocity_clamping(self):
        """Test that velocity is clamped to max_speed."""
        current = 0.0
        target = 1000.0
        velocity = 500.0  # Large initial velocity
        dt = 0.1
        tau = 0.5
        max_speed = 20.0

        new_pos, new_vel = critically_damped_step(
            current, target, velocity, dt, tau, max_speed
        )
        assert abs(new_vel) <= max_speed

    def test_target_overshoot_crossing(self):
        """Test that if step crosses/overshoots the target, position snaps to target and velocity resets to 0."""
        current = 9.9
        target = 10.0
        velocity = 100.0  # High velocity towards target causing overshoot
        dt = 0.1
        tau = 0.1
        max_speed = 1000.0

        new_pos, new_vel = critically_damped_step(
            current, target, velocity, dt, tau, max_speed
        )
        assert new_pos == target
        assert new_vel == 0.0

    def test_zero_dt(self):
        """Test that dt = 0 results in no position change."""
        current = 10.0
        target = 20.0
        velocity = 5.0
        dt = 0.0
        tau = 0.5
        max_speed = 100.0

        new_pos, new_vel = critically_damped_step(
            current, target, velocity, dt, tau, max_speed
        )
        assert new_pos == current

    def test_small_tau(self):
        """Test that tau close to or <= 0 is handled safely without division by zero."""
        current = 0.0
        target = 10.0
        velocity = 0.0
        dt = 0.1
        tau = 0.0
        max_speed = 100.0

        # Should not raise ZeroDivisionError
        new_pos, new_vel = critically_damped_step(
            current, target, velocity, dt, tau, max_speed
        )
        assert isinstance(new_pos, float)
        assert isinstance(new_vel, float)
