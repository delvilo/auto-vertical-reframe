import pytest
from auto_reframe import clamp


def test_clamp_within_range():
    """Test value within the low and high bounds."""
    assert clamp(5.0, 0.0, 10.0) == 5.0
    assert clamp(1.5, 1.0, 2.0) == 1.5


def test_clamp_below_low():
    """Test value below the low bound."""
    assert clamp(-5.0, 0.0, 10.0) == 0.0
    assert clamp(0.5, 1.0, 2.0) == 1.0


def test_clamp_above_high():
    """Test value above the high bound."""
    assert clamp(15.0, 0.0, 10.0) == 10.0
    assert clamp(2.5, 1.0, 2.0) == 2.0


def test_clamp_exact_low_bound():
    """Test value exactly equal to the low bound."""
    assert clamp(0.0, 0.0, 10.0) == 0.0
    assert clamp(1.0, 1.0, 2.0) == 1.0


def test_clamp_exact_high_bound():
    """Test value exactly equal to the high bound."""
    assert clamp(10.0, 0.0, 10.0) == 10.0
    assert clamp(2.0, 1.0, 2.0) == 2.0


def test_clamp_equal_bounds():
    """Test when low and high bounds are equal."""
    assert clamp(5.0, 3.0, 3.0) == 3.0
    assert clamp(1.0, 3.0, 3.0) == 3.0
    assert clamp(3.0, 3.0, 3.0) == 3.0


def test_clamp_negative_values():
    """Test clamping with negative numbers."""
    assert clamp(-5.0, -10.0, -1.0) == -5.0
    assert clamp(-15.0, -10.0, -1.0) == -10.0
    assert clamp(0.0, -10.0, -1.0) == -1.0


def test_clamp_floating_point_precision():
    """Test clamping with floating point numbers."""
    assert clamp(0.123456, 0.1, 0.2) == 0.123456
    assert clamp(0.099999, 0.1, 0.2) == 0.1
    assert clamp(0.200001, 0.1, 0.2) == 0.2
