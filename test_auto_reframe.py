import pytest
from auto_reframe import lerp


def test_lerp_endpoints():
    assert lerp(0.0, 10.0, 0.0) == 0.0
    assert lerp(0.0, 10.0, 1.0) == 10.0
    assert lerp(5.0, 25.0, 0.0) == 5.0
    assert lerp(5.0, 25.0, 1.0) == 25.0


def test_lerp_intermediate_values():
    assert pytest.approx(lerp(0.0, 10.0, 0.5)) == 5.0
    assert pytest.approx(lerp(0.0, 10.0, 0.25)) == 2.5
    assert pytest.approx(lerp(0.0, 10.0, 0.75)) == 7.5
    assert pytest.approx(lerp(10.0, 20.0, 0.3)) == 13.0


def test_lerp_extrapolation():
    assert pytest.approx(lerp(0.0, 10.0, -0.5)) == -5.0
    assert pytest.approx(lerp(0.0, 10.0, 1.5)) == 15.0
    assert pytest.approx(lerp(10.0, 20.0, -1.0)) == 0.0
    assert pytest.approx(lerp(10.0, 20.0, 2.0)) == 30.0


def test_lerp_equal_inputs():
    assert lerp(5.0, 5.0, 0.0) == 5.0
    assert lerp(5.0, 5.0, 0.5) == 5.0
    assert lerp(5.0, 5.0, 1.0) == 5.0
    assert lerp(5.0, 5.0, 2.0) == 5.0


def test_lerp_negative_numbers_and_opposite_signs():
    assert pytest.approx(lerp(-10.0, 10.0, 0.5)) == 0.0
    assert pytest.approx(lerp(-10.0, 10.0, 0.0)) == -10.0
    assert pytest.approx(lerp(-10.0, 10.0, 1.0)) == 10.0
    assert pytest.approx(lerp(-20.0, -10.0, 0.5)) == -15.0


def test_lerp_zeros():
    assert lerp(0.0, 0.0, 0.0) == 0.0
    assert lerp(0.0, 0.0, 0.5) == 0.0
    assert lerp(0.0, 0.0, 1.0) == 0.0
