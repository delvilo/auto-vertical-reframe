from unittest.mock import MagicMock, PropertyMock
import pytest
from auto_reframe import get_track_id


def test_get_track_id_valid():
    """Test get_track_id with a valid box containing track ID."""
    mock_tensor = MagicMock()
    mock_tensor.item.return_value = 42
    box = MagicMock()
    box.id = [mock_tensor]

    assert get_track_id(box) == 42


def test_get_track_id_none_id():
    """Test get_track_id when box.id is None."""
    box = MagicMock()
    box.id = None

    assert get_track_id(box) is None


def test_get_track_id_exception_on_attribute_access():
    """Test get_track_id when accessing box.id raises an exception."""
    box = MagicMock()
    type(box).id = PropertyMock(side_effect=RuntimeError("Property access error"))

    assert get_track_id(box) is None


def test_get_track_id_empty_id_list():
    """Test get_track_id when box.id is empty list (IndexError)."""
    box = MagicMock()
    box.id = []

    assert get_track_id(box) is None


def test_get_track_id_item_raises_exception():
    """Test get_track_id when item() raises an exception."""
    mock_tensor = MagicMock()
    mock_tensor.item.side_effect = RuntimeError("Item error")
    box = MagicMock()
    box.id = [mock_tensor]

    assert get_track_id(box) is None


def test_get_track_id_int_conversion_fails():
    """Test get_track_id when int conversion raises ValueError or TypeError."""
    mock_tensor = MagicMock()
    mock_tensor.item.return_value = "invalid_int"
    box = MagicMock()
    box.id = [mock_tensor]

    assert get_track_id(box) is None


def test_get_track_id_non_box_object():
    """Test get_track_id when box object is None or primitive value."""
    assert get_track_id(None) is None
    assert get_track_id(123) is None
    assert get_track_id("invalid") is None
