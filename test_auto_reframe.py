import logging
import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock third-party dependencies if not installed
for module_name in ["cv2", "numpy", "scenedetect", "ultralytics"]:
    if module_name not in sys.modules:
        try:
            __import__(module_name)
        except ImportError:
            sys.modules[module_name] = MagicMock()

from auto_reframe import setup_logging


class TestSetupLogging(unittest.TestCase):
    @patch("logging.config.dictConfig")
    @patch("logging.basicConfig")
    def test_setup_logging_success(self, mock_basicConfig, mock_dictConfig):
        """Test setup_logging when dictConfig succeeds."""
        setup_logging("INFO")

        mock_dictConfig.assert_called_once()
        config = mock_dictConfig.call_args[0][0]
        self.assertEqual(config["version"], 1)
        self.assertFalse(config["disable_existing_loggers"])
        self.assertEqual(config["handlers"]["default"]["level"], "INFO")
        self.assertEqual(config["loggers"][""]["level"], "INFO")
        mock_basicConfig.assert_not_called()

    @patch("logging.config.dictConfig", side_effect=Exception("dictConfig failed"))
    @patch("logging.basicConfig")
    def test_setup_logging_exception_fallback(self, mock_basicConfig, mock_dictConfig):
        """Test setup_logging fallback to basicConfig when dictConfig raises an Exception."""
        setup_logging("INFO")

        mock_dictConfig.assert_called_once()
        mock_basicConfig.assert_called_once_with(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s"
        )

    @patch("logging.config.dictConfig", side_effect=ValueError("Invalid config"))
    @patch("logging.basicConfig")
    def test_setup_logging_exception_fallback_invalid_level_string(self, mock_basicConfig, mock_dictConfig):
        """Test setup_logging fallback when given lower-case level string."""
        setup_logging("debug")

        mock_dictConfig.assert_called_once()
        mock_basicConfig.assert_called_once_with(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(message)s"
        )


if __name__ == "__main__":
    unittest.main()
