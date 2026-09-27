import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock external dependencies before importing auto_reframe
for mod_name in ["cv2", "numpy", "scenedetect", "ultralytics"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

import auto_reframe  # noqa: E402


class TestDeepGazeMRSaliencyHelper(unittest.TestCase):
    @patch("auto_reframe.torch")
    def test_load_model_trust_repo_check(self, mock_torch):
        """Test that _load_model calls torch.hub.load with trust_repo='check'."""
        mock_torch.is_tensor = MagicMock(return_value=True)
        mock_torch.cuda.is_available.return_value = False

        helper = auto_reframe.DeepGazeMRSaliencyHelper()

        # Test remote load
        mock_torch.hub.load.return_value = MagicMock()

        success = helper._load_model()

        self.assertTrue(success)
        mock_torch.hub.load.assert_called_once_with(
            "mtangemann/deepgazemr",
            "DeepGazeMR",
            pretrained=True,
            trust_repo="check",
        )

        # Ensure trust_repo is not True in kwargs
        _, kwargs = mock_torch.hub.load.call_args
        self.assertEqual(kwargs.get("trust_repo"), "check")
        self.assertNotEqual(kwargs.get("trust_repo"), True)

    @patch("auto_reframe.Path")
    @patch("auto_reframe.torch")
    def test_local_fallback_trust_repo_check(self, mock_torch, mock_path):
        """Test local fallback hub load calls torch.hub.load with trust_repo='check'."""
        mock_torch.is_tensor = MagicMock(return_value=True)
        mock_torch.cuda.is_available.return_value = False

        helper = auto_reframe.DeepGazeMRSaliencyHelper()

        # Make remote load fail
        def hub_load_side_effect(*args, **kwargs):
            if kwargs.get("source") == "local" or (
                len(args) > 0 and str(args[0]).startswith("/")
            ):
                mock_m = MagicMock()
                return mock_m
            raise Exception("Remote load failed")

        mock_torch.hub.load.side_effect = hub_load_side_effect
        mock_tensor = MagicMock()
        mock_torch.load.side_effect = [
            {"model_state_dict": {}},
            mock_tensor,
        ]

        # Mock candidate directory
        mock_candidate = MagicMock()
        mock_candidate.is_dir.return_value = True
        mock_candidate.__truediv__ = lambda self, other: MagicMock(
            is_file=lambda: True
        )

        mock_hub_dir = MagicMock()
        mock_hub_dir.glob.return_value = [mock_candidate]
        mock_torch.hub.get_dir.return_value = "/tmp/hub"
        mock_path.return_value = mock_hub_dir

        success = helper._load_model()
        self.assertTrue(success)

        # Check all torch.hub.load calls
        for call in mock_torch.hub.load.call_args_list:
            _, kwargs = call
            self.assertEqual(kwargs.get("trust_repo"), "check")
            self.assertNotEqual(kwargs.get("trust_repo"), True)


if __name__ == "__main__":
    unittest.main()
