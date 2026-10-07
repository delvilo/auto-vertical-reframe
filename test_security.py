import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Mock dependencies that might not be installed in the environment before importing auto_reframe
for mod in ['cv2', 'scenedetect', 'ultralytics']:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

from reframe.saliency.backends import deepgazemr as backend

class TestSecurityFix(unittest.TestCase):
    @patch("reframe.saliency.backends.deepgazemr.torch")
    def test_torch_load_weights_only(self, mock_torch):
        helper = backend.DeepGazeMRSaliencyHelper()

        # Mock torch.hub.load to fail on first attempt to trigger cached repo fallback
        mock_torch.hub.load.side_effect = [Exception("Hub load failed"), MagicMock()]
        mock_torch.hub.get_dir.return_value = "/fake/hub/dir"

        mock_dir = MagicMock(spec=Path)
        mock_dir.is_dir.return_value = True
        mock_ckpt = MagicMock(spec=Path)
        mock_ckpt.is_file.return_value = True
        mock_bias = MagicMock(spec=Path)
        mock_bias.is_file.return_value = True

        def truediv_side_effect(arg):
            if arg == "data/deepgazemr-ledov.pt":
                return mock_ckpt
            if arg == "data/center-bias-ledov.pt":
                return mock_bias
            return MagicMock()

        mock_dir.__truediv__.side_effect = truediv_side_effect

        mock_hub_dir = MagicMock()
        mock_hub_dir.glob.return_value = [mock_dir]

        mock_checkpoint_dict = {"model_state_dict": {}}
        mock_center_bias = MagicMock()
        mock_torch.is_tensor.return_value = True

        mock_torch.load.side_effect = [mock_checkpoint_dict, mock_center_bias]

        with patch("reframe.saliency.backends.deepgazemr.Path") as mock_path_cls:
            mock_path_cls.return_value = mock_hub_dir
            res = helper._load_model()

        self.assertTrue(res)
        self.assertEqual(mock_torch.load.call_count, 2)
        for call_args in mock_torch.load.call_args_list:
            args, kwargs = call_args
            self.assertIn("weights_only", kwargs)
            self.assertTrue(kwargs["weights_only"])

if __name__ == "__main__":
    unittest.main()

