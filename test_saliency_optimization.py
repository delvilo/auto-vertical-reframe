import unittest
from unittest.mock import MagicMock, patch
import sys

# Safely mock missing dependencies if cv2 is not present in environment
if 'cv2' not in sys.modules:
    sys.modules['cv2'] = MagicMock()
if 'scenedetect' not in sys.modules:
    sys.modules['scenedetect'] = MagicMock()
if 'ultralytics' not in sys.modules:
    sys.modules['ultralytics'] = MagicMock()
if 'numpy' not in sys.modules:
    sys.modules['numpy'] = MagicMock()

import auto_reframe


class TestHandcraftedSaliencyOptimization(unittest.TestCase):

    def test_handcrafted_saliency_compute_map(self):
        """Test HandcraftedSaliencyHelper compute_map execution and flow."""
        helper = auto_reframe.HandcraftedSaliencyHelper()

        mock_frame = MagicMock()
        mock_frame.shape = (720, 1280, 3)

        # Configure cv2 mock returns
        mock_dft_out = MagicMock()
        mock_dft_out.__getitem__.side_effect = lambda idx: MagicMock()
        auto_reframe.cv2.dft.return_value = mock_dft_out
        auto_reframe.cv2.resize.return_value = mock_frame
        auto_reframe.cv2.cvtColor.return_value = MagicMock()
        auto_reframe.cv2.magnitude.return_value = MagicMock()
        auto_reframe.cv2.blur.return_value = MagicMock()
        auto_reframe.cv2.idft.return_value = MagicMock()
        auto_reframe.cv2.GaussianBlur.return_value = MagicMock()
        auto_reframe.cv2.normalize.return_value = MagicMock()

        saliency_map = helper.compute_map(mock_frame)

        self.assertIsNotNone(saliency_map)
        self.assertEqual(helper.frames_total, 1)
        self.assertEqual(helper.frames_backend, 1)
        # Ensure cv2.phase (arctan2) is no longer called
        auto_reframe.cv2.phase.assert_not_called()


if __name__ == "__main__":
    unittest.main()
