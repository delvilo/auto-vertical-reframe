import unittest
from unittest.mock import MagicMock, patch
import sys
import os

class TestHandcraftedSaliency(unittest.TestCase):
    def test_compute_map_polar_to_cart(self):
        # Safely mock dependencies only for auto_reframe import without polluting global sys.modules permanently
        mock_modules = {
            'cv2': MagicMock(),
            'scenedetect': MagicMock(),
            'ultralytics': MagicMock(),
            'numpy': MagicMock(),
        }
        with patch.dict(sys.modules, mock_modules):
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
            import auto_reframe
            from auto_reframe import HandcraftedSaliencyHelper

            helper = HandcraftedSaliencyHelper()
            mock_cv2 = sys.modules['cv2']

            # Setup mock frame
            mock_frame = MagicMock()
            mock_frame.shape = (720, 1280, 3)

            # Configure mock return values for cv2 functions in compute_map
            mock_cv2.resize.return_value = mock_frame
            mock_cv2.cvtColor.return_value = MagicMock()
            mock_dft_out = MagicMock()
            mock_dft_out.__getitem__.return_value = MagicMock()
            mock_cv2.dft.return_value = mock_dft_out
            mock_cv2.magnitude.return_value = MagicMock()
            mock_cv2.blur.return_value = MagicMock()
            mock_cv2.phase.return_value = MagicMock()
            mock_cv2.polarToCart.return_value = (MagicMock(), MagicMock())
            mock_cv2.idft.return_value = MagicMock()
            mock_cv2.GaussianBlur.return_value = MagicMock()
            mock_cv2.normalize.return_value = MagicMock()

            _ = helper.compute_map(mock_frame)

            # Assert cv2.polarToCart was called
            self.assertTrue(mock_cv2.polarToCart.called)

if __name__ == "__main__":
    unittest.main()
