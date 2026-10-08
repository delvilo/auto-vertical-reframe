"""Reject malformed/non-tensor assets instead of silently running random heads."""
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from reframe.saliency import msdb_assets as assets


class TestMSDBAssets(unittest.TestCase):
    def test_incomplete_head_is_rejected_before_loading(self):
        upstream = MagicMock()
        model = upstream.DeepGazeMSDB.return_value
        model.head_state_dict.return_value = {"saliency_network.weight": object(),
                                               "features.size_weights": object()}
        with patch.dict("sys.modules", {"deepgaze_pytorch": MagicMock(deepgazemsdb=upstream)}), \
                patch.object(assets, "_cpu_backbone_factories", return_value=nullcontext()), \
                patch.object(assets, "torch") as torch:
            torch.hub.load_state_dict_from_url.return_value = {"saliency_network.weight": object()}
            with self.assertRaisesRegex(ValueError, "features.size_weights"):
                assets.load_msdb_model()
            model.load_state_dict.assert_not_called()
            self.assertTrue(torch.hub.load_state_dict_from_url.call_args.kwargs["weights_only"])

    def test_center_bias_rejects_pickle_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bias.npy"
            for values in (np.array([[object()]], dtype=object), np.full((4, 4), np.nan)):
                np.save(path, values)
                with self.assertRaises(ValueError):
                    assets._read_template(path)


if __name__ == "__main__":
    unittest.main()
