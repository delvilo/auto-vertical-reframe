import argparse
import unittest
from unittest.mock import MagicMock, patch
import auto_reframe
from auto_reframe import DeepGazeMRSaliencyHelper, build_saliency_helper


class TestSecurityReframe(unittest.TestCase):

    def test_deepgazemr_trust_repo_default_false(self):
        """Verify that DeepGazeMRSaliencyHelper defaults trust_repo to False."""
        helper = DeepGazeMRSaliencyHelper()
        self.assertFalse(helper.trust_repo)

    @patch("auto_reframe.torch")
    def test_load_model_passes_trust_repo_false_by_default(self, mock_torch):
        """Verify torch.hub.load is called with trust_repo=False by default."""
        mock_torch.hub.load.side_effect = Exception("Trust refused")
        helper = DeepGazeMRSaliencyHelper(trust_repo=False)
        result = helper._load_model()

        self.assertFalse(result)
        self.assertTrue(helper._disabled)
        self.assertEqual(helper.active_backend, "handcrafted")

        # Check torch.hub.load was called with trust_repo=False
        mock_torch.hub.load.assert_called_with(
            "mtangemann/deepgazemr",
            "DeepGazeMR",
            pretrained=True,
            trust_repo=False,
        )

    @patch("auto_reframe.torch")
    def test_load_model_passes_trust_repo_true_when_configured(self, mock_torch):
        """Verify torch.hub.load is called with trust_repo=True when trust_repo=True."""
        mock_model = MagicMock()
        mock_torch.hub.load.return_value = mock_model
        mock_torch.is_tensor.return_value = False

        helper = DeepGazeMRSaliencyHelper(trust_repo=True)
        result = helper._load_model()

        self.assertTrue(result)
        self.assertFalse(helper._disabled)
        self.assertEqual(helper.active_backend, "deepgazemr")

        mock_torch.hub.load.assert_called_once_with(
            "mtangemann/deepgazemr",
            "DeepGazeMR",
            pretrained=True,
            trust_repo=True,
        )

    def test_build_saliency_helper_passes_trust_repo_arg(self):
        """Verify build_saliency_helper forwards saliency_trust_repo from CLI args."""
        args_false = argparse.Namespace(
            saliency_model="deepgazemr",
            saliency_device="cpu",
            saliency_max_side=384,
            saliency_trust_repo=False,
            saliency_max_failures=3,
            saliency_amp=False,
            saliency_interval=3,
            saliency_ema=0.65,
        )
        sampled = build_saliency_helper(args_false)
        self.assertFalse(sampled.backend.trust_repo)

        args_true = argparse.Namespace(
            saliency_model="deepgazemr",
            saliency_device="cpu",
            saliency_max_side=384,
            saliency_trust_repo=True,
            saliency_max_failures=3,
            saliency_amp=False,
            saliency_interval=3,
            saliency_ema=0.65,
        )
        sampled_true = build_saliency_helper(args_true)
        self.assertTrue(sampled_true.backend.trust_repo)


if __name__ == "__main__":
    unittest.main()
