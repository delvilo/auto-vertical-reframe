"""Requested MSDB renders must not silently turn into handcrafted renders."""
import unittest
from unittest.mock import patch
from reframe.config import AppConfig
from reframe.saliency.factory import build_saliency_helper
from reframe.saliency.backends import deepgazemsdb as backend


class TestSecurityReframe(unittest.TestCase):
    def test_default_is_strict_msdb_and_full_resolution(self):
        service = build_saliency_helper(AppConfig())
        self.assertFalse(service.backend.allow_fallback)
        self.assertIsNone(service.max_side)
        self.assertEqual(service.backend.center_bias_mode, "mit1003")

    def test_uniform_is_forwarded_and_only_auto_permits_fallback(self):
        service = build_saliency_helper(AppConfig(saliency_model="auto", saliency_center_bias="uniform"))
        self.assertTrue(service.backend.allow_fallback)
        self.assertEqual(service.backend.center_bias_mode, "uniform")

    def test_requested_cuda_must_not_become_cpu(self):
        with patch.object(backend, "torch") as torch, patch.object(backend, "load_msdb_model") as load:
            torch.cuda.is_available.return_value = False
            helper = backend.DeepGazeMSDBSaliencyHelper(device="cuda")
            with self.assertRaisesRegex(RuntimeError, "CUDA device is unavailable"):
                helper.load()
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
