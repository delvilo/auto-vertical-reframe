"""Artifact/strict-load checks plus optional cached upstream equivalence.

Set DEEPGAZE_REFERENCE_REPO to an existing verified upstream checkout to run
the real-checkpoint numerical comparison. No test downloads model weights.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from reframe.saliency.backends import deepgaze_loader as loader
from reframe.saliency.backends.deepgaze_model import DeepGazeMR


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.name = "deepgazemr-ledov.pt"
        self.data = b"verified test checkpoint bytes"
        self.digest = hashlib.sha256(self.data).hexdigest()
        self.torch = SimpleNamespace(hub=SimpleNamespace(download_url_to_file=Mock()))
        hashes = patch.dict(loader.ARTIFACT_SHA256, {self.name: self.digest})
        hashes.start()
        self.addCleanup(hashes.stop)

    def test_existing_hub_data_is_verified_and_reused_without_code_execution(self):
        artifact = self.root / "mtangemann_deepgazemr_master" / "data" / self.name
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(self.data)
        (artifact.parent.parent / "hubconf.py").write_text("raise RuntimeError('must not run')")
        self.assertEqual(loader._resolve_artifact(self.name, self.root, self.torch), artifact)
        self.torch.hub.download_url_to_file.assert_not_called()

    def test_bad_cache_downloads_only_pinned_artifact_atomically(self):
        bad = self.root / "mtangemann_deepgazemr_master" / "data" / self.name
        bad.parent.mkdir(parents=True)
        bad.write_bytes(b"incomplete download")
        def download(url, destination, **kwargs):
            self.assertIn(loader.UPSTREAM_COMMIT, url)
            self.assertTrue(url.endswith("/data/" + self.name))
            self.assertEqual(kwargs["hash_prefix"], self.digest)
            Path(destination).write_bytes(self.data)
        self.torch.hub.download_url_to_file.side_effect = download
        with self.assertLogs(level="WARNING"):
            artifact = loader._resolve_artifact(self.name, self.root, self.torch)
        self.assertEqual(artifact.read_bytes(), self.data)
        self.assertEqual(list(self.root.rglob("*.partial")), [])
        self.torch.hub.download_url_to_file.reset_mock()
        self.assertEqual(loader._resolve_artifact(self.name, self.root, self.torch), artifact)
        self.torch.hub.download_url_to_file.assert_not_called()

    def test_bad_download_never_becomes_visible_as_checkpoint(self):
        self.torch.hub.download_url_to_file.side_effect = (
            lambda url, destination, **kwargs: Path(destination).write_bytes(b"wrong"))
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            loader._resolve_artifact(self.name, self.root, self.torch)
        self.assertEqual(list(self.root.rglob("*.pt")), [])
        self.assertEqual(list(self.root.rglob("*.partial")), [])

    def test_native_download_error_propagates_and_tempfile_is_cleaned(self):
        self.torch.hub.download_url_to_file.side_effect = OSError("network unavailable")
        with self.assertRaisesRegex(OSError, "network unavailable"):
            loader._resolve_artifact(self.name, self.root, self.torch)
        self.assertEqual(list(self.root.rglob("*.partial")), [])


class ArchitectureTests(unittest.TestCase):
    def test_feature_skeleton_has_no_classifier_or_allocated_vgg_weights(self):
        model = DeepGazeMR()
        self.assertFalse(any(isinstance(module, nn.Linear) for module in model.modules()))
        self.assertEqual(len(model.features.vgg.features), 37)
        self.assertIsInstance(model.features.vgg.features[-1], nn.Identity)
        self.assertEqual(len(model.features.vgg.state_dict()), 32)
        self.assertTrue(all(value.is_meta for value in model.features.parameters()))
        self.assertTrue(all(not value.requires_grad for value in model.features.parameters()))
        self.assertEqual(model._window_length, 16)

    def test_invalid_center_bias_rejected_before_model_loading(self):
        for bias in (None, torch.ones(3), torch.zeros(2, 2), torch.full((2, 2), float("nan"))):
            with self.subTest(bias=bias), patch.object(
                torch, "load", side_effect=[{"model_state_dict": {}}, bias]
            ), self.assertRaisesRegex(ValueError, "center bias"):
                loader._load_checkpoint_model(Path("model"), Path("bias"))

    def test_missing_state_dict_or_feature_key_is_rejected(self):
        with patch.object(torch, "load", return_value={}), self.assertRaisesRegex(
            ValueError, "model_state_dict"
        ):
            loader._load_checkpoint_model(Path("model"), Path("bias"))
        state = DeepGazeMR().state_dict()
        del state["features.vgg.features.0.weight"]
        with patch.object(torch, "load", side_effect=[{"model_state_dict": state}, torch.ones(2, 2)]):
            with self.assertRaisesRegex(RuntimeError, "Missing key.*features.vgg.features.0.weight"):
                loader._load_checkpoint_model(Path("model"), Path("bias"))

    def test_unexpected_or_mismatched_state_dict_tensors_are_rejected(self):
        for case in ("unexpected", "shape"):
            with self.subTest(case=case):
                state = DeepGazeMR().state_dict()
                if case == "unexpected":
                    state["features.vgg.classifier.0.weight"] = torch.empty(1)
                    error = "Unexpected key"
                else:
                    state["features.vgg.features.0.weight"] = torch.empty(1)
                    error = "size mismatch"
                with patch.object(torch, "load", side_effect=[
                    {"model_state_dict": state}, torch.ones(2, 2)
                ]), self.assertRaisesRegex(RuntimeError, error):
                    loader._load_checkpoint_model(Path("model"), Path("bias"))


REFERENCE = os.environ.get("DEEPGAZE_REFERENCE_REPO")


@unittest.skipUnless(REFERENCE, "set DEEPGAZE_REFERENCE_REPO for real-checkpoint equivalence")
class UpstreamEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torchvision.models

        cls.repo = Path(REFERENCE)
        source = cls.repo / "deepgazemr/model.py"
        if loader._sha256(source) != "071a7a4897afb1f3f8b781f23526dd06e1e54796ee489f3488338e46b42789ca":
            raise ValueError("Reference architecture differs from pinned upstream")
        for name, expected in loader.ARTIFACT_SHA256.items():
            if loader._sha256(cls.repo / "data" / name) != expected:
                raise ValueError(f"Reference artifact hash mismatch: {name}")
        spec = importlib.util.spec_from_file_location("reference_deepgaze", source)
        reference_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference_module)
        factory = torchvision.models.vgg19
        def no_download_vgg(**kwargs):
            with torch.device("meta"):
                return factory(weights=None, init_weights=False)
        with patch.object(torchvision.models, "vgg19", side_effect=no_download_vgg):
            cls.reference = reference_module.DeepGazeMR()
        checkpoint = torch.load(cls.repo / "data/deepgazemr-ledov.pt",
                                weights_only=True, map_location="cpu")
        cls.reference.load_state_dict(checkpoint["model_state_dict"], strict=True, assign=True)
        cls.reference.center_bias = torch.load(cls.repo / "data/center-bias-ledov.pt",
                                              weights_only=True, map_location="cpu")
        cls.reference.eval()
        cls.lean = loader._load_checkpoint_model(
            cls.repo / "data/deepgazemr-ledov.pt", cls.repo / "data/center-bias-ledov.pt")

    def test_all_checkpoint_tensors_and_dtypes_match(self):
        expected, actual = self.reference.state_dict(), self.lean.state_dict()
        self.assertEqual(expected.keys(), actual.keys())
        for key in expected:
            self.assertTrue(torch.equal(expected[key], actual[key]), key)
            self.assertEqual(expected[key].dtype, actual[key].dtype)
        self.assertEqual(self.lean.center_bias.dtype, torch.float64)
        self.assertTrue(torch.equal(self.reference.center_bias, self.lean.center_bias))
        self.assertFalse(any(parameter.is_meta for parameter in self.lean.parameters()))

    def test_saliency_identical_for_two_temporal_windows_and_geometries(self):
        generator = torch.Generator().manual_seed(20261011)
        records = []
        for height, width in ((64, 96), (96, 64)):
            clip = torch.rand((16, 3, height, width), generator=generator)
            with torch.inference_mode():
                expected = self.reference(clip)
                actual = self.lean(clip)
            error = (expected - actual).abs().max().item()
            records.append({"clip_shape": list(clip.shape), "max_abs_error": error,
                            "equal": torch.equal(expected, actual), "dtype": str(actual.dtype)})
            self.assertEqual(error, 0.)
        print("DeepGaze upstream numerical equivalence:", json.dumps(records))


if __name__ == "__main__":
    unittest.main()
