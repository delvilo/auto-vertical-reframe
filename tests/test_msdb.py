"""Real tensor/image contracts; neural weights are replaced with controlled outputs."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from reframe import cli
from reframe.contracts import FrameContext, FrameObservations
from reframe.saliency import msdb_assets as assets
from reframe.saliency.backends import deepgazemsdb as backend
from reframe.saliency.factory import build_saliency_helper
from reframe.saliency.viewing import ViewingGeometry

torch = backend.torch


class ViewingTests(unittest.TestCase):
    def test_display_and_crop_scaling(self):
        view = ViewingGeometry()
        frame = FrameContext(1, 0, 1920, 1080, 1, 1080)
        self.assertAlmostEqual(view.output_ppd, 37.84346, places=4)
        self.assertAlmostEqual(view.input_ppd(frame), 21.28695, places=4)
        self.assertAlmostEqual(view.input_ppd(replace(frame, view_crop_height=540)),
                               view.input_ppd(frame) / 2)
        self.assertAlmostEqual(replace(view, distance_cm=120).input_ppd(frame),
                               view.input_ppd(frame) * 2)
        # Reducing the output file resolution must not change assumed viewing size.
        self.assertAlmostEqual(replace(view, output_width=540, output_height=960).input_ppd(frame),
                               view.input_ppd(frame))
        self.assertEqual(replace(view, override_ppd=35).input_ppd(frame), 35)
        frame_4k = replace(frame, width=3840, height=2160, view_crop_height=2160)
        self.assertAlmostEqual(view.input_ppd(frame_4k), 2 * view.input_ppd(frame))

    def test_invalid_cli_geometry_is_rejected(self):
        for flag, value in (("--saliency-screen-inches", "0"),
                            ("--saliency-viewing-distance-cm", "nan"),
                            ("--saliency-pixel-per-dva", "-1")):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                cli.parse_args(["in.mp4", "out.mp4", flag, value])

    def test_log_priors_normalize_on_nonsquare_frames(self):
        template = np.array([[-7., -5., -7.], [-5., 0., -5.], [-7., -5., -7.]])
        biased = assets.center_bias_log_density((80, 160), template)
        uniform = assets.center_bias_log_density((80, 160), None)
        for density in (biased, uniform):
            self.assertEqual(density.shape, (80, 160))
            self.assertAlmostEqual(float(np.exp(density).sum()), 1., places=5)
        self.assertGreater(biased[40, 80], biased[0, 0])
        self.assertEqual(float(uniform.max()), float(uniform.min()))


@unittest.skipIf(torch is None, "requires PyTorch tensors, no model weights")
class MSDBTests(unittest.TestCase):
    def test_upstream_factories_use_cpu_and_restore_after_failure(self):
        import clip
        from deepgaze_pytorch import deepgazemsdb as upstream
        old_factories = upstream.CLIPResNet50x64, upstream.DINOv2_ViTB14
        visual = torch.nn.Sequential()
        dino = torch.nn.Linear(2, 2)
        with patch.object(clip, "load", return_value=(SimpleNamespace(visual=visual), None)) as load_clip, \
                patch.object(torch.hub, "load", return_value=dino) as load_dino, \
                patch.object(torch.hub, "load_state_dict_from_url", return_value=dino.state_dict()) as weights:
            with self.assertRaisesRegex(RuntimeError, "constructor interrupted"):
                with assets._cpu_backbone_factories(upstream):
                    upstream.CLIPResNet50x64()
                    upstream.DINOv2_ViTB14()
                    raise RuntimeError("constructor interrupted")
            self.assertEqual((upstream.CLIPResNet50x64, upstream.DINOv2_ViTB14), old_factories)
        self.assertEqual(load_clip.call_args.kwargs["device"], "cpu")
        self.assertEqual(load_dino.call_args.args[0], assets.DINO_REPOSITORY)
        self.assertFalse(load_dino.call_args.kwargs["pretrained"])
        self.assertTrue(weights.call_args.kwargs["weights_only"])

    def make_model(self):
        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, image, bias, pixel_per_dva, dataset):
                self.calls.append((image.clone(), bias.clone(), pixel_per_dva, dataset))
                return bias  # known log-density makes output conversion verifiable
        return RecordingModel()

    def test_factory_preserves_4k_rgb_range_and_predicts_first_frame(self):
        args = cli.parse_args(["in.mp4", "out.mp4", "--saliency-device", "cpu",
                               "--saliency-max-side", "64"])
        service = build_saliency_helper(args)
        self.addCleanup(service.close)
        model = self.make_model()
        template = np.array([[-6., -4., -6.], [-4., 0., -4.], [-6., -4., -6.]])
        frame = np.empty((2160, 3840, 3), np.uint8)
        frame[:] = [10, 100, 255]
        context = FrameContext(1, 0, 3840, 2160, 1, 2160)
        with patch.object(backend, "load_msdb_model", return_value=model), \
                patch.object(backend, "load_mit1003_template", return_value=template):
            result = service.process(frame, FrameObservations(context))
        image, prior, ppd, dataset = model.calls[0]
        self.assertEqual(tuple(image.shape), (1, 3, 2160, 3840))
        np.testing.assert_array_equal(image[0, :, 0, 0].numpy(), [255, 100, 10])
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(tuple(prior.shape), (1, 2160, 3840))
        self.assertIsNone(dataset)
        self.assertAlmostEqual(ppd, 42.57390, places=4)
        self.assertEqual((result.backend, result.status), ("deepgazemsdb", "predicted"))
        self.assertEqual(result.map.shape, (2160, 3840))
        np.testing.assert_allclose(result.map, np.exp(prior[0].numpy() - prior.max().item()), rtol=1e-5)

    def test_uniform_never_downloads_bias_and_geometry_tracks_crop(self):
        helper = backend.DeepGazeMSDBSaliencyHelper(device="cpu", center_bias="uniform")
        self.addCleanup(helper.close)
        model = self.make_model()
        frame = np.zeros((80, 160, 3), np.uint8)
        context = FrameContext(1, 0, 160, 80, 1, 80)
        with patch.object(backend, "load_msdb_model", return_value=model), \
                patch.object(backend, "load_mit1003_template") as download:
            helper.predict(frame, context)
            helper.predict(frame, replace(context, frame_index=2, view_crop_height=40))
        download.assert_not_called()
        self.assertAlmostEqual(model.calls[0][2], 2 * model.calls[1][2])
        self.assertTrue(torch.all(model.calls[0][1] == model.calls[0][1][0, 0, 0]))

    def test_invalid_model_output_and_oom_do_not_resize_or_fallback(self):
        for failure in (RuntimeError("CUDA out of memory"), None):
            helper = backend.DeepGazeMSDBSaliencyHelper(device="cpu", center_bias="uniform")
            self.addCleanup(helper.close)
            model = self.make_model()
            with patch.object(backend, "load_msdb_model", return_value=model), \
                    patch.object(model, "forward", side_effect=failure,
                                 return_value=torch.full((1, 64, 128), float("nan"))) as forward, \
                    patch.object(helper.fallback, "compute_map") as fallback:
                with self.assertRaises((RuntimeError, ValueError)):
                    helper.predict(np.zeros((64, 128, 3), np.uint8), FrameContext(1, 0, 128, 64, 1))
                self.assertEqual(forward.call_count, 1)
                fallback.assert_not_called()

    def test_auto_fallback_is_visible_and_close_releases_assets(self):
        helper = backend.DeepGazeMSDBSaliencyHelper(device="cpu", center_bias="uniform", allow_fallback=True)
        with patch.object(backend, "load_msdb_model", side_effect=RuntimeError("missing weights")), \
                self.assertLogs(level="WARNING") as logs:
            result = helper.predict(np.zeros((64, 128, 3), np.uint8), FrameContext(1, 0, 128, 64, 1))
        self.assertEqual((result.backend, result.status), ("handcrafted", "fallback"))
        self.assertIn("missing weights", result.reason)
        self.assertIn("Traceback", "\n".join(logs.output))
        self.assertEqual(helper.telemetry()["frames_fallback"], 1)
        helper.close()
        self.assertIsNone(helper.model)
        self.assertIsNone(helper.bias_tensor)


if __name__ == "__main__":
    unittest.main()
