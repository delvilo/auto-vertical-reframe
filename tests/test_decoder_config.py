import contextlib
import io
import unittest

from reframe.cli import parse_args
from reframe.config import AppConfig, validate_config


class DecoderConfigTests(unittest.TestCase):
    def test_tensor_processing_requires_torchcodec(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            validate_config(AppConfig(input="in", output="out", video_decoder="opencv",
                                      frame_processing="tensor"))
        args = parse_args(["in", "out", "--video-decoder", "torchcodec",
                           "--frame-processing", "tensor"])
        self.assertEqual(args.frame_processing, "tensor")

    def test_bounded_batch_and_explicit_device(self):
        for key, values in {"decode_batch_size": [0, 9, True, 1.5, None],
                            "decode_device": ["auto", "mps", "cuda:-1", "0", None],
                            "video_decoder": ["auto", None]}.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_config(AppConfig(input="in", output="out", **{key: value}))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["in", "out", "--decode-batch-size", "100"])
