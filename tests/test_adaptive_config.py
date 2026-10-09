"""Public configuration for adaptive segmentation scheduling."""
import contextlib
import io
import unittest

from reframe.cli import parse_args
from reframe.config import AppConfig, validate_config


class AdaptiveConfigTests(unittest.TestCase):
    def test_default_schedule(self):
        args = parse_args(["input.mp4", "output.mp4"])
        self.assertEqual(args.seg_max_gap, 3)
        self.assertEqual(args.seg_max_age, 0.1)
        self.assertEqual(args.seg_thumbnail_mode, "lazy")
        self.assertFalse(args.post_restore)

    def test_baseline_and_precrop_are_independent(self):
        args = parse_args([
            "input.mp4", "output.mp4", "--seg-max-gap", "1",
            "--seg-max-age", "0.05", "--precrop", "middle",
        ])
        self.assertEqual(args.seg_max_gap, 1)
        self.assertEqual(args.seg_max_age, 0.05)
        self.assertEqual(args.precrop, "middle")

    def test_invalid_cli_gap(self):
        for value in ("0", "-1", "1.5", "nan", "inf"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    parse_args(["input.mp4", "output.mp4", "--seg-max-gap", value])
                self.assertEqual(caught.exception.code, 2)

    def test_invalid_cli_age(self):
        for value in ("0", "-0.1", "nan", "inf", "-inf"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    parse_args(["input.mp4", "output.mp4", f"--seg-max-age={value}"])
                self.assertEqual(caught.exception.code, 2)

    def test_direct_config_rejects_invalid_gap(self):
        for value in (0, -1, True, False, 1.5, 3.0, "3", None, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "seg-max-gap"):
                    validate_config(AppConfig(input="in.mp4", output="out.mp4", seg_max_gap=value))

    def test_direct_config_rejects_invalid_age(self):
        for value in (0, -1, True, False, "0.1", None, float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "seg-max-age"):
                    validate_config(AppConfig(input="in.mp4", output="out.mp4", seg_max_age=value))

    def test_direct_config_accepts_positive_values(self):
        validate_config(AppConfig(input="in.mp4", output="out.mp4", seg_max_gap=1, seg_max_age=1))
        validate_config(AppConfig(input="in.mp4", output="out.mp4", seg_max_gap=6, seg_max_age=0.1))

    def test_thumbnail_reference_mode_and_validation(self):
        args = parse_args(["in.mp4", "out.mp4", "--seg-thumbnail-mode", "eager"])
        self.assertEqual(args.seg_thumbnail_mode, "eager")
        with self.assertRaisesRegex(ValueError, "seg-thumbnail-mode"):
            validate_config(AppConfig(input="in.mp4", output="out.mp4", seg_thumbnail_mode="fast"))


if __name__ == "__main__":
    unittest.main()
