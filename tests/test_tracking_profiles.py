"""Tracking profile resolution and backward-compatible CLI migration."""
import contextlib
import io
import math
from types import SimpleNamespace
import unittest

from reframe.cli import LEGACY_OPTIONS, parse_args
from reframe.config import AppConfig, PRESETS, apply_preset, tracking_settings, validate_config


class TrackingProfileTests(unittest.TestCase):
    def parse(self, *flags):
        return parse_args(["input.mp4", "output.mp4", *flags])

    def test_profile_resolves_the_coordinated_limits(self):
        expected = {
            "conservative": (3, 0.1, 0.2, 0.2, 3),
            "balanced": (6, 0.1, 0.2, 0.35, 3),
            "fast": (6, 0.1, 0.25, 0.4, 3),
        }
        for name, values in expected.items():
            with self.subTest(profile=name):
                args = self.parse("--tracking-profile", name)
                self.assertEqual(
                    (args.seg_max_gap, args.seg_max_age, args.cue_interval,
                     args.pose_max_age, args.saliency_interval), values,
                )
                self.assertEqual(args.seg_skip_policy, "primary")
                self.assertEqual(args.dead_zone, 0)
        self.assertEqual(self.parse().tracking_profile, "balanced")

    def test_explicit_values_override_profile_in_either_order(self):
        # These equal old defaults; comparison with defaults cannot detect them.
        override = ["--seg-max-gap", "3", "--seg-max-age", "0.07",
                    "--cue-interval", "0.2", "--pose-max-age", "0.15",
                    "--saliency-interval", "1", "--seg-skip-policy", "all"]
        profile = ["--tracking-profile", "fast"]
        first = self.parse(*profile, *override)
        second = self.parse(*override, *profile)
        self.assertEqual(tracking_settings(first), tracking_settings(second))
        self.assertEqual(tracking_settings(first), {
            "tracking_profile": "fast", "seg_max_gap": 3, "seg_max_age": 0.07,
            "cue_interval": 0.2, "pose_max_age": 0.15, "saliency_interval": 1,
            "seg_skip_policy": "all",
        })

    def test_direct_constructor_honors_profile_and_overrides(self):
        conservative = AppConfig(tracking_profile="conservative")
        fast = AppConfig(tracking_profile="fast", cue_interval=0.2, seg_max_gap=3)
        self.assertEqual(conservative.seg_max_gap, 3)
        self.assertEqual(conservative.pose_max_age, 0.2)
        self.assertEqual(fast.seg_max_gap, 3)
        self.assertEqual(fast.cue_interval, 0.2)
        self.assertEqual(fast.pose_max_age, 0.4)

    def test_presets_and_camera_overrides_are_independent(self):
        for preset in PRESETS:
            for profile in ("conservative", "balanced", "fast"):
                with self.subTest(preset=preset, profile=profile):
                    args = apply_preset(self.parse("--preset", preset,
                                                  "--tracking-profile", profile))
                    self.assertEqual(args.dead_zone, 0)
                    self.assertEqual(args.max_zoom, PRESETS[preset]["max_zoom"])
                    override = apply_preset(self.parse("--preset", preset,
                        "--tracking-profile", profile, "--dead-zone", "0.08"))
                    self.assertEqual(override.dead_zone, 0.08)
                    self.assertEqual(tracking_settings(args), tracking_settings(override))

    def test_zero_and_hard_limit_shorter_than_refresh_are_allowed(self):
        args = self.parse("--cue-interval", "0.4", "--pose-max-age", "0")
        self.assertEqual(args.pose_max_age, 0)
        args = self.parse("--cue-interval", "0", "--pose-max-age", "0.1")
        self.assertEqual(args.cue_interval, 0)

    def test_invalid_tracking_values(self):
        for key in ("cue_interval", "pose_max_age"):
            for value in (None, True, False, "0.2", -0.1, math.inf, -math.inf, math.nan):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_config(AppConfig(input="in", output="out", **{key: value}))
        for value in (None, True, False, "3", 0, -1, 3.5, math.inf, math.nan):
            with self.subTest(saliency_interval=value), self.assertRaises(ValueError):
                validate_config(AppConfig(input="in", output="out", saliency_interval=value))
        for flags in (("--tracking-profile", "unknown"), ("--pose-max-age", "nan"),
                      ("--pose-max-age", "inf"), ("--pose-max-age", "-0.1"),
                      ("--cue-interval", "nan"), ("--saliency-interval", "0")):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    self.parse(*flags)
                self.assertEqual(raised.exception.code, 2)
        with self.assertRaisesRegex(ValueError, "tracking-profile"):
            AppConfig(tracking_profile="unknown")

    def test_effective_settings_support_collectors_without_all_fields(self):
        self.assertEqual(tracking_settings(SimpleNamespace()), tracking_settings(AppConfig()))
        namespace = SimpleNamespace(tracking_profile="fast", cue_interval=None, seg_max_gap=1)
        result = tracking_settings(namespace)
        self.assertEqual(result["cue_interval"], 0.25)
        self.assertEqual(result["pose_max_age"], 0.4)
        self.assertEqual(result["seg_max_gap"], 1)


class LegacyParameterTests(unittest.TestCase):
    def parse(self, *flags):
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            args = apply_preset(parse_args(["in.mp4", "out.mp4", *flags]))
        return args, output.getvalue()

    def test_legacy_conversion_retains_numeric_behavior_and_one_notice(self):
        args, notice = self.parse(
            "--motion-response", "0.16", "--motion-damping", "0.46",
            "--zoom-response", "0.1", "--zoom-damping", "0.45",
            "--target-alpha", "0.2", "--zoom-alpha", "0.07",
            "--max-step-x", "10", "--max-step-y", "8",
            "--max-missed-frames", "75", "--min-subject-hold-frames", "9",
            "--saliency-trust-repo",
        )
        self.assertAlmostEqual(args.pan_time, 0.35 * math.sqrt(0.08 / 0.16) * 0.5)
        self.assertAlmostEqual(args.zoom_time, 0.6 * math.sqrt(0.05 / 0.1) * 0.5)
        self.assertAlmostEqual(args.target_time, -(1 / 30) / math.log1p(-0.2))
        self.assertAlmostEqual(args.zoom_speed, 2.1)
        self.assertAlmostEqual(args.pan_speed_x, 10 * 30 / 1920)
        self.assertAlmostEqual(args.pan_speed_y, 8 * 30 / 1080)
        self.assertEqual(args.lost_hold_seconds, 2.5)
        self.assertEqual(args.subject_hold_seconds, 0.3)
        self.assertTrue(args.saliency_trust_repo)
        self.assertEqual(len(notice.splitlines()), 1)
        self.assertEqual(set(args.legacy_flags), set(LEGACY_OPTIONS))
        for old, new in LEGACY_OPTIONS.items():
            self.assertIn(f"{old} -> {new}", notice)

    def test_modern_values_take_priority_regardless_of_argument_order(self):
        legacy = ["--motion-response", "0.4", "--motion-damping", "0.4",
                  "--zoom-response", "0.4", "--zoom-damping", "0.4",
                  "--target-alpha", "0.4", "--zoom-alpha", "0.4",
                  "--max-step-x", "4", "--max-step-y", "4",
                  "--max-missed-frames", "4", "--min-subject-hold-frames", "4"]
        overrides = {"pan_time": 0.22, "zoom_time": 0.33, "target_time": 0.44,
                     "pan_speed_x": 0.55, "pan_speed_y": 0.66, "zoom_speed": 0.77,
                     "lost_hold_seconds": 0.88, "subject_hold_seconds": 0.99}
        modern = [item for key, value in overrides.items()
                  for item in ("--" + key.replace("_", "-"), str(value))]
        for flags in (legacy + modern, modern + legacy):
            args, notice = self.parse(*flags)
            for key, value in overrides.items():
                self.assertEqual(getattr(args, key), value)
            self.assertEqual(len(notice.splitlines()), 1)

    def test_default_and_modern_commands_have_no_migration_notice(self):
        args, notice = self.parse()
        self.assertEqual(notice, "")
        self.assertEqual(args.legacy_flags, ())
        args, notice = self.parse("--pan-time", "0.4")
        self.assertEqual(notice, "")
        self.assertEqual(args.pan_time, 0.4)

    def test_repeated_legacy_flag_is_reported_once(self):
        args, notice = self.parse("--zoom-alpha", ".1", "--zoom-alpha=.2")
        self.assertEqual(args.zoom_speed, 6)
        self.assertEqual(args.legacy_flags, ("--zoom-alpha",))
        self.assertEqual(notice.count("--zoom-alpha"), 1)

    def test_help_promotes_modern_options_and_hides_legacy_flags(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            parse_args(["--help"])
        self.assertEqual(raised.exception.code, 0)
        help_text = output.getvalue()
        for flag in LEGACY_OPTIONS:
            self.assertNotIn(flag, help_text)
        for flag in ("--tracking-profile", "--pose-max-age", "--pan-time",
                     "--zoom-speed", "--dead-zone", "--lost-hold-seconds"):
            self.assertIn(flag, help_text)


if __name__ == "__main__":
    unittest.main()
