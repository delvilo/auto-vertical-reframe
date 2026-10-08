"""Explicit display assumptions, converted to original-input pixels per degree."""
from __future__ import annotations

import math
from dataclasses import dataclass
from reframe.contracts import FrameContext
from reframe.geometry import compute_base_crop


@dataclass(frozen=True)
class ViewingGeometry:
    screen_inches: float = 24.0
    distance_cm: float = 60.0
    output_width: int = 1080
    output_height: int = 1920
    override_ppd: float | None = None

    def __post_init__(self):
        for value in (self.screen_inches, self.distance_cm, self.output_width, self.output_height):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Display size, distance and output dimensions must be finite and positive")
        if self.override_ppd is not None and (
            not math.isfinite(self.override_ppd) or self.override_ppd <= 0
        ):
            raise ValueError("pixel_per_dva override must be finite and positive")

    @property
    def output_ppd(self) -> float:
        # A 16:9 monitor rotated to portrait. Fit the output without stretching;
        # the usual 1080x1920 output fills the screen. One degree is measured at
        # the screen centre, rather than averaging nonlinear angles at its edges.
        diagonal_cm = self.screen_inches * 2.54
        width_cm = diagonal_cm * 9 / math.hypot(9, 16)
        height_cm = diagonal_cm * 16 / math.hypot(9, 16)
        pixels_per_cm = max(self.output_width / width_cm, self.output_height / height_cm)
        return pixels_per_cm * 2 * self.distance_cm * math.tan(math.radians(0.5))

    def input_ppd(self, frame: FrameContext) -> float:
        if self.override_ppd is not None:
            return self.override_ppd
        crop_h = frame.view_crop_height
        if crop_h is None:
            _, crop_h = compute_base_crop(frame.width, frame.height,
                                          self.output_width, self.output_height)
        if not 0 < crop_h <= frame.height:
            raise ValueError("Viewing crop height must be within the original frame")
        return self.output_ppd * crop_h / self.output_height
