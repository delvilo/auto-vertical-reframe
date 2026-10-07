"""Refresh policy, separate from frame ingestion and map propagation."""
from reframe.contracts import FrameObservations


class FixedIntervalScheduler:
    """Preserves refreshes at frames 1, 1+interval, ... within each scene."""

    def __init__(self, interval: int = 3):
        if interval < 1:
            raise ValueError("Saliency interval must be >= 1")
        self.interval = interval
        self.counter = 0

    def reset(self) -> None:
        self.counter = 0

    def should_refresh(self, observations: FrameObservations, missing: bool) -> bool:
        self.counter += 1
        return missing or (self.counter - 1) % self.interval == 0
