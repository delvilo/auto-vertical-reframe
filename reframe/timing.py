"""Host wall-clock accounting without accelerator synchronization.

These timers measure time observed by Python, not isolated GPU kernel duration.
Nested sections on the same accumulator are exclusive; separate accumulators
(pipeline, saliency cascade, neural backend) describe nested work and must not
be added together.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from time import perf_counter
from typing import Iterator


class HostTimings:
    """Accumulate exclusive sections, including time spent on failing calls."""

    def __init__(self, keys: tuple[str, ...]):
        self.seconds = dict.fromkeys(keys, 0.0)
        self._active: str | None = None
        self._started = 0.0

    @contextmanager
    def measure(self, key: str) -> Iterator[None]:
        if key not in self.seconds:
            raise KeyError(key)
        now = perf_counter()
        parent = self._active
        if parent is not None:
            self.seconds[parent] += now - self._started
        self._active, self._started = key, now
        try:
            yield
        finally:
            now = perf_counter()
            self.seconds[key] += now - self._started
            self._active, self._started = parent, now

    @property
    def total(self) -> float:
        return sum(self.seconds.values())


def timed_method(key: str):
    """Time a method using its instance's private ``_timings`` accumulator."""
    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            with self._timings.measure(key):
                return method(self, *args, **kwargs)
        return wrapped
    return decorate


class PhaseTimings:
    """Partition a sequential pipeline from its first timestamp through finish."""

    def __init__(self, keys: tuple[str, ...], initial: str, started: float):
        self.seconds = dict.fromkeys(keys, 0.0)
        if initial not in self.seconds:
            raise KeyError(initial)
        self._active: str | None = initial
        self._started = started

    def switch(self, key: str | None) -> float:
        if key is not None and key not in self.seconds:
            raise KeyError(key)
        now = perf_counter()
        if self._active is not None:
            self.seconds[self._active] += now - self._started
        self._active, self._started = key, now
        return now

    def snapshot(self) -> dict[str, float]:
        """Copy current phases without advancing or synchronizing an accelerator."""
        result = dict(self.seconds)
        if self._active is not None:
            result[self._active] += perf_counter() - self._started
        return result
