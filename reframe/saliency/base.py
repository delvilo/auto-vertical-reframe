from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
from reframe.contracts import BackendPrediction, FrameContext
if TYPE_CHECKING:
    import numpy as np


class SaliencyBackend(ABC):
    """observe receives every BGR uint8 frame, including skipped predictions.

    The factory selects original or reduced input geometry. MSDB uses original
    frames; handcrafted can use reduced frames.

    FrameContext always describes the original image; predictions span the input
    image and the service maps them to original coordinates. reset drops temporal
    state, not lifetime counters. load may choose a visible fallback; close frees
    resources. No backend-specific dispatch is permitted in the service.
    """
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def observe(self, frame: np.ndarray, context: FrameContext) -> None: ...

    @abstractmethod
    def predict(self, frame: np.ndarray, context: FrameContext) -> BackendPrediction: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def telemetry(self) -> dict[str, Any]: ...
