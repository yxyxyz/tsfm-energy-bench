from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np


class Forecaster(Protocol):
    name: str
    checkpoint: str
    max_context: int | None

    def forecast(
        self, contexts: Sequence[np.ndarray], horizon: int, **metadata: Any
    ) -> np.ndarray:
        """Return point forecasts shaped [series, horizon]."""
        ...
