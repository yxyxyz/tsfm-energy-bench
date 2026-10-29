from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class SeasonalPersistenceForecaster:
    """Repeat the observations from one day or one week earlier."""

    def __init__(self, *, name: str, lag_hours: int) -> None:
        self.name = name
        self.checkpoint = f"seasonal-naive-lag-{lag_hours}h"
        self.lag_hours = lag_hours
        self.max_context = lag_hours
        self.internal_batch_size = None
        self.native_prediction_length = lag_hours

    def forecast(
        self, contexts: Sequence[np.ndarray], horizon: int, **metadata: Any
    ) -> np.ndarray:
        if horizon > self.lag_hours:
            raise ValueError(
                f"{self.name} requires horizon <= lag ({self.lag_hours} h)"
            )
        predictions = []
        for context in contexts:
            values = np.asarray(context, dtype=np.float32)
            if len(values) < self.lag_hours:
                raise ValueError(
                    f"{self.name} requires at least {self.lag_hours} context points"
                )
            start = len(values) - self.lag_hours
            predictions.append(values[start : start + horizon])
        return np.stack(predictions)
