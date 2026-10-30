from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def _xreg_inputs(
    contexts: Sequence[np.ndarray],
    horizon: int,
    metadata: dict[str, Any],
    max_context: int,
) -> tuple[
    list[np.ndarray], dict[str, list[np.ndarray]], dict[str, list[str]]
]:
    context_covariates = metadata.get("context_covariates")
    future_covariates = metadata.get("future_covariates")
    series_ids = metadata.get("series_ids")
    if context_covariates is None or future_covariates is None or series_ids is None:
        raise ValueError("ARX mode requires context/future covariates and series_ids")
    trimmed = [np.asarray(x[-max_context:], dtype=np.float32) for x in contexts]
    dynamic: dict[str, list[np.ndarray]] = {}
    for column in context_covariates.columns:
        combined = np.concatenate(
            [
                context_covariates[column].to_numpy(dtype=np.float32)[-max_context:],
                future_covariates[column].to_numpy(dtype=np.float32)[:horizon],
            ]
        )
        dynamic[column] = [combined.copy() for _ in trimmed]
    static = {
        "meter_type": [str(value).rsplit("_", 1)[-1] for value in series_ids]
    }
    return trimmed, dynamic, static


class TimesFM10Forecaster:
    """Adapter for the archived TimesFM 1.0 API (timesfm==1.3.0)."""

    name = "timesfm10"
    max_context = 512

    def __init__(
        self, *, checkpoint: str, device: str, input_mode: str,
        batch_size: int, horizon: int,
    ) -> None:
        import timesfm

        self.checkpoint = checkpoint
        self.input_mode = input_mode
        self.internal_batch_size = batch_size
        self.decoding_horizon = horizon
        backend = "gpu" if device.startswith("cuda") else "cpu"
        self.model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=backend,
                per_core_batch_size=batch_size,
                # Requested forecast length; the pretrained output patch stays 128.
                horizon_len=horizon,
                context_len=self.max_context,
                num_layers=20,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=checkpoint),
        )

    def forecast(
        self, contexts: Sequence[np.ndarray], horizon: int, **metadata: Any
    ) -> np.ndarray:
        trimmed = [np.asarray(x[-self.max_context :], dtype=np.float32) for x in contexts]
        if self.input_mode == "arx":
            trimmed, dynamic, static = _xreg_inputs(
                contexts, horizon, metadata, self.max_context
            )
            point, _xreg = self.model.forecast_with_covariates(
                trimmed,
                dynamic_numerical_covariates=dynamic,
                static_categorical_covariates=static,
                freq=[0] * len(trimmed),
                xreg_mode="xreg + timesfm",
            )
            return np.asarray(point, dtype=np.float32)[:, :horizon]
        point, _quantiles = self.model.forecast(trimmed, freq=[0] * len(trimmed))
        return np.asarray(point[:, :horizon], dtype=np.float32)


class TimesFM25Forecaster:
    """Adapter for the current TimesFM 2.5 PyTorch API."""

    name = "timesfm25"

    def __init__(
        self,
        *,
        checkpoint: str,
        device: str,
        input_mode: str,
        max_context: int,
        batch_size: int,
    ) -> None:
        import timesfm
        import torch

        self.requested_max_context = max_context
        self.max_context = math.ceil(max_context / 32) * 32
        self.internal_batch_size = batch_size
        model_class = timesfm.TimesFM_2p5_200M_torch
        self.checkpoint = checkpoint
        self.input_mode = input_mode
        self.model = model_class.from_pretrained(checkpoint)
        requested_device = torch.device(device if device.startswith("cuda") else "cpu")
        self.model.model.to(requested_device)
        self.model.model.device = requested_device
        self.model.model.device_count = 1
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=self.max_context,
                max_horizon=256,
                per_core_batch_size=self.internal_batch_size,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
                return_backcast=input_mode == "arx",
            )
        )

    def forecast(
        self, contexts: Sequence[np.ndarray], horizon: int, **metadata: Any
    ) -> np.ndarray:
        inputs = [np.asarray(x[-self.max_context :], dtype=np.float32) for x in contexts]
        if self.input_mode == "arx":
            inputs, dynamic, static = _xreg_inputs(
                contexts, horizon, metadata, self.max_context
            )
            point, _quantiles = self.model.forecast_with_covariates(
                inputs,
                dynamic_numerical_covariates=dynamic,
                static_categorical_covariates=static,
                xreg_mode="xreg + timesfm",
            )
            return np.asarray(point, dtype=np.float32)[:, :horizon]
        point, _quantiles = self.model.forecast(horizon=horizon, inputs=inputs)
        return np.asarray(point[:, :horizon], dtype=np.float32)


TIMESFM30_MAX_CONTEXT = 15_360


class TimesFM30Forecaster:
    """Adapter for the TimesFM 3.0 PyTorch API (``timesfm3``).

    TimesFM 3.0 takes past-only and past-and-future numerical covariate
    channels instead of the XReg static/dynamic split used by 1.0 and 2.5, so
    ARX mode passes the evaluation covariates as past-and-future channels.
    Contexts retain at most the requested history and the 15,360-point limit.
    The library pads each batch to a multiple of 32 internally. Each meter is
    an independent target with its own covariate channels, not a target channel
    in a joint multimeter forecast.
    """

    name = "timesfm30"
    native_prediction_length = 64
    # Forecaster defaults, explicitly recorded to distinguish them from the
    # official Evaluator's symmetric averaging and non-negativity settings.
    forecast_options = {
        "return_quantiles": False,
        "use_symmetric_averaging": False,
        "make_positive": False,
        "sort_quantiles": True,
        "use_znorm": False,
        "padding_mode": "none",
    }

    def __init__(
        self,
        *,
        checkpoint: str,
        device: str,
        input_mode: str,
        max_context: int,
        batch_size: int,
    ) -> None:
        import timesfm3

        self.requested_max_context = max_context
        self.max_context = min(max_context, TIMESFM30_MAX_CONTEXT)
        self.internal_batch_size = batch_size
        self.checkpoint = checkpoint
        self.input_mode = input_mode
        self.model = timesfm3.TimesFM3Forecaster.from_pretrained(
            checkpoint,
            device=device if device.startswith("cuda") else "cpu",
            per_core_batch_size=batch_size,
        )

    def forecast(
        self, contexts: Sequence[np.ndarray], horizon: int, **metadata: Any
    ) -> np.ndarray:
        inputs = [
            np.asarray(x[-self.max_context :], dtype=np.float32) for x in contexts
        ]
        covariates = None
        if self.input_mode == "arx":
            past = metadata["context_covariates"].iloc[-self.max_context :]
            future = metadata["future_covariates"].loc[:, past.columns].iloc[:horizon]
            channels = np.concatenate(
                [past.to_numpy(dtype=np.float32), future.to_numpy(dtype=np.float32)]
            ).T
            covariates = [channels] * len(inputs)
        outputs = self.model.predict_batch(
            contexts=inputs,
            horizon=horizon,
            past_future_covariates=covariates,
            **self.forecast_options,
        )
        return np.stack(
            [
                np.asarray(output.forecast, dtype=np.float32)[:horizon]
                for output in outputs
            ]
        )
