from __future__ import annotations

import logging
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd


class ChronosForecaster:
    """One adapter for original Chronos, Chronos-Bolt, and Chronos-2."""

    def __init__(
        self,
        *,
        name: str,
        checkpoint: str,
        device: str,
        num_samples: int,
        seed: int,
        input_mode: str,
    ) -> None:
        import torch
        from chronos import BaseChronosPipeline

        self.name = name
        self.checkpoint = checkpoint
        self.num_samples = num_samples
        self.input_mode = input_mode
        self._torch = torch
        self.set_seed(seed)

        self.pipeline = BaseChronosPipeline.from_pretrained(checkpoint, device_map=device)
        # Repeated Chronos horizon warnings would add logging I/O to the energy
        # run. The 168-hour horizon remains recorded in the manifest.
        pipeline_module = sys.modules.get(self.pipeline.__class__.__module__)
        pipeline_file = getattr(pipeline_module, "__file__", None)
        logging.getLogger(self.pipeline.__class__.__module__).setLevel(logging.ERROR)
        if pipeline_file:
            logging.getLogger(pipeline_file).setLevel(logging.ERROR)
        documented_context = {
            "chronos": 512,
            "chronos-bolt": 2048,
            "chronos2": 8192,
        }
        self.max_context = getattr(self.pipeline, "model_context_length", None)
        if self.max_context is None:
            self.max_context = documented_context[name]
        self.native_prediction_length = getattr(
            self.pipeline, "model_prediction_length", None
        )
        if self.native_prediction_length is None:
            model = getattr(self.pipeline, "model", None)
            model_config = getattr(model, "config", None)
            self.native_prediction_length = getattr(
                model_config, "prediction_length", None
            )

    def set_seed(self, seed: int) -> None:
        self._torch.manual_seed(seed)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(seed)

    def _forecast_chronos2_dataframe(
        self,
        contexts: Sequence[np.ndarray],
        horizon: int,
        metadata: dict[str, Any],
    ) -> np.ndarray:
        series_ids = [str(value) for value in metadata["series_ids"]]
        context_index = pd.DatetimeIndex(metadata["context_index"])[
            -self.max_context :
        ]
        contexts = [
            np.asarray(values)[-self.max_context :] for values in contexts
        ]
        future_index = pd.DatetimeIndex(metadata["future_index"])
        context_covariates = metadata.get("context_covariates")
        if context_covariates is not None:
            context_covariates = context_covariates.iloc[-self.max_context :]
        future_covariates = metadata.get("future_covariates")
        if len(contexts) != len(series_ids):
            raise ValueError("Chronos-2 series_ids do not match the context batch")

        context_parts = []
        future_parts = []
        for series_id, values in zip(series_ids, contexts):
            part = pd.DataFrame(
                {
                    "building_id": series_id,
                    "timestamp": context_index,
                    "target": np.asarray(values, dtype=np.float32),
                }
            )
            if context_covariates is not None:
                for column in context_covariates.columns:
                    part[column] = context_covariates[column].to_numpy()
            # Present in the original context_df but intentionally dropped from
            # future_df, making it a past-only categorical covariate.
            part["meter_type"] = series_id.rsplit("_", 1)[-1]
            context_parts.append(part)

            future = pd.DataFrame(
                {"building_id": series_id, "timestamp": future_index}
            )
            if future_covariates is not None:
                for column in future_covariates.columns:
                    future[column] = future_covariates[column].to_numpy()
            future_parts.append(future)

        pred_df = self.pipeline.predict_df(
            pd.concat(context_parts, ignore_index=True),
            future_df=pd.concat(future_parts, ignore_index=True),
            prediction_length=horizon,
            quantile_levels=[0.5],
            id_column="building_id",
            timestamp_column="timestamp",
            target="target",
            cross_learning=True,
        )
        pred_df = pred_df.copy()
        pred_df["building_id"] = pred_df["building_id"].astype(str)
        pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
        indexed = pred_df.set_index(["building_id", "timestamp"])["predictions"]
        return np.stack(
            [
                indexed.loc[series_id].reindex(future_index).to_numpy(dtype=np.float32)
                for series_id in series_ids
            ]
        )

    def forecast(
        self, contexts: Sequence[np.ndarray], horizon: int, **metadata: Any
    ) -> np.ndarray:
        if self.name == "chronos2" and self.input_mode == "arx":
            return self._forecast_chronos2_dataframe(contexts, horizon, metadata)
        tensors = [
            self._torch.as_tensor(
                x[-self.max_context :], dtype=self._torch.float32
            )
            for x in contexts
        ]
        kwargs: dict[str, object] = {}
        if self.name == "chronos":
            kwargs["num_samples"] = self.num_samples
        with self._torch.inference_mode():
            quantiles, _mean = self.pipeline.predict_quantiles(
                tensors,
                prediction_length=horizon,
                quantile_levels=[0.5],
                **kwargs,
            )
        if isinstance(quantiles, list):
            # Chronos-2 returns one [1, horizon, quantile] tensor per series.
            # Stack the list to the [series, horizon, quantile] layout used by
            # this evaluator. The official GluonTS adapter swaps the final two
            # axes only because QuantileForecast expects [quantile, horizon].
            quantiles = np.stack(quantiles).squeeze(axis=1)
        result = quantiles[..., 0]
        if hasattr(result, "detach"):
            result = result.detach().cpu().numpy()
        return np.asarray(result, dtype=np.float32)
