from __future__ import annotations

import json
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
import platform
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import BDG2Data
from .energy import EnergyMeasurement, NvidiaSmiPowerSampler, gpu_index_from_device
from .models.base import Forecaster


RESULT_COLUMNS = [
    "model",
    "checkpoint",
    "input_mode",
    "history_months",
    "window",
    "cutoff",
    "meter_id",
    "meter_type",
    "rmse_kwh",
    "inference_seconds",
    "horizon_points",
    "requested_context_points",
    "effective_context_points",
]

ENERGY_SUMMARY_COLUMNS = (
    "measurement_protocol",
    "measurement_seconds",
    "codecarbon_interval_seconds",
    "codecarbon_energy_kwh",
    "codecarbon_energy_kwh_5s",
    "carbontracker_interval_seconds",
    "carbontracker_energy_kwh",
    "carbontracker_energy_kwh_1s",
    "system_energy_kwh_mean_cc_ct",
    "nvidia_smi_native_interval_ms",
    "nvidia_smi_energy_kwh",
    "nvidia_smi_energy_kwh_native",
    "nvidia_smi_energy_kwh_1s_decimated",
    "nvidia_smi_samples",
    "nvidia_smi_samples_1s_decimated",
    "system_energy_mwh_per_meter_window",
    "system_energy_wh_per_1000_forecast_points",
    "cc_ct_ratio",
    "nvidia_smi_mean_power_w",
    "nvidia_smi_peak_power_w",
    "nvidia_smi_power_cv",
)


def rolling_cutoffs(
    start: str, count: int, stride_hours: int = 168
) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=count, freq=f"{stride_hours}h")


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape:
        raise ValueError(f"Shape mismatch: truth {truth.shape}, prediction {prediction.shape}")
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _synchronize_cuda(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    torch.cuda.synchronize(torch.device(device))


def _set_batch_seed(forecaster: Forecaster, seed: int) -> None:
    setter = getattr(forecaster, "set_seed", None)
    if setter is not None:
        setter(seed)


def _runtime_metadata(device: str) -> dict[str, object]:
    metadata: dict[str, object] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    metadata["package_versions"] = {}
    for package in ("torch", "chronos-forecasting", "timesfm", "lightgbm", "scikit-learn", "codecarbon", "carbontracker"):
        try:
            metadata["package_versions"][package] = version(package)
        except PackageNotFoundError:
            pass
    # CPU tabular baselines do not need PyTorch. Do not load another native
    # OpenMP runtime merely to collect optional GPU metadata.
    if not device.startswith("cuda") and "torch" not in sys.modules:
        return metadata
    try:
        import torch
    except ImportError:
        return metadata
    metadata.update(
        {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        }
    )
    if device.startswith("cuda") and torch.cuda.is_available():
        index = torch.device(device).index
        index = torch.cuda.current_device() if index is None else index
        props = torch.cuda.get_device_properties(index)
        metadata.update(
            {
                "gpu_name": props.name,
                "gpu_total_vram_gib": round(props.total_memory / 2**30, 2),
                "gpu_compute_capability": f"{props.major}.{props.minor}",
            }
        )
    return metadata


def _window_data(
    data: BDG2Data,
    *,
    cutoff: pd.Timestamp,
    history_months: int,
    horizon: int,
) -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    target_index = pd.date_range(cutoff, periods=horizon, freq="h")
    missing_target = target_index.difference(data.values.index)
    if len(missing_target):
        raise ValueError(f"Target horizon exceeds available data at {cutoff}")
    context_start = cutoff - pd.Timedelta(days=30 * history_months)
    context = data.values.loc[
        (data.values.index >= context_start) & (data.values.index < cutoff)
    ]
    if context.empty:
        raise ValueError(f"No context for {cutoff} with {history_months} months")
    context_covariates = (
        data.covariates.loc[context.index]
        if data.covariates is not None else None
    )
    if pd.infer_freq(context.index) is None:
        regular_index = pd.date_range(
            context.index.min(), context.index.max(), freq="h"
        )
        context = context.reindex(regular_index).ffill().bfill()
        if context_covariates is not None:
            context_covariates = (
                context_covariates.reindex(regular_index).ffill().bfill()
            )
    requested_points = 24 * 30 * history_months
    if (
        len(context) != requested_points
        or context.index[0] != context_start
        or context.index[-1] != cutoff - pd.Timedelta(hours=1)
    ):
        raise ValueError(
            f"Expected {requested_points} hourly context points ending at "
            f"{cutoff}, found {len(context)}"
        )
    return target_index, context, context_covariates, data.values.loc[target_index]


def _warm_up(
    data: BDG2Data,
    forecaster: Forecaster,
    config: ExperimentConfig,
    meter_ids: list[str],
    cutoffs: pd.DatetimeIndex,
) -> None:
    if config.warmup_batches == 0:
        return
    target_index, context, context_covariates, _truth = _window_data(
        data,
        cutoff=cutoffs[0],
        history_months=config.history_months[0],
        horizon=config.horizon,
    )
    batches = list(_chunks(meter_ids, config.batch_size))
    for warmup_index in range(config.warmup_batches):
        batch_index = warmup_index % len(batches)
        batch_ids = batches[batch_index]
        contexts = [
            context[meter_id].to_numpy(dtype=np.float32) for meter_id in batch_ids
        ]
        batch_seed = (config.seed + batch_index) % (2**31 - 1)
        _set_batch_seed(forecaster, batch_seed)
        _synchronize_cuda(config.device)
        forecaster.forecast(
            contexts,
            config.horizon,
            series_ids=batch_ids,
            context_index=context.index,
            future_index=target_index,
            context_covariates=context_covariates,
            future_covariates=(
                data.covariates.loc[target_index]
                if data.covariates is not None else None
            ),
        )
        _synchronize_cuda(config.device)


def summarize(results_path: Path, output_dir: Path) -> tuple[Path, Path]:
    columns = [
            "model", "checkpoint", "input_mode", "history_months", "window",
            "meter_id", "meter_type", "rmse_kwh", "inference_seconds",
            "horizon_points",
            "requested_context_points", "effective_context_points",
        ]
    available = pd.read_csv(results_path, nrows=0).columns
    results = pd.read_csv(
        results_path, usecols=[column for column in columns if column in available]
    )
    if "requested_context_points" not in results:
        results["requested_context_points"] = 24 * 30 * results["history_months"]
    if "effective_context_points" not in results:
        results["effective_context_points"] = results["requested_context_points"]
    # Each row stores RMSE over one meter/window. Recover its sum of squared
    # errors so the summary reports the pooled RMSE over every forecast point.
    # Pooling is the only accuracy metric; no macro meter-window mean is kept.
    results["squared_error_sum_kwh2"] = (
        results["rmse_kwh"].pow(2) * results["horizon_points"]
    )
    per_type = (
        results.groupby(
            ["model", "checkpoint", "input_mode", "history_months", "meter_type"],
            as_index=False,
        )
        .agg(
            squared_error_sum_kwh2=("squared_error_sum_kwh2", "sum"),
            windows=("window", "nunique"),
            meters=("meter_id", "nunique"),
            meter_windows=("rmse_kwh", "size"),
            forecast_points=("horizon_points", "sum"),
            inference_seconds=("inference_seconds", "sum"),
            requested_context_points=("requested_context_points", "max"),
            effective_context_points=("effective_context_points", "max"),
        )
    )
    overall = (
        results.groupby(
            ["model", "checkpoint", "input_mode", "history_months"],
            as_index=False,
        )
        .agg(
            squared_error_sum_kwh2=("squared_error_sum_kwh2", "sum"),
            windows=("window", "nunique"),
            meters=("meter_id", "nunique"),
            inference_seconds=("inference_seconds", "sum"),
            meter_windows=("rmse_kwh", "size"),
            forecast_points=("horizon_points", "sum"),
            requested_context_points=("requested_context_points", "max"),
            effective_context_points=("effective_context_points", "max"),
        )
    )
    for summary, position in ((per_type, 5), (overall, 4)):
        summary.insert(
            position,
            "rmse_kwh",
            np.sqrt(
                summary["squared_error_sum_kwh2"] / summary["forecast_points"]
            ),
        )
        summary.drop(columns="squared_error_sum_kwh2", inplace=True)
    overall["meter_windows_per_second"] = (
        overall["meter_windows"] / overall["inference_seconds"]
    )
    overall["forecast_points_per_second"] = (
        overall["forecast_points"] / overall["inference_seconds"]
    )
    energy_path = output_dir / "energy_summary.json"
    if energy_path.is_file():
        energy = json.loads(energy_path.read_text(encoding="utf-8"))
        overall["energy_status"] = energy.get("status")
        for column in ENERGY_SUMMARY_COLUMNS:
            overall[column] = energy.get(column)
    training_path = output_dir / "training_windows.csv"
    if training_path.is_file():
        training = pd.read_csv(training_path)
        totals = training.groupby("history_months")["training_seconds"].sum()
        overall["training_seconds"] = overall.history_months.map(totals)
        per_type = per_type.merge(
            training.groupby(["history_months", "meter_type"], as_index=False)["training_seconds"].sum(),
            on=["history_months", "meter_type"], how="left",
        )
        # Not measured is deliberately distinct from zero energy.
        if not energy_path.is_file():
            overall["energy_status"] = "not_measured"
            overall["nvidia_smi_energy_kwh_native"] = np.nan
    per_type_path = output_dir / "summary_by_meter_type.csv"
    overall_path = output_dir / "summary_overall.csv"
    per_type.to_csv(per_type_path, index=False)
    overall.to_csv(overall_path, index=False)
    return per_type_path, overall_path


def _write_batch_results(
    batch_result: dict[str, object],
    *,
    data: BDG2Data,
    forecaster: Forecaster,
    config: ExperimentConfig,
    results_path: Path,
) -> None:
    batch_ids = list(batch_result["batch_ids"])
    truth = np.asarray(batch_result["truth"], dtype=np.float64)
    prediction = np.asarray(batch_result["prediction"], dtype=np.float64)
    history_months = int(batch_result["history_months"])
    window = int(batch_result["window"])
    cutoff = pd.Timestamp(batch_result["cutoff"])
    elapsed = float(batch_result["elapsed"])
    rows = []
    for row_index, meter_id in enumerate(batch_ids):
        rows.append(
            {
                "model": config.model,
                "checkpoint": forecaster.checkpoint,
                "input_mode": config.input_mode,
                "history_months": history_months,
                "window": window,
                "cutoff": cutoff.isoformat(),
                "meter_id": meter_id,
                "meter_type": data.meter_types[meter_id],
                "rmse_kwh": rmse(truth[row_index], prediction[row_index]),
                "inference_seconds": elapsed / len(batch_ids),
                "horizon_points": config.horizon,
                "requested_context_points": int(
                    batch_result["requested_context_points"]
                ),
                "effective_context_points": int(
                    batch_result["effective_context_points"]
                ),
            }
        )
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(
        results_path,
        mode="a",
        header=not results_path.exists() or results_path.stat().st_size == 0,
        index=False,
    )


def _write_forecast_arrays(
    batch_results: list[dict[str, object]],
    *,
    output_path: Path,
    history_months: tuple[int, ...],
    cutoffs: pd.DatetimeIndex,
    meter_ids: list[str],
    horizon: int,
) -> None:
    shape = (len(history_months), len(cutoffs), len(meter_ids), horizon)
    truth = np.full(shape, np.nan, dtype=np.float32)
    predictions = np.full(shape, np.nan, dtype=np.float32)
    history_index = {value: index for index, value in enumerate(history_months)}
    meter_index = {meter_id: index for index, meter_id in enumerate(meter_ids)}

    for result in batch_results:
        history_position = history_index[int(result["history_months"])]
        window = int(result["window"])
        batch_ids = list(result["batch_ids"])
        positions = [meter_index[meter_id] for meter_id in batch_ids]
        truth[history_position, window, positions, :] = np.asarray(
            result["truth"], dtype=np.float32
        )
        predictions[history_position, window, positions, :] = np.asarray(
            result["prediction"], dtype=np.float32
        )

    np.savez_compressed(
        output_path,
        truth=truth,
        predictions=predictions,
        history_months=np.asarray(history_months, dtype=np.int16),
        cutoffs=np.asarray([cutoff.isoformat() for cutoff in cutoffs]),
        meter_ids=np.asarray(meter_ids),
    )


class _StageTimer:
    """Opt-in per-stage wall-clock accumulation (no-op when disabled)."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.seconds: dict[str, float] = {}
        self.calls: dict[str, int] = {}

    @contextmanager
    def stage(self, name: str):
        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - started
            self.calls[name] = self.calls.get(name, 0) + 1


def _fit_supervised_window(data, forecaster, config, meter_ids, cutoff, history_months):
    """Only pre-origin data: extra 168 h supplies lags for the training rows."""
    print(f"Training {config.model}: cutoff={cutoff}, history={history_months}m", flush=True)
    train_start = cutoff - pd.Timedelta(days=30 * history_months)
    index = pd.date_range(train_start - pd.Timedelta(hours=168), cutoff,
                          freq="h", inclusive="left")
    values = data.values[meter_ids].reindex(index).ffill()
    covariates = data.covariates.reindex(index).ffill() if data.covariates is not None else None
    forecaster.fit_window(values, covariates, data.meter_types.loc[meter_ids], train_start)


def evaluate(
    data: BDG2Data,
    forecaster: Forecaster,
    config: ExperimentConfig,
    *,
    model_load_seconds: float | None = None,
) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = config.output_dir / "per_meter_window.csv"
    if (config.output_dir / "run_manifest.json").exists() or results_path.exists():
        raise ValueError(
            "Use a fresh --output-dir for a complete run. "
            "The grid skips completed configurations; partial runs are not resumed."
        )
    meter_ids = data.values.columns.tolist()
    if config.max_meters is not None:
        meter_ids = meter_ids[: config.max_meters]
    cutoffs = rolling_cutoffs(
        config.forecast_start, config.num_windows, config.stride_hours
    )
    supervised = getattr(forecaster, "requires_fit", False)
    if supervised and (len(config.history_months) != 1 or config.horizon > 168):
        raise ValueError("Supervised runs require one history and horizon <= 168 h")
    if config.model == "lightgbm" and config.energy_tool != "none":
        raise ValueError("LightGBM energy is not measured")

    run_manifest = {
        "site_id": config.site_id,
        "model": config.model,
        "checkpoint": forecaster.checkpoint,
        "input_mode": config.input_mode,
        "history_months": list(config.history_months),
        "forecast_start": config.forecast_start,
        "num_windows": config.num_windows,
        "horizon": config.horizon,
        "stride_hours": config.stride_hours,
        "missing_threshold": config.missing_threshold,
        "batch_size": config.batch_size,
        "device": config.device,
        "seed": config.seed,
        "num_samples": config.num_samples,
        "meters": len(meter_ids),
        "max_meters": config.max_meters,
        "meter_types": list(config.meter_types),
        "model_requested_max_context": getattr(
            forecaster, "requested_max_context", forecaster.max_context
        ),
        "model_max_context": forecaster.max_context,
        "model_internal_batch_size": getattr(
            forecaster, "internal_batch_size", config.batch_size
        ),
        "model_native_prediction_length": getattr(
            forecaster, "native_prediction_length", None
        ),
        "warmup_batches": config.warmup_batches,
        "energy_tool": config.energy_tool,
        "nvsmi_interval_ms": config.nvsmi_interval_ms,
        "model_load_seconds": model_load_seconds,
        "model_forecast_options": getattr(forecaster, "forecast_options", {}),
        **_runtime_metadata(config.device),
    }
    if config.model == "timesfm10":
        run_manifest["model_decoding_horizon"] = forecaster.decoding_horizon
    manifest_path = config.output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    training_rows = []
    prepared_models = []
    if supervised and config.energy_tool == "all":
        # Fail before fitting a year's models if GPU power cannot be read.
        # prepare() checks again after training; neither check starts monitoring.
        NvidiaSmiPowerSampler(
            config.output_dir / "energy" / "nvidia_smi_power.csv",
            interval_ms=config.nvsmi_interval_ms,
            gpu_index=gpu_index_from_device(config.device),
        ).validate()
        # MLPs are small enough to hold every origin's independently trained
        # network in memory. Only inference is inside the measurement below.
        for window, cutoff in enumerate(cutoffs):
            _fit_supervised_window(data, forecaster, config, meter_ids, cutoff, config.history_months[0])
            prepared_models.append(forecaster.models)
            training_rows.extend(dict(row, window=window, cutoff=cutoff.isoformat(),
                                      history_months=config.history_months[0])
                                 for row in forecaster.fit_details)
        pd.DataFrame(training_rows).to_csv(config.output_dir / "training_windows.csv", index=False)
        forecaster.models = prepared_models[0]

    forecasts_path = config.output_dir / "forecast_arrays.npz"
    energy: EnergyMeasurement | None = None
    if config.energy_tool == "all":
        energy = EnergyMeasurement(
            config.output_dir,
            interval_ms=config.nvsmi_interval_ms,
            gpu_index=gpu_index_from_device(config.device),
            metadata={
                "site_id": config.site_id,
                "model": config.model,
                "checkpoint": forecaster.checkpoint,
                "input_mode": config.input_mode,
                "history_months": list(config.history_months),
                "num_windows": config.num_windows,
                "meters": len(meter_ids),
                "meter_types": list(config.meter_types),
                "horizon": config.horizon,
                "stride_hours": config.stride_hours,
                "batch_size": config.batch_size,
                "model_internal_batch_size": getattr(
                    forecaster, "internal_batch_size", config.batch_size
                ),
                "model_max_context": forecaster.max_context,
                "num_samples": config.num_samples,
                "model_load_seconds_excluded": model_load_seconds,
                "training_seconds_excluded": sum(row["training_seconds"] for row in training_rows),
                "independent_window_models_prepared": bool(prepared_models),
                **_runtime_metadata(config.device),
            },
        )
        energy.prepare()

    if not supervised or prepared_models:
        _warm_up(data, forecaster, config, meter_ids, cutoffs)
    measurement_status = "failed"
    timer = _StageTimer(config.profile_stages)
    batch_results: list[dict[str, object]] = []
    if energy is not None:
        energy.start()
    try:
        for history_months in config.history_months:
            for window, cutoff in enumerate(cutoffs):
                if supervised:
                    if prepared_models:
                        forecaster.models = prepared_models[window]
                    else:
                        _fit_supervised_window(data, forecaster, config, meter_ids, cutoff, history_months)
                        training_rows.extend(dict(row, window=window, cutoff=cutoff.isoformat(),
                                                  history_months=history_months)
                                             for row in forecaster.fit_details)
                        if window == 0:
                            _warm_up(data, forecaster, config, meter_ids, cutoffs)
                with timer.stage("window_data"):
                    target_index, context, context_covariates, truth = _window_data(
                        data,
                        cutoff=cutoff,
                        history_months=history_months,
                        horizon=config.horizon,
                    )
                for batch_index, batch_ids in enumerate(
                    _chunks(meter_ids, config.batch_size)
                ):
                    with timer.stage("context_tensors"):
                        contexts = [
                            context[meter_id].to_numpy(dtype=np.float32)
                            for meter_id in batch_ids
                        ]
                    # Requested history is deliberately absent from the seed:
                    # capped models receive identical sampled forecasts for
                    # identical effective contexts.
                    batch_seed = (
                        config.seed + window * 10_007 + batch_index
                    ) % (2**31 - 1)
                    _set_batch_seed(forecaster, batch_seed)
                    with timer.stage("cuda_sync"):
                        _synchronize_cuda(config.device)
                    started = time.perf_counter()
                    with timer.stage("model_call"):
                        prediction = forecaster.forecast(
                            contexts,
                            config.horizon,
                            series_ids=batch_ids,
                            context_index=context.index,
                            future_index=target_index,
                            context_covariates=context_covariates,
                            future_covariates=(
                                data.covariates.loc[target_index]
                                if data.covariates is not None else None
                            ),
                        )
                    with timer.stage("cuda_sync"):
                        _synchronize_cuda(config.device)
                    elapsed = time.perf_counter() - started
                    with timer.stage("checks_collection"):
                        expected_shape = (len(batch_ids), config.horizon)
                        if prediction.shape != expected_shape:
                            raise ValueError(
                                f"{config.model} returned {prediction.shape}, "
                                f"expected {expected_shape}"
                            )
                        if not np.isfinite(prediction).all():
                            raise ValueError(
                                f"{config.model} returned non-finite predictions"
                            )
                        effective = min(
                            len(context), forecaster.max_context or len(context)
                        )
                        batch_result = {
                            "history_months": history_months,
                            "window": window,
                            "cutoff": cutoff,
                            "batch_ids": batch_ids,
                            "truth": truth.loc[:, batch_ids].to_numpy(dtype=np.float64).T,
                            "prediction": np.asarray(prediction).copy(),
                            "elapsed": elapsed,
                            "requested_context_points": len(context),
                            "effective_context_points": effective,
                        }
                    batch_results.append(batch_result)
        measurement_status = "completed"
    finally:
        if energy is not None:
            energy.stop(status=measurement_status)

    # RMSE and file writes are outside the measurement for every run.
    if supervised:
        pd.DataFrame(training_rows).to_csv(config.output_dir / "training_windows.csv", index=False)
    for batch_result in batch_results:
        _write_batch_results(
            batch_result,
            data=data,
            forecaster=forecaster,
            config=config,
            results_path=results_path,
        )
    if config.profile_stages:
        (config.output_dir / "profile.json").write_text(
            json.dumps(
                {
                    "stage_seconds": timer.seconds,
                    "stage_calls": timer.calls,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    if batch_results:
        _write_forecast_arrays(
            batch_results,
            output_path=forecasts_path,
            history_months=config.history_months,
            cutoffs=cutoffs,
            meter_ids=meter_ids,
            horizon=config.horizon,
        )

    summarize(results_path, config.output_dir)
    return results_path


def data_audit(data: BDG2Data, config: ExperimentConfig) -> dict[str, object]:
    cutoffs = rolling_cutoffs(
        config.forecast_start, config.num_windows, config.stride_hours
    )
    last_end = cutoffs[-1] + pd.Timedelta(hours=config.horizon)
    coverage_hours = int((last_end - cutoffs[0]) / pd.Timedelta(hours=1))
    available_end = data.values.index.max() + pd.Timedelta(hours=1)
    available_hours = int(
        (available_end - cutoffs[0]) / pd.Timedelta(hours=1)
    )
    max_windows = (
        (available_hours - config.horizon) // config.stride_hours + 1
    )
    return {
        "site_id": data.site_id,
        "rows": len(data.values),
        "meters": len(data.values.columns),
        "meters_by_type": data.meter_types.value_counts().sort_index().to_dict(),
        "selected_meter_types": list(config.meter_types),
        "first_timestamp": data.values.index.min().isoformat(),
        "last_timestamp": data.values.index.max().isoformat(),
        "mean_missing_fraction_before_imputation": float(data.missing_fraction.mean()),
        "max_missing_fraction_retained": float(data.missing_fraction.max()),
        "forecast_windows": len(cutoffs),
        "first_cutoff": cutoffs[0].isoformat(),
        "last_cutoff": cutoffs[-1].isoformat(),
        "last_forecast_end_exclusive": last_end.isoformat(),
        "evaluation_span_hours": coverage_hours,
        "available_forecast_hours": available_hours,
        "maximum_forecast_windows": max_windows,
        "horizon_hours": config.horizon,
        "stride_hours": config.stride_hours,
        "input_mode": config.input_mode,
        "covariates": (
            [] if data.covariates is None else data.covariates.columns.tolist()
        ),
    }
