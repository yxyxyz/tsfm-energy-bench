from __future__ import annotations

from dataclasses import dataclass

from .base import Forecaster


@dataclass(frozen=True)
class ModelSpec:
    checkpoint: str
    package: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "chronos": ModelSpec("amazon/chronos-t5-base", "chronos"),
    "chronos-bolt": ModelSpec("amazon/chronos-bolt-base", "chronos"),
    "chronos2": ModelSpec("amazon/chronos-2", "chronos"),
    "timesfm10": ModelSpec("google/timesfm-1.0-200m-pytorch", "timesfm10"),
    "timesfm25": ModelSpec("google/timesfm-2.5-200m-pytorch", "timesfm25"),
    "timesfm30": ModelSpec("google/timesfm-3.0-pytorch", "timesfm30"),
    "persistence-daily": ModelSpec("seasonal-naive-lag-24h", "persistence"),
    "persistence-weekly": ModelSpec("seasonal-naive-lag-168h", "persistence"),
    "lightgbm": ModelSpec("lightgbm", "supervised"),
    "mlp": ModelSpec("mlp", "supervised"),
}

ARX_MODELS = frozenset({"chronos2", "timesfm10", "timesfm25", "timesfm30", "lightgbm", "mlp"})


def resolve_input_mode(name: str, requested: str) -> str:
    if requested not in {"auto", "ar", "arx"}:
        raise ValueError(f"Unknown input mode {requested!r}")
    if requested == "auto":
        return "arx" if name in ARX_MODELS else "ar"
    if requested == "arx" and name not in ARX_MODELS:
        raise ValueError(
            f"{name} does not support covariate-informed forecasting; use AR"
        )
    return requested


def create_forecaster(
    name: str,
    *,
    checkpoint: str | None,
    device: str,
    num_samples: int,
    seed: int,
    input_mode: str,
    batch_size: int,
    max_context: int,
) -> Forecaster:
    if name not in MODEL_SPECS:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_SPECS)}")
    spec = MODEL_SPECS[name]
    input_mode = resolve_input_mode(name, input_mode)
    checkpoint = checkpoint or spec.checkpoint
    if spec.package == "supervised":
        from .supervised import SupervisedForecaster

        return SupervisedForecaster(
            name=name, input_mode=input_mode, seed=seed, device=device
        )
    if spec.package == "chronos":
        from .chronos import ChronosForecaster

        return ChronosForecaster(
            name=name,
            checkpoint=checkpoint,
            device=device,
            num_samples=num_samples,
            seed=seed,
            input_mode=input_mode,
        )
    if spec.package == "timesfm10":
        from .timesfm import TimesFM10Forecaster

        return TimesFM10Forecaster(
            checkpoint=checkpoint, device=device, input_mode=input_mode
        )
    if spec.package == "persistence":
        from .persistence import SeasonalPersistenceForecaster

        lag_hours = 24 if name == "persistence-daily" else 168
        return SeasonalPersistenceForecaster(name=name, lag_hours=lag_hours)
    if spec.package == "timesfm30":
        from .timesfm import TimesFM30Forecaster

        return TimesFM30Forecaster(
            checkpoint=checkpoint,
            device=device,
            input_mode=input_mode,
            max_context=max_context,
            batch_size=batch_size,
        )
    from .timesfm import TimesFM25Forecaster

    return TimesFM25Forecaster(
        checkpoint=checkpoint,
        device=device,
        input_mode=input_mode,
        max_context=max_context,
        batch_size=batch_size,
    )
