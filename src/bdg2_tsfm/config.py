from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PAPER_HISTORY_MONTHS = (1, 3, 6, 9, 12)
PAPER_FORECAST_START = "2017-01-01"
PAPER_NUM_WINDOWS = 365
PAPER_HORIZON = 24
PAPER_STRIDE_HOURS = 24


@dataclass(frozen=True)
class ExperimentConfig:
    data_paths: tuple[Path, ...]
    output_dir: Path
    model: str
    site_id: str = "Fox"
    checkpoint: str | None = None
    history_months: tuple[int, ...] = (1,)
    forecast_start: str = PAPER_FORECAST_START
    num_windows: int = PAPER_NUM_WINDOWS
    horizon: int = PAPER_HORIZON
    stride_hours: int = PAPER_STRIDE_HOURS
    missing_threshold: float = 0.05
    batch_size: int = 32
    device: str = "cuda"
    seed: int = 17
    num_samples: int = 20
    max_meters: int | None = None
    meter_types: tuple[str, ...] = ("electricity",)
    warmup_batches: int = 1
    energy_tool: str = "none"
    nvsmi_interval_ms: int = 100
    # Stored configs always contain the resolved mode; CLI `auto` is resolved
    # before this dataclass is constructed.
    input_mode: str = "ar"
    profile_stages: bool = False
