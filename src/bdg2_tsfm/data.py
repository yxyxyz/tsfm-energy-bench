from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


METER_TYPES = ("electricity", "chilledwater", "hotwater")


@dataclass(frozen=True)
class BDG2Data:
    """Hourly, imputed meter matrix, metadata, and site covariates."""

    values: pd.DataFrame
    meter_types: pd.Series
    missing_fraction: pd.Series
    covariates: pd.DataFrame | None = None
    site_id: str = "Fox"


def _infer_meter_type(path: Path) -> str:
    name = path.name.lower().replace("_", "")
    for meter_type in METER_TYPES:
        if meter_type in name:
            return meter_type
    raise ValueError(
        f"Cannot infer meter type from {path}. Expected electricity, "
        "chilledwater, or hotwater in the filename."
    )


def _resolve_paths(
    paths: Iterable[Path], meter_types: tuple[str, ...]
) -> list[Path]:
    resolved: list[Path] = []
    for path in paths:
        path = path.expanduser().resolve()
        if path.is_file():
            if (
                path.name.lower() != "weather.csv"
                and _infer_meter_type(path) in meter_types
            ):
                resolved.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)

        # The supplied experiment reads the official `*_cleaned.csv` files and
        # then applies its own 5% missingness rule and ffill/bfill once more.
        cleaned_dir = path / "data" / "meters" / "cleaned"
        search_dir = cleaned_dir if cleaned_dir.is_dir() else path
        for meter_type in meter_types:
            exact_candidates = (
                search_dir / f"{meter_type}_cleaned.csv",
                search_dir / f"{meter_type}.csv",
            )
            exact = next((candidate for candidate in exact_candidates if candidate.is_file()), None)
            if exact is not None:
                resolved.append(exact)
                continue

    unique = list(dict.fromkeys(resolved))
    found_types = {_infer_meter_type(path) for path in unique}
    missing = set(meter_types) - found_types
    if missing:
        raise FileNotFoundError(
            "Missing BDG2 meter CSV(s) for: " + ", ".join(sorted(missing))
        )
    return unique


def _read_wide_meter_file(path: Path, site_id: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if frame.shape[1] < 2:
        raise ValueError(f"Expected a timestamp plus meter columns in {path}")
    timestamp_col = "timestamp" if "timestamp" in frame.columns else frame.columns[0]
    timestamps = pd.to_datetime(frame.pop(timestamp_col), errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"Unparseable timestamps in {path}")
    if timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_localize(None)
    frame.index = pd.DatetimeIndex(timestamps, name="timestamp")
    if frame.index.has_duplicates:
        frame = frame.groupby(level=0).mean(numeric_only=True)
    site_columns = [
        column for column in frame.columns
        if str(column).startswith(f"{site_id}_")
    ]
    if not site_columns:
        raise ValueError(f"No {site_id}-site meter columns found in {path}")
    return frame[site_columns].apply(pd.to_numeric, errors="coerce")


def _resolve_weather_path(paths: Iterable[Path]) -> Path:
    candidates: list[Path] = []
    for supplied in paths:
        path = supplied.expanduser().resolve()
        if path.is_file() and path.name.lower() == "weather.csv":
            candidates.append(path)
        if not path.is_dir():
            continue
        candidates.extend(
            candidate
            for candidate in (
                path / "data" / "weather" / "weather.csv",
                path / "weather" / "weather.csv",
                path / "weather.csv",
            )
            if candidate.is_file()
        )
    if not candidates:
        raise FileNotFoundError(
            "Missing BDG2 weather.csv. ARX experiments require site-level "
            "airTemperature and calendar covariates."
        )
    return candidates[0]


def _load_site_covariates(
    path: Path, index: pd.DatetimeIndex, site_id: str
) -> pd.DataFrame:
    import holidays

    frame = pd.read_csv(path, low_memory=False)
    required = {"timestamp", "site_id", "airTemperature"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing weather columns in {path}: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError(f"Unparseable weather timestamps in {path}")
    if frame["timestamp"].dt.tz is not None:
        frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
    frame = frame.loc[
        frame["site_id"] == site_id, ["timestamp", "airTemperature"]
    ]
    frame = frame.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    frame = frame.loc[frame.index.intersection(index, sort=False)]
    frame["airTemperature"] = pd.to_numeric(
        frame["airTemperature"], errors="coerce"
    )
    if not frame["airTemperature"].notna().any():
        raise ValueError(f"{site_id} airTemperature has no usable observations")
    frame["weekday"] = frame.index.weekday.astype(float)
    frame["hour"] = frame.index.hour.astype(float)
    frame["is_weekend"] = frame["weekday"].isin([5.0, 6.0]).astype(float)
    holiday_dates = set(holidays.US(years=sorted(set(frame.index.year))).keys())
    frame["is_holiday"] = [float(ts.date() in holiday_dates) for ts in frame.index]
    return frame.astype("float32")


def load_site(
    paths: Iterable[Path],
    *,
    site_id: str = "Fox",
    missing_threshold: float = 0.05,
    meter_types: tuple[str, ...] = METER_TYPES,
) -> BDG2Data:
    """Load one BDG2 site and apply the reference >5% missingness filter.

    The official BDG2 files are wide CSVs. Missingness is measured on each
    cleaned file as in the supplied loader. The complete 2016-2017 hourly index
    is then restored and retained series are imputed with forward-fill followed
    by back-fill, matching Section 2.1 of the paper.
    """

    if not 0 <= missing_threshold < 1:
        raise ValueError("missing_threshold must be in [0, 1)")
    resolved = _resolve_paths(paths, meter_types)
    full_index = pd.date_range(
        "2016-01-01", "2018-01-01", freq="h", inclusive="left", name="timestamp"
    )
    covariates = _load_site_covariates(
        _resolve_weather_path(paths), full_index, site_id
    )
    frames: list[pd.DataFrame] = []
    type_parts: list[pd.Series] = []
    missing_parts: list[pd.Series] = []

    for path in resolved:
        meter_type = _infer_meter_type(path)
        frame = _read_wide_meter_file(path, site_id)
        # The official cleaned BDG2 headers identify the building but not the
        # resource.  The supplied BuildSys loader appends the filename-derived
        # meter type before concatenating the three wide tables; ARX adapters
        # also use this suffix as static categorical metadata.
        suffix = f"_{meter_type}"
        frame = frame.rename(
            columns=lambda column: (
                str(column)
                if str(column).endswith(suffix)
                else f"{column}{suffix}"
            )
        )
        missing = frame.isna().mean()
        keep = missing[missing <= missing_threshold].index
        frame = frame.loc[:, keep].reindex(full_index).ffill().bfill()
        if frame.isna().any().any():
            bad = frame.columns[frame.isna().any()].tolist()
            raise ValueError(f"Could not impute all values for meters: {bad[:5]}")
        frames.append(frame.astype("float32"))
        type_parts.append(pd.Series(meter_type, index=keep, dtype="string"))
        missing_parts.append(missing.loc[keep])

    values = pd.concat(frames, axis=1)
    if values.columns.duplicated().any():
        duplicates = values.columns[values.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate meter identifiers across files: {duplicates[:5]}")
    if values.empty:
        raise ValueError(f"No {site_id} meters passed the missingness filter")
    # Match the source loader's inner timestamp merge with Fox weather. The
    # official weather table has one missing Fox hour in 2016. Context windows
    # containing that gap are regularized in evaluation, just as the supplied
    # Chronos adapter calls `ensure_regular_freq` immediately before inference.
    values = values.loc[covariates.index]
    values = values.sort_index(axis=1)
    return BDG2Data(
        values=values,
        meter_types=pd.concat(type_parts).reindex(values.columns),
        missing_fraction=pd.concat(missing_parts).reindex(values.columns),
        covariates=covariates,
        site_id=site_id,
    )


def load_fox_site(
    paths: Iterable[Path],
    *,
    missing_threshold: float = 0.05,
    meter_types: tuple[str, ...] = METER_TYPES,
) -> BDG2Data:
    """Backward-compatible loader for the original BuildSys Fox protocol."""

    return load_site(
        paths,
        site_id="Fox",
        missing_threshold=missing_threshold,
        meter_types=meter_types,
    )
