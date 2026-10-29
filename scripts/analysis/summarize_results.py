#!/usr/bin/env python
"""Summarize all RMSE result CSVs under a results directory.

Scans recursively for ``per_meter_window.csv`` files, reads each run's existing
``summary_overall.csv``, and prints one combined RMSE matrix across all runs.
Pass ``--regenerate-run-summaries`` only when the per-run summaries themselves
need to be rebuilt.

Usage:
    python scripts/analysis/summarize_results.py --results-dir results
    python scripts/analysis/summarize_results.py --results-dir results --out combined.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_COLUMNS = [
    "forecast_start",
    "horizon",
    "stride_hours",
    "num_windows",
    "meter_types",
    "meters",
    "batch_size",
    "device",
    "gpu_name",
    "energy_tool",
    "measurement_protocol",
    "num_samples",
    "seed",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Directory tree containing per_meter_window.csv files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional path to write the combined per-run summary CSV.",
    )
    parser.add_argument(
        "--effective-out",
        type=Path,
        help=(
            "Effective-context table (default: "
            "<results-dir>/summary_by_effective_context.csv)."
        ),
    )
    parser.add_argument(
        "--regenerate-run-summaries",
        action="store_true",
        help="Rebuild summary_overall.csv and summary_by_meter_type.csv per run.",
    )
    return parser


def _parameter_lookup() -> dict[str, float]:
    path = PROJECT_ROOT / "configs" / "models.json"
    return {
        str(entry["checkpoint"]): float(entry["parameters_m"])
        for entry in json.loads(path.read_text(encoding="utf-8"))
        if "parameters_m" in entry
    }


def _effective_context_table(combined: pd.DataFrame) -> pd.DataFrame:
    for column in (
        "nvidia_smi_energy_kwh_native",
        "rmse_kwh",
        "inference_seconds",
    ):
        if column not in combined:
            combined[column] = pd.NA
    keys = [
        "site_id", "model", "checkpoint", "input_mode", "parameters_m",
        "effective_context_points", *PROTOCOL_COLUMNS,
    ]
    grouped = combined.groupby(keys, dropna=False, as_index=False)
    table = grouped.agg(
        requested_history_months=(
            "history_months",
            lambda values: ",".join(str(int(value)) for value in sorted(set(values))),
        ),
        repeat_count=("history_months", "size"),
        rmse_kwh_mean=("rmse_kwh", "mean"),
        rmse_kwh_sd=("rmse_kwh", "std"),
        inference_seconds_mean=("inference_seconds", "mean"),
        inference_seconds_sd=("inference_seconds", "std"),
        gpu_energy_kwh_mean=("nvidia_smi_energy_kwh_native", "mean"),
        gpu_energy_kwh_sd=("nvidia_smi_energy_kwh_native", "std"),
    )
    table["gpu_energy_wh_mean"] = 1000 * table["gpu_energy_kwh_mean"]
    table["gpu_energy_wh_sd"] = 1000 * table["gpu_energy_kwh_sd"]
    return table.sort_values(keys, na_position="last")


def _add_context_columns(df: pd.DataFrame, per_meter_path: Path) -> pd.DataFrame:
    """Read context metadata without rewriting an existing run summary."""
    required = {"requested_context_points", "effective_context_points"}
    missing = required.difference(df.columns)
    if not missing:
        return df

    columns = ["history_months", *sorted(missing)]
    context = pd.read_csv(per_meter_path, usecols=columns).drop_duplicates()
    return df.merge(context, on="history_months", how="left", validate="many_to_one")


def _run_metadata(
    manifest: dict[str, object], per_meter_path: Path
) -> tuple[dict[str, object], int]:
    available = pd.read_csv(per_meter_path, nrows=0).columns
    usecols = [column for column in ("meter_type", "cutoff") if column in available]
    rows = pd.read_csv(per_meter_path, usecols=usecols)
    meter_types = ",".join(sorted(rows["meter_type"].dropna().astype(str).unique()))
    stride_hours = manifest.get("stride_hours")
    if stride_hours is None and "cutoff" in rows:
        cutoffs = pd.DatetimeIndex(pd.to_datetime(rows["cutoff"].dropna().unique())).sort_values()
        if len(cutoffs) > 1:
            strides = pd.Series(cutoffs[1:] - cutoffs[:-1]).drop_duplicates()
            if len(strides) == 1:
                stride_hours = int(strides.iloc[0] / pd.Timedelta(hours=1))
    energy_path = per_meter_path.parent / "energy_summary.json"
    energy = json.loads(energy_path.read_text()) if energy_path.is_file() else {}
    metadata = {
        "site_id": manifest.get("site_id", "Fox"),
        "forecast_start": manifest.get("forecast_start"),
        "horizon": manifest.get("horizon"),
        "stride_hours": stride_hours,
        "num_windows": manifest.get("num_windows"),
        "meter_types": meter_types,
        "meters": manifest.get("meters"),
        "batch_size": manifest.get("batch_size"),
        "device": manifest.get("device"),
        "gpu_name": manifest.get("gpu_name"),
        "energy_tool": manifest.get("energy_tool", "none"),
        "measurement_protocol": energy.get("measurement_protocol", "none"),
        "num_samples": manifest.get("num_samples"),
        "seed": manifest.get("seed"),
    }
    return metadata, len(rows)


def _invalid_run_reason(
    manifest: dict[str, object], output_dir: Path, actual_rows: int
) -> str | None:
    if not manifest:
        return "missing run_manifest.json"
    histories = manifest.get("history_months")
    if not isinstance(histories, list):
        return "manifest has no history_months list"
    expected_rows = (
        int(manifest.get("meters", 0))
        * int(manifest.get("num_windows", 0))
        * len(histories)
    )
    if actual_rows != expected_rows:
        return f"row count {actual_rows} != expected {expected_rows}"

    if manifest.get("energy_tool") == "all":
        energy_path = output_dir / "energy_summary.json"
        if not energy_path.is_file():
            return "formal run has no energy_summary.json"
        energy = json.loads(energy_path.read_text(encoding="utf-8"))
        if energy.get("status") != "completed":
            return f"energy status is {energy.get('status')!r}"
        if str(manifest.get("device", "")).startswith("cuda"):
            if manifest.get("cuda_available") is not True:
                return "CUDA was requested but unavailable"
    return None


def main() -> None:
    args = build_parser().parse_args()

    # Make the package importable without pip install -e .
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from bdg2_tsfm.evaluation import summarize

    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")

    csv_files = sorted(results_dir.rglob("per_meter_window.csv"))
    if not csv_files:
        raise SystemExit(f"No per_meter_window.csv found under {results_dir}")

    overviews: list[pd.DataFrame] = []
    excluded: list[tuple[Path, str]] = []
    parameters = _parameter_lookup()
    for path in csv_files:
        out_dir = path.parent
        manifest_path = out_dir / "run_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file() else {}
        )
        metadata, actual_rows = _run_metadata(manifest, path)
        invalid_reason = _invalid_run_reason(manifest, out_dir, actual_rows)
        if invalid_reason is not None:
            excluded.append((out_dir, invalid_reason))
            print(
                f"[excluded] {out_dir.relative_to(results_dir)}: {invalid_reason}",
                file=sys.stderr,
            )
            continue
        overall_path = out_dir / "summary_overall.csv"
        if args.regenerate_run_summaries or not overall_path.is_file():
            _per_type_path, overall_path = summarize(path, out_dir)
            action = "rebuilt"
        else:
            action = "read"
        df = pd.read_csv(overall_path)
        # In the original experiment repo, rmse_kwh was the macro metric.
        # Its explicitly named pooled column is authoritative when present.
        if "rmse_pooled_kwh" in df:
            df["rmse_kwh"] = df["rmse_pooled_kwh"]
            df = df.drop(columns=["rmse_pooled_kwh", "rmse_macro_meter_window_kwh"], errors="ignore")
        df = _add_context_columns(df, path)
        print(f"[{action}] {path.relative_to(results_dir)} -> {overall_path.name}")
        df["dir"] = str(out_dir.relative_to(results_dir))
        for key, value in metadata.items():
            df[key] = value
        df["parameters_m"] = df["checkpoint"].map(parameters)
        overviews.append(df)

    if not overviews:
        raise SystemExit("No complete, valid runs found")
    combined = pd.concat(overviews, ignore_index=True)
    matrix = (
        combined.pivot_table(
            index=["model", "checkpoint", "input_mode", "dir"],
            columns="history_months",
            values="rmse_kwh",
            aggfunc="mean",
        )
        .round(4)
        .sort_index()
    )

    print("\n=== Combined RMSE matrix (kWh); columns = history months ===")
    print(matrix.to_string())
    print(
        f"\nSummarized {len(overviews)} run(s); excluded {len(excluded)} "
        f"under {results_dir}"
    )

    if args.out is not None:
        combined.to_csv(args.out, index=False)
        print(f"Combined detail written to {args.out}")

    effective_path = (
        args.effective_out
        if args.effective_out is not None
        else results_dir / "summary_by_effective_context.csv"
    )
    effective = _effective_context_table(combined)
    effective.to_csv(effective_path, index=False)
    print(f"Effective-context table written to {effective_path}")


if __name__ == "__main__":
    main()
