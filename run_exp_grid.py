#!/usr/bin/env python3
"""Run the experiment grid sequentially."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the BDG2 experiment grid sequentially."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--site", default="Fox")
    parser.add_argument(
        "--results-dir", type=Path, default=PROJECT_ROOT / "results"
    )
    parser.add_argument(
        "--grid", choices=("base", "full", "baselines"), default="base"
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        help="Advanced override for the model-grid JSON file.",
    )
    parser.add_argument("--modern-env", default="cikm-tsfm-modern")
    parser.add_argument("--baselines-env", default="cikm-baselines")
    parser.add_argument("--timesfm10-env", default="cikm-timesfm10")
    parser.add_argument(
        "--timesfm30-env",
        default="cikm-timesfm30",
        help="Conda environment that has the timesfm 3.0 package installed.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        help=(
            "Only run selected model families, for example: "
            "--families chronos or --families timesfm10 timesfm25 timesfm30."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument(
        "--history-months",
        type=int,
        nargs="+",
        default=[1, 3, 6, 9, 12],
        help=(
            "History settings. Each value runs in a separate process so the "
            "model context configuration matches that input length."
        ),
    )
    parser.add_argument("--num-windows", type=int, default=365)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument(
        "--stride-hours",
        type=int,
        default=24,
        help="Hours between forecast origins (24 for daily rolling forecasts).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-meters", type=int)
    parser.add_argument(
        "--meter-types",
        nargs="+",
        choices=("electricity", "chilledwater", "hotwater"),
        default=["electricity"],
        help="Meter types to retain (default: electricity).",
    )
    parser.add_argument(
        "--energy-tool",
        choices=("none", "all"),
        default="all",
        help="all enables CodeCarbon, CarbonTracker, and nvidia-smi measurement.",
    )
    parser.add_argument("--nvsmi-interval-ms", type=int, default=100)
    parser.add_argument(
        "--modes",
        choices=("default", "both"),
        default="default",
        help="Use each checkpoint's default mode, or run every supported AR/ARX mode.",
    )
    parser.add_argument(
        "--include-scope-review",
        action="store_true",
        help="Include derivative checkpoints marked scope_review in the matrix.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print commands without creating output directories or running models.",
    )
    return parser


def _output_name(
    checkpoint: str,
    input_mode: str,
    *,
    batch_size: int,
    history_months: list[int] | None = None,
) -> str:
    short_name = {
        "amazon/chronos-2": "chronos2",
        "google/timesfm-1.0-200m-pytorch": "timesfm10",
        "google/timesfm-2.5-200m-pytorch": "timesfm25",
        "google/timesfm-3.0-pytorch": "timesfm30",
    }.get(checkpoint, checkpoint.rsplit("/", maxsplit=1)[-1])
    name = f"{short_name}-{input_mode}"
    if history_months is not None and len(history_months) == 1:
        name += f"-h{history_months[0]:02d}m"
    return f"{name}-b{batch_size}"


def _environment(args: argparse.Namespace, family: str) -> str:
    if family in {"lightgbm", "mlp"}:
        return args.baselines_env
    if family == "timesfm10":
        return args.timesfm10_env
    if family == "timesfm30":
        return args.timesfm30_env
    return args.modern_env


def _command(
    entry: dict[str, object],
    input_mode: str,
    history_months: list[int],
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    family = str(entry["family"])
    environment = _environment(args, family)
    command = [
        "conda", "run", "--no-capture-output", "-n", environment,
        "python", "-m", "bdg2_tsfm.main",
        "--data", str(args.data.resolve()),
        "--site", args.site,
        "--model", family,
        "--checkpoint", str(entry["checkpoint"]),
        "--input-mode", input_mode,
        "--history-months", *[str(value) for value in history_months],
        "--num-windows", str(args.num_windows),
        "--horizon", str(args.horizon),
        "--stride-hours", str(args.stride_hours),
        "--batch-size", str(args.batch_size),
        "--meter-types", *args.meter_types,
        "--warmup-batches", str(args.warmup_batches),
        "--num-samples", str(args.num_samples),
        "--device", str(entry.get("device", args.device)),
        "--energy-tool", str(entry.get("energy_tool", args.energy_tool)),
        "--nvsmi-interval-ms", str(args.nvsmi_interval_ms),
        "--output-dir", str(output_dir.resolve()),
    ]
    if args.max_meters is not None:
        command.extend(("--max-meters", str(args.max_meters)))
    return command


def _is_complete(
    output_dir: Path, energy_tool: str, expected_config: dict[str, object]
) -> bool:
    summary_path = output_dir / "summary_overall.csv"
    results_path = output_dir / "per_meter_window.csv"
    forecasts_path = output_dir / "forecast_arrays.npz"
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(manifest.get(key) != value for key, value in expected_config.items()):
        return False
    if not results_path.is_file() or not forecasts_path.is_file():
        return False
    expected = (
        int(manifest["meters"])
        * int(manifest["num_windows"])
        * len(manifest["history_months"])
    )
    with results_path.open(encoding="utf-8") as handle:
        rows = sum(1 for _line in handle) - 1
    if rows != expected:
        return False
    if energy_tool == "all":
        energy_path = output_dir / "energy_summary.json"
        trace_path = output_dir / "energy" / "nvidia_smi_power.csv"
        if (
            not energy_path.is_file()
            or not summary_path.is_file()
            or not trace_path.is_file()
        ):
            return False
        summary = json.loads(energy_path.read_text(encoding="utf-8"))
        return (
            summary.get("status") == "completed"
            and "codecarbon_energy_kwh" in summary
            and "carbontracker_energy_kwh" in summary
            and "nvidia_smi_energy_kwh" in summary
        )

    if not summary_path.is_file():
        return False
    return rows == expected


def _filter_entries(
    entries: list[dict[str, object]], families: list[str] | None
) -> list[dict[str, object]]:
    if not families:
        return entries
    available = {str(entry["family"]) for entry in entries}
    unknown = set(families) - available
    if unknown:
        raise SystemExit(
            f"Unknown model families: {sorted(unknown)}; "
            f"choose from {sorted(available)}"
        )
    selected = set(families)
    return [entry for entry in entries if str(entry["family"]) in selected]


def _distinct_histories(
    entry: dict[str, object], history_months: list[int]
) -> list[int]:
    """Keep one requested history for each distinct model-visible context."""

    cap = entry.get("max_context_points")
    seen: set[int] = set()
    selected: list[int] = []
    for months in history_months:
        requested = 24 * 30 * months
        effective = min(requested, int(cap)) if cap is not None else requested
        if effective not in seen:
            selected.append(months)
            seen.add(effective)
    return selected


def main() -> None:
    args = build_parser().parse_args()
    if args.stride_hours <= 0:
        raise SystemExit("--stride-hours must be positive")
    if not args.data.is_dir():
        raise SystemExit(f"BDG2 root not found: {args.data}")

    matrix_path = args.matrix or PROJECT_ROOT / "configs" / "models.json"
    entries = json.loads(matrix_path.read_text(encoding="utf-8"))
    if args.matrix is None:
        entries = [entry for entry in entries if args.grid in entry["grids"]]
    entries = _filter_entries(entries, args.families)
    commands: list[tuple[list[str], Path]] = []
    skipped: list[Path] = []
    for entry in entries:
        if entry.get("scope_review") and not args.include_scope_review:
            continue
        modes = (
            entry["input_modes"]
            if args.modes == "both"
            else [entry["default_input_mode"]]
        )
        selected_histories = _distinct_histories(entry, args.history_months)
        dropped_histories = [
            value for value in args.history_months if value not in selected_histories
        ]
        if dropped_histories:
            print(
                f"[context dedup] {entry['checkpoint']}: kept "
                f"{selected_histories}; dropped {dropped_histories} because "
                "they duplicate an earlier effective context.",
                flush=True,
            )
        history_groups = [[value] for value in selected_histories]
        for mode in modes:
            for history_months in history_groups:
                output_dir = (
                    (args.results_dir if args.site == "Fox" else args.results_dir / args.site)
                    / "-".join(args.meter_types)
                    / _output_name(
                        str(entry["checkpoint"]),
                        str(mode),
                        batch_size=args.batch_size,
                        history_months=history_months,
                    )
                )
                expected_config = {
                    "checkpoint": str(entry["checkpoint"]),
                    "input_mode": str(mode),
                    "history_months": history_months,
                    "num_windows": args.num_windows,
                    "horizon": args.horizon,
                    "stride_hours": args.stride_hours,
                    "batch_size": args.batch_size,
                    "meter_types": args.meter_types,
                    "warmup_batches": args.warmup_batches,
                    "num_samples": args.num_samples,
                    "device": str(entry.get("device", args.device)),
                    "energy_tool": str(entry.get("energy_tool", args.energy_tool)),
                    "nvsmi_interval_ms": args.nvsmi_interval_ms,
                    "max_meters": args.max_meters,
                    "site_id": args.site,
                }
                if str(entry["family"]) == "timesfm10":
                    expected_config.update(
                        model_internal_batch_size=args.batch_size,
                        model_decoding_horizon=args.horizon,
                    )
                if str(entry["family"]) in {"timesfm25", "timesfm30"}:
                    requested_context = 24 * 30 * max(history_months)
                    model_context = math.ceil(requested_context / 32) * 32
                    if str(entry["family"]) == "timesfm30":
                        model_context = min(requested_context, 15_360)
                    expected_config.update(
                        {
                            "model_requested_max_context": requested_context,
                            "model_max_context": model_context,
                            "model_internal_batch_size": args.batch_size,
                        }
                    )
                if _is_complete(output_dir, str(expected_config["energy_tool"]), expected_config):
                    skipped.append(output_dir)
                    continue
                commands.append(
                    (
                        _command(
                            entry, str(mode), history_months, output_dir, args
                        ),
                        output_dir,
                    )
                )

    print(
        f"Prepared {len(commands)} sequential runs; skipped {len(skipped)} complete.",
        flush=True,
    )
    for run_number, (command, output_dir) in enumerate(commands, start=1):
        print(f"[{run_number}/{len(commands)}] {output_dir.name}", flush=True)
        if args.print_only:
            print(" ".join(command), flush=True)
        else:
            subprocess.run(command, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
