"""Run one BDG2 forecasting experiment."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import (
    ExperimentConfig, PAPER_HISTORY_MONTHS, PAPER_NUM_WINDOWS,
    PAPER_HORIZON, PAPER_STRIDE_HOURS,
)
from .data import METER_TYPES, load_site
from .evaluation import data_audit, evaluate
from .models import MODEL_SPECS, create_forecaster, resolve_input_mode


DEFAULT_OUTPUT_ROOT = Path("results")


def meter_scope(meter_types: tuple[str, ...]) -> str:
    return "-".join(meter_types)


def resolve_output_dir(
    model: str,
    requested: Path | None,
    meter_types: tuple[str, ...],
    site_id: str = "Fox",
) -> Path:
    if requested is not None:
        return requested
    root = DEFAULT_OUTPUT_ROOT if site_id == "Fox" else DEFAULT_OUTPUT_ROOT / site_id
    return root / meter_scope(meter_types) / model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate TSFMs and supervised baselines on one BDG2 site."
    )
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="BDG2 repository root or cleaned meter CSVs plus weather.csv.",
    )
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument(
        "--site", default="Fox", help="BDG2 site identifier (default: Fox)."
    )
    parser.add_argument("--checkpoint", help="Override the official default model ID.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Result directory (default: results/<meter-type>/<model>).",
    )
    parser.add_argument(
        "--history-months",
        type=int,
        nargs="+",
        default=[PAPER_HISTORY_MONTHS[0]],
        help="One input-history length per run (default: 1 month).",
    )
    parser.add_argument("--num-windows", type=int, default=PAPER_NUM_WINDOWS)
    parser.add_argument("--horizon", type=int, default=PAPER_HORIZON)
    parser.add_argument(
        "--stride-hours",
        type=int,
        default=PAPER_STRIDE_HOURS,
        help="Hours between forecast origins (default: 24).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--profile-stages", action="store_true",
                        help="record per-stage wall-clock timings to "
                             "profile.json")
    parser.add_argument(
        "--input-mode",
        choices=("auto", "ar", "arx"),
        default="auto",
        help=(
            "auto uses ARX wherever supported and AR otherwise; "
            "use ar explicitly for the common target-only ablation."
        ),
    )
    parser.add_argument("--max-meters", type=int)
    parser.add_argument(
        "--meter-types",
        nargs="+",
        choices=METER_TYPES,
        default=["electricity"],
        help="Meter types to retain (default: electricity).",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=1,
        help="Untimed warm-up batches before measured inference (default: 1).",
    )
    parser.add_argument(
        "--energy-tool",
        choices=("none", "all"),
        default="none",
        help=(
            "all measures the complete inference/evaluation loop with "
            "CodeCarbon, CarbonTracker, and an nvidia-smi power trace"
        ),
    )
    parser.add_argument(
        "--nvsmi-interval-ms",
        type=int,
        default=100,
        help="nvidia-smi power.draw.instant sampling interval (default: 100 ms).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and audit data/splits without importing or downloading a model.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.stride_hours <= 0:
        raise SystemExit("--stride-hours must be positive")
    if args.energy_tool != "none" and args.warmup_batches < 1:
        raise SystemExit("formal energy measurement requires --warmup-batches >= 1")
    if args.model.startswith("persistence-") and args.energy_tool != "none":
        raise SystemExit("persistence baselines do not use GPU energy measurement")
    if args.model == "lightgbm" and args.energy_tool != "none":
        raise SystemExit("LightGBM energy is not measured; use --energy-tool none")
    if args.model in {"lightgbm", "mlp"} and args.horizon > 168:
        raise SystemExit("BuildSys weekly-lag baselines require --horizon <= 168")
    if args.model == "mlp" and (args.horizon < 24 or args.horizon % 24):
        raise SystemExit("MLP requires --horizon 24, 48, ..., 168 (complete days)")
    if args.model == "mlp" and args.energy_tool == "all" and not args.device.startswith("cuda"):
        raise SystemExit("MLP GPU energy measurement requires --device cuda")
    if len(args.history_months) != 1:
        raise SystemExit(
            "each run requires exactly one --history-months value so model "
            "context configuration matches the requested input length"
        )
    try:
        input_mode = resolve_input_mode(args.model, args.input_mode)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    config = ExperimentConfig(
        data_paths=tuple(args.data),
        output_dir=resolve_output_dir(
            args.model, args.output_dir, tuple(args.meter_types), args.site
        ),
        model=args.model,
        site_id=args.site,
        checkpoint=args.checkpoint,
        history_months=tuple(args.history_months),
        num_windows=args.num_windows,
        horizon=args.horizon,
        stride_hours=args.stride_hours,
        batch_size=args.batch_size,
        device="cpu" if args.model.startswith("persistence-") or args.model == "lightgbm" else args.device,
        seed=args.seed,
        num_samples=args.num_samples,
        max_meters=args.max_meters,
        meter_types=tuple(args.meter_types),
        warmup_batches=args.warmup_batches,
        energy_tool=args.energy_tool,
        nvsmi_interval_ms=args.nvsmi_interval_ms,
        input_mode=input_mode,
        profile_stages=args.profile_stages,
    )
    data = load_site(
        config.data_paths,
        site_id=config.site_id,
        missing_threshold=config.missing_threshold,
        meter_types=config.meter_types,
    )
    audit = data_audit(data, config)
    print(json.dumps(audit, indent=2, sort_keys=True))
    coverage_hours = int(audit["evaluation_span_hours"])
    available_hours = int(audit["available_forecast_hours"])
    max_windows = int(audit["maximum_forecast_windows"])
    if config.num_windows > max_windows:
        overrun_hours = coverage_hours - available_hours
        raise SystemExit(
            f"--num-windows {config.num_windows} overruns the data: the last "
            f"forecast ends {overrun_hours} h past the available target range. "
            f"Maximum for stride {config.stride_hours} h and horizon "
            f"{config.horizon} h is --num-windows {max_windows}."
        )
    if (
        config.energy_tool != "none"
        and coverage_hours < 0.9 * available_hours
    ):
        print(
            f"WARNING: evaluation spans only {coverage_hours} h "
            f"({100 * coverage_hours / available_hours:.0f}% of the available "
            f"target range). For the longest evaluation at stride "
            f"{config.stride_hours} h and horizon {config.horizon} h, use "
            f"--num-windows {max_windows}.",
            file=sys.stderr,
        )
    if args.dry_run:
        return
    if config.device.startswith("cuda") and config.model not in {
        "persistence-daily", "persistence-weekly"
    }:
        import torch

        if not torch.cuda.is_available():
            raise SystemExit(
                f"--device {config.device} requested, but CUDA is unavailable"
            )
        try:
            torch.empty(1, device=config.device)
        except RuntimeError as exc:
            raise SystemExit(
                f"--device {config.device} could not execute a CUDA allocation: {exc}"
            ) from exc
    load_started = time.perf_counter()
    forecaster = create_forecaster(
        config.model,
        checkpoint=config.checkpoint,
        device=config.device,
        num_samples=config.num_samples,
        seed=config.seed,
        input_mode=config.input_mode,
        batch_size=config.batch_size,
        max_context=24 * 30 * max(config.history_months),
    )
    model_load_seconds = time.perf_counter() - load_started
    results = evaluate(
        data, forecaster, config, model_load_seconds=model_load_seconds
    )
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
