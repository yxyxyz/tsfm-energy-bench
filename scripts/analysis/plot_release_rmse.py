#!/usr/bin/env python3
"""Plot model release date and GPU energy against day-ahead pooled RMSE.

The input is the effective-context summary written by
``scripts/analysis/summarize_results.py``. Two figures are produced in the output
directory:

- ``release_date_rmse.{png,pdf}``: pooled RMSE per public release event;
- ``energy_rmse.{png,pdf}``: pooled RMSE against GPU energy per run.

Usage:
    python scripts/analysis/plot_release_rmse.py
    python scripts/analysis/plot_release_rmse.py \
        --summary results/summary_by_effective_context.csv --out-dir figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# matplotlib is imported inside the plotting functions so that this module can
# be imported (and unit-tested) in environments that only run the evaluation.


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RMSE_COLUMN = "rmse_kwh_mean"
ENERGY_COLUMN = "gpu_energy_wh_mean"

COLORS = {
    "chronos": "#C76D3A",
    "timesfm10": "#7651A8",
    "chronos-bolt": "#2E8B57",
    "timesfm25": "#D84A73",
    "chronos2": "#2F6FA3",
    "timesfm30": "#0F8B8D",
}
FALLBACK_COLOR = "#666666"

LABELS = {
    "chronos-t5-tiny": "T5-tiny",
    "chronos-t5-mini": "T5-mini",
    "chronos-t5-small": "T5-small",
    "chronos-t5-base": "T5-base",
    "chronos-t5-large": "T5-large",
    "chronos-bolt-tiny": "Bolt-tiny",
    "chronos-bolt-mini": "Bolt-mini",
    "chronos-bolt-small": "Bolt-small",
    "chronos-bolt-base": "Bolt-base",
    "chronos-2": "Chronos-2",
    "chronos-2-small": "Chronos-2-small",
    "chronos-2-synth": "Chronos-2-synth",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "results" / "summary_by_effective_context.csv",
        help="Effective-context summary CSV from scripts/analysis/summarize_results.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "figures",
        help="Directory that receives the PNG/PDF figures.",
    )
    parser.add_argument(
        "--release-dates",
        type=Path,
        default=Path(__file__).with_name("model_release_dates.json"),
    )
    parser.add_argument("--site", default="Fox")
    parser.add_argument(
        "--protocol",
        default="h24-electricity",
        choices=("h24-electricity", "any"),
        help=(
            "h24-electricity (default) keeps the day-ahead 24 h electricity "
            "protocol: horizon 24, stride 24, 365 origins, electricity meters. "
            "Use any to plot every protocol in the summary."
        ),
    )
    return parser


PROTOCOL_COLUMNS = {
    "horizon": 24,
    "stride_hours": 24,
    "num_windows": 365,
    "meter_types": "electricity",
}


def _restrict_protocol(data: pd.DataFrame, protocol: str) -> pd.DataFrame:
    if protocol == "any":
        return data
    for column, expected in PROTOCOL_COLUMNS.items():
        if column not in data:
            raise SystemExit(
                f"Summary has no {column!r} column; regenerate it with "
                "scripts/analysis/summarize_results.py"
            )
        data = data.loc[data[column] == expected]
    return data


def release_axis(release_dates: dict[str, str]) -> tuple[dict[str, int], list[str]]:
    """Map checkpoint -> evenly spaced release-event index plus tick labels."""
    ordered = sorted(set(release_dates.values()))
    positions = {date: index for index, date in enumerate(ordered)}
    labels = [_event_label(date) for date in ordered]
    index = {
        checkpoint: positions[date] for checkpoint, date in release_dates.items()
    }
    return index, labels


def _event_label(iso_date: str) -> str:
    stamp = pd.Timestamp(iso_date)
    return f"{stamp.strftime('%b')} {stamp.year}"


def load_data(
    summary_path: Path,
    release_dates_path: Path,
    site: str,
    protocol: str = "h24-electricity",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if not summary_path.is_file():
        raise SystemExit(
            f"Summary not found: {summary_path}\n"
            "Run scripts/analysis/summarize_results.py first."
        )
    data = pd.read_csv(summary_path)
    if "site_id" in data:
        data = data.loc[data["site_id"] == site].copy()
    if data.empty:
        raise SystemExit(f"No rows found for site {site} in {summary_path}")
    if RMSE_COLUMN not in data:
        raise SystemExit(
            f"{summary_path} has no {RMSE_COLUMN!r} column; regenerate it with "
            "the current scripts/analysis/summarize_results.py"
        )
    data = _restrict_protocol(data, protocol)
    if data.empty:
        raise SystemExit(
            f"No rows found for site {site} and protocol {protocol} in "
            f"{summary_path}"
        )
    if RMSE_COLUMN not in data:
        raise SystemExit(
            f"{summary_path} has no {RMSE_COLUMN!r} column; regenerate it with "
            "the current scripts/analysis/summarize_results.py"
        )

    release_dates = json.loads(release_dates_path.read_text(encoding="utf-8"))
    release_index, tick_labels = release_axis(release_dates)

    baselines = data.loc[data["model"].str.startswith("persistence-")].copy()
    models = data.loc[~data["model"].str.startswith("persistence-")].copy()
    known = models["checkpoint"].isin(release_index)
    if (~known).any():
        missing = sorted(models.loc[~known, "checkpoint"].unique())
        print(f"[skipped] no release date configured for {missing}")
        models = models.loc[known].copy()

    # One point per checkpoint/mode: its shortest evaluated effective context.
    models = models.loc[
        models.groupby(["checkpoint", "input_mode"])[
            "effective_context_points"
        ].idxmin()
    ]
    models["release_x"] = models["checkpoint"].map(release_index)
    models = models.dropna(subset=[RMSE_COLUMN]).sort_values("release_x")
    return models, baselines, tick_labels


def mode_label(row) -> str:
    if row.input_mode != "arx":
        return "AR"
    return "XReg" if row.model in {"timesfm10", "timesfm25"} else "ARX"


def label(row) -> str:
    name = row.checkpoint.rsplit("/", 1)[-1]
    if row.model.startswith("timesfm"):
        return f"TimesFM {row.model[-2]}.{row.model[-1]}-{mode_label(row)}"
    suffix = f"-{mode_label(row)}" if row.model == "chronos2" else ""
    return LABELS.get(name, name) + suffix


def marker(row):
    if row.input_mode != "arx":
        return "o"
    return "d" if mode_label(row) == "XReg" else "^"


def marker_size(row):
    return 42 + 0.72 * row.parameters_m


def colour(row) -> str:
    return COLORS.get(row.model, FALLBACK_COLOR)


def add_baselines(ax, baselines, x_label):
    for row in baselines.itertuples(index=False):
        name = "Daily" if row.model.endswith("daily") else "Weekly"
        value = getattr(row, RMSE_COLUMN)
        color = "#555555" if name == "Daily" else "#929292"
        ax.axhline(value, color=color, linestyle=(0, (5, 3)), lw=1.25)
        ax.text(
            x_label,
            value,
            f"{name} persistence  {value:.2f}",
            ha="left",
            va="bottom",
            fontsize=8.5,
            color=color,
            bbox={"facecolor": "#FCFCFB", "edgecolor": "none", "alpha": 0.88,
                  "pad": 1.2},
        )


def finish_axes(ax, xlabel):
    from matplotlib.lines import Line2D

    ax.set_xlabel(xlabel, fontsize=10.5, labelpad=11)
    ax.set_ylabel("Pooled RMSE (kWh)  ↓", fontsize=10.5)
    ax.grid(axis="y", color="#DADADA", linewidth=0.7, alpha=0.7)
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9.5)
    family_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=COLORS[key],
               label=name, markersize=6)
        for key, name in [
            ("chronos", "Chronos-T5"),
            ("timesfm10", "TimesFM 1.0"),
            ("chronos-bolt", "Chronos-Bolt"),
            ("timesfm25", "TimesFM 2.5"),
            ("chronos2", "Chronos-2"),
            ("timesfm30", "TimesFM 3.0"),
        ]
    ]
    family_legend = ax.legend(
        handles=family_handles, frameon=True,
        loc="upper right", bbox_to_anchor=(0.99, 0.82), ncol=1,
        handletextpad=0.45, columnspacing=1.0, fontsize=8.3,
        borderpad=0.55, labelspacing=0.42,
        facecolor="#FCFCFB", edgecolor="#B8B8B8", framealpha=0.96,
    )
    ax.add_artist(family_legend)

    mode_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color="#555555", label="AR"),
        Line2D([0], [0], marker="^", linestyle="none", color="#555555",
               label="Native ARX"),
        Line2D([0], [0], marker="d", linestyle="none", color="#555555",
               label="XReg"),
    ]
    ax.legend(handles=mode_handles, frameon=False, loc="lower left", ncol=3,
              handletextpad=0.35, columnspacing=0.8, fontsize=8.2)


def _y_limits(models: pd.DataFrame, baselines: pd.DataFrame) -> tuple[float, float]:
    values = pd.concat([models[RMSE_COLUMN], baselines[RMSE_COLUMN]]) \
        .astype(float).dropna()
    lower = min(20.3, float(values.min()) - 0.6)
    upper = max(33.35, float(values.max()) + 0.6)
    return lower, upper


def plot_release(models, baselines, tick_labels, out_dir: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.6, 5.9))
    ax.set_facecolor("#FCFCFB")

    # Horizontal displacement reveals checkpoints sharing one release event.
    release_offsets = {
        "chronos-t5-tiny": 0.10,
        "chronos-t5-mini": -0.10,
        "chronos-t5-small": 0.03,
        "chronos-t5-base": -0.15,
        "chronos-t5-large": 0.16,
        "timesfm-1.0-200m-pytorch:ar": -0.04,
        "timesfm-1.0-200m-pytorch:arx": 0.04,
        "chronos-bolt-tiny": -0.17,
        "chronos-bolt-mini": -0.06,
        "chronos-bolt-small": 0.06,
        "chronos-bolt-base": 0.17,
        "timesfm-2.5-200m-pytorch:ar": -0.04,
        "timesfm-2.5-200m-pytorch:arx": 0.04,
        "timesfm-3.0-pytorch:ar": -0.04,
        "timesfm-3.0-pytorch:arx": 0.04,
        "chronos-2:ar": -0.04,
        "chronos-2:arx": 0.04,
        "chronos-2-small:ar": -0.18,
        "chronos-2-small:arx": -0.06,
        "chronos-2-synth:ar": 0.06,
        "chronos-2-synth:arx": 0.18,
    }
    text_offsets = {
        "chronos-t5-base": (5, -14), "chronos-t5-large": (5, 13),
        "chronos-bolt-tiny": (-15, 19), "chronos-bolt-mini": (-20, -14),
        "chronos-bolt-small": (4, 13), "chronos-bolt-base": (5, -15),
        "chronos-2:ar": (-26, 18), "chronos-2:arx": (-70, -23),
        "chronos-2-small:ar": (-50, 38),
        "chronos-2-small:arx": (-8, -21),
        "chronos-2-synth:ar": (-8, 28),
        "chronos-2-synth:arx": (-8, 12),
    }

    for row in models.itertuples(index=False):
        name = row.checkpoint.rsplit("/", 1)[-1]
        key = (
            f"{name}:{row.input_mode}"
            if row.model in {"timesfm10", "timesfm25", "timesfm30", "chronos2"}
            else name
        )
        x = row.release_x + release_offsets.get(key, 0.0)
        value = getattr(row, RMSE_COLUMN)
        ax.scatter(x, value, s=marker_size(row), color=colour(row),
                   marker=marker(row), edgecolor="white", linewidth=0.9,
                   alpha=0.9, zorder=3)
        dx, dy = text_offsets.get(key, (6, 0))
        is_small = key.startswith("chronos-2-small:")
        is_synth = key.startswith("chronos-2-synth:")
        folded_variant = "small" if is_small else "synth"
        point_label = (
            f"Chronos-2-{folded_variant}\n{mode_label(row)}"
            if is_small or is_synth else label(row)
        )
        ax.annotate(
            point_label, (x, value), xytext=(dx, dy),
            textcoords="offset points", va="center",
            ha="right" if is_small or is_synth else "left", fontsize=8.5,
            color=colour(row), fontweight="semibold",
            arrowprops={"arrowstyle": "-", "color": colour(row), "lw": 0.7}
            if key in text_offsets else None,
        )

    add_baselines(ax, baselines, -0.28)
    ax.set_xlim(-0.36, max(0, len(tick_labels) - 1) + 0.60)
    ax.set_ylim(*_y_limits(models, baselines))
    ax.set_xticks(range(len(tick_labels)), tick_labels)
    ax.set_yticks([21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33])
    finish_axes(ax, "Public model release date  →")
    ax.text(1, 1.01, "Marker area ∝ parameters", transform=ax.transAxes,
            ha="right", fontsize=9, color="#555555")
    ax.text(1, -0.13,
            "Release events are evenly spaced; points are displaced within each event.",
            transform=ax.transAxes, ha="right", fontsize=8.5, color="#666666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_dir / "release_date_rmse.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "release_date_rmse.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_energy(models, baselines, out_dir: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.6, 5.9))
    ax.set_facecolor("#FCFCFB")
    label_offsets = {
        "chronos-bolt-tiny": (7, 16), "chronos-bolt-mini": (7, -17),
        "chronos-bolt-small": (7, 11), "chronos-bolt-base": (-15, 25),
        "chronos-2:ar": (-12, -28), "chronos-2:arx": (9, -24),
        "chronos-2-small:ar": (-60, 24),
        "chronos-2-small:arx": (-35, 23),
        "chronos-2-synth:ar": (-105, 24),
        "chronos-2-synth:arx": (10, 24),
        "timesfm-2.5-200m-pytorch:ar": (10, -22),
        "timesfm-2.5-200m-pytorch:arx": (10, 28),
        "timesfm-1.0-200m-pytorch:ar": (10, -28),
        "timesfm-1.0-200m-pytorch:arx": (7, 12),
        "chronos-t5-base": (-53, -15), "chronos-t5-large": (-52, 13),
    }

    energy = models.dropna(subset=[ENERGY_COLUMN])
    for row in energy.itertuples(index=False):
        name = row.checkpoint.rsplit("/", 1)[-1]
        key = (
            f"{name}:{row.input_mode}"
            if row.model in {"timesfm10", "timesfm25", "timesfm30", "chronos2"}
            else name
        )
        x = getattr(row, ENERGY_COLUMN)
        value = getattr(row, RMSE_COLUMN)
        ax.scatter(x, value, s=marker_size(row), color=colour(row),
                   marker=marker(row), edgecolor="white", linewidth=0.9,
                   alpha=0.9, zorder=3)
        dx, dy = label_offsets.get(key, (7, 0))
        ax.annotate(
            label(row), (x, value), xytext=(dx, dy),
            textcoords="offset points", va="center", fontsize=8.2,
            color=colour(row), fontweight="semibold",
            arrowprops={"arrowstyle": "-", "color": colour(row), "lw": 0.7}
            if key in label_offsets else None,
        )

    add_baselines(ax, baselines, 0.32)
    ax.set_xscale("log")
    ax.set_xlim(0.28, 700)
    ax.set_xticks(
        [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
        ["0.5", "1", "2", "5", "10", "20", "50", "100", "200", "500"],
    )
    ax.set_ylim(*_y_limits(models, baselines))
    ax.set_yticks([21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33])
    finish_axes(ax, "GPU energy for the full evaluation run (Wh, log scale)  →")
    ax.text(1, 1.01, "Marker area ∝ parameters", transform=ax.transAxes,
            ha="right", fontsize=9, color="#555555")
    fig.tight_layout()
    fig.savefig(out_dir / "energy_rmse.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "energy_rmse.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    models, baselines, tick_labels = load_data(
        args.summary, args.release_dates, args.site, args.protocol
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_release(models, baselines, tick_labels, args.out_dir)
    plot_energy(models, baselines, args.out_dir)
    print(f"Wrote release_date_rmse and energy_rmse figures to {args.out_dir}")


if __name__ == "__main__":
    main()
