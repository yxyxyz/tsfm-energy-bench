from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


class NvidiaSmiPowerSampler:
    """Capture a raw NVML-backed GPU power trace with one nvidia-smi process."""

    def __init__(self, output_path: Path, *, interval_ms: int, gpu_index: int = 0):
        self.output_path = output_path
        self.interval_ms = interval_ms
        self.gpu_index = gpu_index
        self.samples: list[tuple[float, float, float]] = []
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._started_monotonic: float | None = None
        self._stopped_monotonic: float | None = None

    def validate(self) -> float:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={self.gpu_index}",
                "--query-gpu=power.draw.instant",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        power_w = float(result.stdout.strip().splitlines()[0])
        if not math.isfinite(power_w) or power_w <= 0:
            raise RuntimeError(f"nvidia-smi returned invalid GPU power: {power_w}")
        return power_w

    def start(self) -> None:
        command = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=power.draw.instant",
            "--format=csv,noheader,nounits",
            "-lms",
            str(self.interval_ms),
        ]
        self._started_monotonic = time.perf_counter()
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._thread.start()

    def _read_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            value = line.strip().split()[0] if line.strip() else ""
            try:
                power_w = float(value)
            except ValueError:
                continue
            self.samples.append(
                (
                    time.perf_counter(),
                    time.time(),
                    power_w,
                )
            )

    def _terminate_process(self) -> None:
        if self._process is None or self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=3)

    def stop(self) -> dict[str, float | int]:
        self._stopped_monotonic = time.perf_counter()
        self._terminate_process()
        if self._thread is not None:
            self._thread.join(timeout=3)
        # Buffered subprocess output may arrive while the sampler is stopping.
        self.samples = [
            sample for sample in self.samples
            if self._started_monotonic <= sample[0] <= self._stopped_monotonic
        ]
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["elapsed_seconds", "timestamp_unix", "power_w"])
            start = self._started_monotonic or 0.0
            for monotonic, timestamp, power_w in self.samples:
                writer.writerow([f"{monotonic - start:.6f}", timestamp, power_w])
        return self.summary()

    def summary(self) -> dict[str, float | int]:
        if not self.samples:
            raise RuntimeError("nvidia-smi power trace contains no samples")
        start = self._started_monotonic or self.samples[0][0]
        stop = self._stopped_monotonic or self.samples[-1][0]
        powers = [sample[2] for sample in self.samples]
        energy_kwh = _integrate_power(self.samples, start, stop)
        samples_1s = _decimate_power(self.samples, interval_seconds=1.0)
        mean_power = statistics.fmean(powers)
        return {
            "nvidia_smi_energy_kwh": energy_kwh,
            "nvidia_smi_energy_kwh_native": energy_kwh,
            "nvidia_smi_energy_kwh_1s_decimated": _integrate_power(
                samples_1s, start, stop
            ),
            "nvidia_smi_native_interval_ms": self.interval_ms,
            "nvidia_smi_samples": len(powers),
            "nvidia_smi_samples_1s_decimated": len(samples_1s),
            "nvidia_smi_mean_power_w": mean_power,
            "nvidia_smi_peak_power_w": max(powers),
            "nvidia_smi_power_cv": (
                statistics.pstdev(powers) / mean_power if mean_power else 0.0
            ),
        }


class EnergyMeasurement:
    """Independent GPU trace with auxiliary CodeCarbon/CarbonTracker estimates."""

    def __init__(
        self,
        output_dir: Path,
        *,
        interval_ms: int,
        gpu_index: int,
        metadata: dict[str, Any],
    ):
        self.output_dir = output_dir
        self.energy_dir = output_dir / "energy"
        self.summary_path = output_dir / "energy_summary.json"
        self.metadata = metadata
        self._started_monotonic: float | None = None
        self._codecarbon: Any = None
        self._carbontracker: Any = None
        self._sampler = NvidiaSmiPowerSampler(
            self.energy_dir / "nvidia_smi_power.csv",
            interval_ms=interval_ms,
            gpu_index=gpu_index,
        )

    def prepare(self) -> None:
        if self.summary_path.exists() or self.energy_dir.exists():
            raise ValueError(
                f"Energy output already exists in {self.output_dir}. "
                "Use a fresh checkpoint/mode output directory for a formal run."
            )
        self._sampler.validate()
        self.energy_dir.mkdir(parents=True)
        from codecarbon import EmissionsTracker
        from carbontracker.tracker import CarbonTracker
        self._codecarbon = EmissionsTracker(
            project_name="bdg2-tsfm-inference",
            output_dir=str(self.energy_dir),
            output_file="codecarbon.csv",
            measure_power_secs=5,
            save_to_file=True,
            log_level="error",
        )
        self._carbontracker = CarbonTracker(
            epochs=1,
            update_interval=1,
            ignore_errors=True,
            log_dir=str(self.energy_dir / "carbontracker"),
            log_file_prefix="bdg2_tsfm_",
            verbose=0,
        )

    def start(self) -> None:
        self._codecarbon.start()
        self._carbontracker.epoch_start()
        self._sampler.start()
        self._started_monotonic = time.perf_counter()
        self._sampler._started_monotonic = self._started_monotonic

    def stop(self, *, status: str) -> Path:
        duration = (
            time.perf_counter() - self._started_monotonic
            if self._started_monotonic is not None else 0.0
        )
        errors: list[str] = []
        # Stop the primary GPU trace before auxiliary tracker shutdown and I/O.
        try:
            nv_summary = self._sampler.stop()
        except Exception as exc:
            nv_summary = {}
            errors.append(f"nvidia-smi stop failed: {exc}")
        try:
            self._codecarbon.stop()
        except Exception as exc:  # preserve the other two raw measurements
            errors.append(f"CodeCarbon stop failed: {exc}")
        try:
            self._carbontracker.epoch_end()
            self._carbontracker.stop()
        except Exception as exc:
            errors.append(f"CarbonTracker stop failed: {exc}")

        summary: dict[str, Any] = {
            **self.metadata,
            **nv_summary,
            "status": status if nv_summary else "failed",
            "measurement_protocol": "forecast_loop_v2",
            "measurement_scope_includes": [
                "walk_forward_forecasting",
                "model_input_output_processing",
                "covariate_or_xreg_processing",
                "prediction_collection",
            ],
            "measurement_scope_excludes": [
                "dataset_loading",
                "model_loading",
                "warmup",
                "metric_calculation",
                "result_writes",
            ],
            "measurement_seconds": duration,
            "codecarbon_interval_seconds": 5,
            "carbontracker_interval_seconds": 1,
            "nvidia_smi_interval_ms": self._sampler.interval_ms,
        }
        try:
            cc = _read_codecarbon_energy(self.energy_dir / "codecarbon.csv")
            summary["codecarbon_energy_kwh"] = cc
            summary["codecarbon_energy_kwh_5s"] = cc
        except Exception as exc:
            errors.append(f"CodeCarbon result parse failed: {exc}")
        try:
            ct = _read_carbontracker_energy(self.energy_dir / "carbontracker")
            summary["carbontracker_energy_kwh"] = ct
            summary["carbontracker_energy_kwh_1s"] = ct
        except Exception as exc:
            errors.append(f"CarbonTracker result parse failed: {exc}")
        if "codecarbon_energy_kwh" in summary and "carbontracker_energy_kwh" in summary:
            cc = float(summary["codecarbon_energy_kwh"])
            ct = float(summary["carbontracker_energy_kwh"])
            mean = (cc + ct) / 2
            meter_windows = (
                int(summary["meters"])
                * int(summary["num_windows"])
                * len(summary["history_months"])
            )
            forecast_points = meter_windows * int(summary["horizon"])
            summary.update(
                {
                    "system_energy_kwh_mean_cc_ct": mean,
                    "meter_windows": meter_windows,
                    "forecast_points": forecast_points,
                    "system_energy_mwh_per_meter_window": (
                        mean * 1_000_000 / meter_windows
                    ),
                    "system_energy_wh_per_1000_forecast_points": (
                        mean * 1_000_000 / forecast_points
                    ),
                }
            )
            if min(cc, ct) > 0:
                summary["cc_ct_ratio"] = max(cc, ct) / min(cc, ct)
        summary["measurement_warnings"] = errors
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return self.summary_path


def _read_codecarbon_energy(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not rows[-1].get("energy_consumed"):
        raise ValueError(f"No energy_consumed value in {path}")
    return float(rows[-1]["energy_consumed"])


def _integrate_power(
    samples: list[tuple[float, float, float]], start: float, stop: float
) -> float:
    energy_j = samples[0][2] * max(0.0, samples[0][0] - start)
    for left, right in zip(samples, samples[1:]):
        energy_j += 0.5 * (left[2] + right[2]) * (right[0] - left[0])
    energy_j += samples[-1][2] * max(0.0, stop - samples[-1][0])
    return energy_j / 3_600_000


def _decimate_power(
    samples: list[tuple[float, float, float]], *, interval_seconds: float
) -> list[tuple[float, float, float]]:
    """Thin the native trace to approximately 1 Hz for sensitivity analysis."""
    selected = [samples[0]]
    for sample in samples[1:]:
        if sample[0] - selected[-1][0] >= interval_seconds:
            selected.append(sample)
    return selected


def _read_carbontracker_energy(log_dir: Path) -> float:
    from carbontracker import parser

    energy_kwh, _co2_g, _equivalents = parser.aggregate_consumption(str(log_dir))
    if energy_kwh is None or not math.isfinite(float(energy_kwh)):
        raise ValueError(f"No CarbonTracker energy in {log_dir}")
    return float(energy_kwh)


def gpu_index_from_device(device: str) -> int:
    if ":" not in device:
        return 0
    return int(device.split(":", maxsplit=1)[1])
