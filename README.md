# TSFM Energy Benchmark

Compare forecasting accuracy, runtime and GPU energy on [BDG2](https://github.com/buds-lab/building-data-genome-project-2).

## Structure

```text
tsfm-energy-bench/
├── run_exp_grid.py          # Run experiments
├── configs/
│   └── models.json         # Models and grid membership
├── requirements/           # Dependencies for each environment
├── scripts/
│   ├── setup/              # Environment setup and data download
│   └── analysis/           # Summaries and figures
└── src/bdg2_tsfm/           # Data loading, models and evaluation
```

## Setup

Run from the repository root. With conda installed, create only the environments you need:

```bash
# Chronos models and TimesFM 2.5
bash scripts/setup/setup_env.sh cikm-tsfm-modern requirements/modern.txt
# TimesFM 1.0
bash scripts/setup/setup_env.sh cikm-timesfm10 requirements/timesfm10.txt
# TimesFM 3.0
bash scripts/setup/setup_env.sh cikm-timesfm30 requirements/timesfm30.txt
# MLP
bash scripts/setup/setup_env.sh cikm-baselines requirements/baselines.txt
```

Check CUDA and download the data (requires Git LFS):

```bash
conda run -n cikm-tsfm-modern python scripts/setup/check_cuda.py
bash scripts/setup/download_bdg2.sh data/bdg2
```

## Run

Defaults: Fox electricity meters, 24-hour forecasts, 365 daily forecast origins in 2017, batch size 32, and histories of 1/3/6/9/12 months (30 days per month).

```bash
# Base checkpoints, AR and ARX where supported
python run_exp_grid.py --data data/bdg2 --modes both

# All TSFM checkpoints
python run_exp_grid.py --data data/bdg2 --grid full --include-scope-review --modes both

# Select one family
python run_exp_grid.py --data data/bdg2 --families timesfm30 --modes both

# MLP
python run_exp_grid.py --data data/bdg2 --grid baselines --families mlp --modes both
```

The grid selects the conda environment automatically and skips completed runs. Results go to `results/`; use `--results-dir PATH` to change this. Add `--print-only` to preview commands or `--history-months 1` to run only one history. Use a separate results directory for a different dataset or forecasting protocol.

For a small check before the full grid:

```bash
python run_exp_grid.py --data data/bdg2 --families chronos2 \
  --history-months 1 --num-windows 1 --max-meters 2 \
  --energy-tool none --results-dir results/smoke
```

## Results

Each run writes:

| File | Contents |
|---|---|
| `summary_overall.csv` | RMSE, runtime and energy for the run |
| `summary_by_meter_type.csv` | Metrics by meter type |
| `per_meter_window.csv` | Errors for each meter and forecast window |
| `forecast_arrays.npz` | Actual and predicted values |
| `energy_summary.json` | Energy measurements and sampling settings |
| `energy/nvidia_smi_power.csv` | Raw GPU power trace |
| `run_manifest.json` | Run settings |
| `training_windows.csv` | MLP training time per window |

GPU energy is recorded in kWh. For MLP, training time is recorded separately; measured evaluation energy excludes training.

Combine results and plot:

```bash
conda activate cikm-tsfm-modern
python scripts/analysis/summarize_results.py --results-dir results --out results/combined_summary.csv
python -m pip install -e '.[plotting]'
python scripts/analysis/plot_release_rmse.py --summary results/summary_by_effective_context.csv --out-dir figures
```

`combined_summary.csv` contains one row per run. `summary_by_effective_context.csv` groups repeated effective contexts within the same protocol and is used for plotting.

## Acknowledgments and license

The data protocol follows the BuildSys Chronos-2 reference experiments; energy measurement follows the benchmark design shared by Aymeric Trigaux.

Code is MIT-licensed. BDG2 and model weights retain their own licenses; see the official [Chronos](https://github.com/amazon-science/chronos-forecasting) and [TimesFM](https://github.com/google-research/timesfm) repositories. [TimesFM 3.0 weights](https://huggingface.co/google/timesfm-3.0-pytorch) have a separate non-commercial, non-production license.
