[![DOI](https://zenodo.org/badge/1059429283.svg)](https://doi.org/10.5281/zenodo.21979000)

This repository contains the code for the paper:

> **When and How Should a Power Trader Engage in Arbitrage? Predict, then Contextually Optimize**
> Yannick Heiser, Jalal Kazempour, Farzaneh Pourahmadi
> *Department of Wind and Energy Systems, Technical University of Denmark (DTU)*

The paper can be found on [Arxiv](https://arxiv.org/abs/2607.07351).

## Tech stack

- **[Hydra](https://hydra.cc/)** — composable YAML configuration (datasets × models × experiments)
- **[uv](https://docs.astral.sh/uv/)** — dependency management and reproducible environments
- **LightGBM / XGBoost / scikit-learn / PyTorch** — probabilistic classifiers
- **[linopy](https://linopy.readthedocs.io/) + [HiGHS](https://highs.dev/)** — linear-policy optimization
- **SHAP** — feature selection

## Project structure

```txt
day_ahead_v2/
├── configs/                       # Hydra configuration
│   ├── config_dev.yaml            #   Default config used by train & benchmark
│   ├── config_prod.yaml
│   ├── config_test.yaml
│   ├── datasets/                  #   Data sources + feature/target definitions
│   │   ├── data2025_DK_1.yaml     #     DK1 bidding zone
│   │   ├── data2025_DE_LU.yaml    #     DE/LU bidding zone
│   │   └── ...                    #     raw-source configs (entsoe, energinet, openmeteo, ...)
│   ├── models/                    #   Classifier configs
│   │   ├── lightgbm.yaml
│   │   ├── xgboost.yaml
│   │   ├── logistic_regression.yaml
│   │   ├── mlp.yaml
│   │   └── gaussian_process_classification.yaml
│   └── experiments/               #   Rolling windows, thresholds, portfolio, CVaR weight, ...
│       ├── wasserstein_DK_1_hpp.yaml
│       ├── benchmark_exp1_hpp.yaml
│       └── ...
├── data/
│   ├── raw/                       # Raw market & weather downloads (ENTSO-E, Energinet, Open-Meteo, ...)
│   └── processed/                 # Model-ready parquet/csv datasets
├── src/day_ahead_v2/
│   ├── train.py                   # Predict-then-contextual-optimize framework  (main model)
│   ├── benchmark.py               # Benchmark bidding strategies
│   ├── optimization.py            # linopy optimization models (class policies, hindsight, ...)
│   ├── model.py                   # Classifier wrappers (LightGBM, XGBoost, MLP, ...)
│   ├── data.py                    # Data loading, preprocessing & feature engineering
│   ├── evaluate.py                # Metrics, bid & profit calculation
│   ├── wasserstein.py             # Distributional-drift (Wasserstein) analysis
│   ├── descriptive_analysis.py    # Market/price descriptive statistics & figures
│   ├── graph.py                   # Plotting utilities
│   ├── api.py                     # FastAPI serving endpoint
│   └── utils/                     # Electrolyzer efficiency, PCA, plotting, helpers
├── reports/                       # Backtest outputs (metrics, hourly results, figures)
├── outputs/ · multirun/           # Hydra run directories
├── paper/                         # LaTeX source of the accompanying paper
├── docs/                          # mkdocs documentation
├── tests/                         # pytest test suite
├── pyproject.toml                 # Project metadata & dependencies (uv)
├── uv.lock
└── tasks.py                       # invoke task shortcuts
```

## Installation

The project targets **Python 3.12** and uses [uv](https://docs.astral.sh/uv/) for
reproducible environments. From the project root:

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the environment and install all (locked) dependencies
uv sync
```

This creates a `.venv/` from `uv.lock`. Prefix commands with `uv run`, or activate the
environment with `source .venv/bin/activate`.

## Configuration

Runs are configured with Hydra by composing three config groups plus an experiment
definition. The default entry-point config is [configs/config_dev.yaml](configs/config_dev.yaml),
which selects a `datasets`, a `models`, and an `experiments` group. Any of these can be
overridden on the command line:

| Group | Location | Examples |
|-------|----------|----------|
| `datasets` | [configs/datasets/](configs/datasets/) | `data2025_DK_1`, `data2025_DE_LU` |
| `models` | [configs/models/](configs/models/) | `lightgbm`, `xgboost`, `mlp`, `logistic_regression` |
| `experiments` | [configs/experiments/](configs/experiments/) | `wasserstein_DK_1_hpp`, `benchmark_exp1_hpp` |

Each experiment config sets the rolling-window lengths, confidence thresholds, the portfolio (`wind_only` / `hpp`), the CVaR weight, and the optimization
model. See [configs/experiments/alpha_tuning_exp1.yaml](configs/experiments/alpha_tuning_exp1.yaml)
for a fully commented example.

## Running the models

The full predict-then-contextual-optimize framework runs a rolling-window backtest:

```bash
# Run with the default config (configs/config_dev.yaml)
uv run python -m day_ahead_v2.train

# Run with a different config file
uv run python -m day_ahead_v2.train --config-name=config_prod

# Select a bidding zone, classifier and experiment explicitly
uv run python -m day_ahead_v2.train \
    datasets=data2025_DK_1 \
    models=lightgbm \
    experiments=alpha_tuning_exp1
```

For each rolling window this: trains and tunes the classifier, tunes the confidence
thresholds and per-class linear policies via contextual optimization, then evaluates
out-of-sample profit. Set `experiments.experiment_parameters.mode=classif_only` to run
the classification stage without the optimization layer.

Results (aggregate metrics, per-window results, and hourly bids/profit) are written to
`reports/<experiment_name>/<model_name>/<dataset_name>/`.

### Hyperparameter sweeps

Hydra multirun sweeps across any override, e.g. across CVaR weights and thresholds:

```bash
uv run python -m day_ahead_v2.train --multirun \
    experiments=beta_0.1_alpha0_0.15_alpha1_0.55,beta_0.5_alpha0_0.25_alpha1_0.65
```

## Running the benchmarks

The benchmark script evaluates several reference bidding strategies over the same rolling
windows for comparison:

```bash
uv run python -m day_ahead_v2.benchmark \
    datasets=data2025_DK_1 \
    experiments=benchmark_exp1_hpp
```

It runs four strategies:

| Strategy | Description |
|----------|-------------|
| **Hindsight** | Perfect-foresight upper bound (uses realized prices/production) |
| **Single linear policy** | One linear decision policy for all samples (no classifier gating) |
| **Bid forecast** | Arbitrage-free: day-ahead bid equals the production forecast |
| **Always bid max** | Always bids full wind capacity |

Each strategy's metrics and hourly results are written to
`reports/<experiment_name>/<strategy>/<dataset_name>/`.

## Additional entry points

```bash
# Distributional-drift (Wasserstein) analysis between training and test windows
uv run python -m day_ahead_v2.wasserstein

# Descriptive market/price statistics and figures
uv run python -m day_ahead_v2.descriptive_analysis
```

## Testing

```bash
uv run pytest tests/
# or, with coverage (see tasks.py):
uv run invoke test
```

## Citation

If you use this code, please cite the accompanying paper:

```bibtex
@article{heiser_predict_then_optimize,
  title   = {When and How Should a Power Trader Engage in Arbitrage?
             Predict, then Contextually Optimize},
  author  = {Heiser, Yannick and Kazempour, Jalal and Pourahmadi, Farzaneh},
  institution = {Technical University of Denmark},
  note    = {Working paper}
}
```

## Acknowledgements

This research was supported by EUDP (Grant number: 640222-496237) and Innovation Fund
Denmark (Grant number: 150-00001B).
