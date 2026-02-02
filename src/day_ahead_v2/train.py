#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File name: predict_IM.py
Author: Yannick Heiser
Created: 2025-09-25
Version: 1.0
Description:
    Model to predict the system imbalance (IM).

Contact: yahei@dtu.dk
Dependencies:
    - pandas
    - numpy
    - matplotlib
    - scikit-learn
    - tabulate
    - src.data_handler (custom module)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import logging
import random
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from datetime import timedelta
import time
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from joblib import Parallel, delayed
from day_ahead_v2.data import DataHandler
from day_ahead_v2.evaluate import evaluate_classifier, compute_accuracy_f1
from day_ahead_v2.utils.sanitize_names import sanitize_column_names



logger = logging.getLogger(__name__)


def rolling_windows(cfg: DictConfig):
    """
    Generate rolling time windows for training, validation, and testing.

    Args:
        cfg (DictConfig): Configuration object containing experiment and training parameters.
    """
    train_days = cfg.experiments.train_parameters.train_length
    valid_days = cfg.experiments.train_parameters.valid_length
    test_days  = cfg.experiments.train_parameters.test_length

    start_date = pd.Timestamp(cfg.experiments.experiment_parameters.start_date)
    end_date   = pd.Timestamp(cfg.experiments.experiment_parameters.end_date)
    step_days  = test_days

    if start_date.tzinfo is None:
        start_date = start_date.tz_localize("UTC")
    else:
        start_date = start_date.tz_convert("UTC")

    if end_date.tzinfo is None:
        end_date = end_date.tz_localize("UTC")
    else:
        end_date = end_date.tz_convert("UTC")

    t = start_date
    window_count = 0

    while True:
        train_start = t
        train_end   = t + timedelta(days=train_days)

        valid_start = train_end
        valid_end   = valid_start + timedelta(days=valid_days)

        test_start  = valid_end
        test_end    = test_start + timedelta(days=test_days)

        # stop when test window would exceed backtest horizon
        if test_end > end_date:
            break

        window_count += 1
        logger.info(f"Generated rolling window: {window_count} ")
        yield {
            "train": (train_start, train_end),
            "valid": (valid_start, valid_end),
            "test":  (test_start, test_end),
        }

        t += timedelta(days=step_days)

        # Raise error if no windows were generated
    if window_count == 0:
        raise ValueError(
            f"No rolling windows could be generated with the current configuration:\n"
            f"start_date={start_date.date()}, end_date={end_date.date()}, "
            f"train_days={train_days}, valid_days={valid_days}, test_days={test_days}, "
            f"step_days={step_days}"
        )

def split_features_target(cfg: DictConfig, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split data into features and target based on config.

    Args:
        cfg (DictConfig): Configuration object containing dataset parameters.
        data (pd.DataFrame): DataFrame with loaded data.

    Returns:
        X (pd.DataFrame): Features DataFrame.
        y (pd.Series): Target variable.
    """
    datetime_column = cfg.datasets.training.get("datetime_column", None)
    target_column = cfg.datasets.training.target_column
    feature_columns = list(cfg.datasets.training.feature_columns)

    missing = set(feature_columns + [target_column]) - set(data.columns)
    if missing:
        logger.warning(f"Missing columns in dataframe: {missing}")
        feature_columns = [col for col in feature_columns if col in data.columns]

    X = data[feature_columns]
    y = data[target_column]

    if datetime_column is not None:
        X.set_index(data[datetime_column], inplace=True)
        y.index = data[datetime_column]

    # Sanitize column names
    X = sanitize_column_names(X)

    return X, y

def get_hyperparameter_combinations(cfg: DictConfig) -> list[dict]:
    """
    Generate all hyperparameter combinations from config file.

    Args:
        cfg (DictConfig): Configuration object containing model hyperparameters.

    Returns:
        List[Dict]: List of dictionaries, each representing a unique combination of hyperparameters.
    """
    hyperparameters = OmegaConf.to_container(cfg.models.model_hyperparameters, resolve=True)

    # Ensure all values are lists
    param_grid = {}
    for k, v in hyperparameters.items():
        if not isinstance(v, (list, tuple)):
            logger.warning(f"Hyperparameter '{k}' is not a list. Converting to list automatically.")
            param_grid[k] = [v]
        else:
            param_grid[k] = v

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    combos = list(product(*values))
    return [dict(zip(keys, combo)) for combo in combos]

def train_batch(cfg: DictConfig, X_train: pd.DataFrame, y_train: pd.Series, hyperparameters: dict) -> object:
    """Train model for a single rolling window and hyperparameter set."""
    # Check if types are correct
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(f"X_train must be a pd.DataFrame, got {type(X_train)}")
    if not isinstance(y_train, pd.Series):
        raise TypeError(f"y_train must be a pd.Series, got {type(y_train)}")

    # Instantiate model for given window and hyperparameters
    base_params = OmegaConf.to_container(cfg.models.model_parameters, resolve=True)

    model = instantiate({"_target_": cfg.models._target_, **base_params, **hyperparameters})

    # Fit model
    model.fit(X_train, y_train)
    logger.info(f"Model trained with hyperparameters: {hyperparameters}")

    return model

def test_batch(cfg: DictConfig, model: object, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """Test model for a single rolling window and hyperparameter set."""
    # Predict and evaluate on validation set
    val_metrics, _ = evaluate_classifier(model, X_val, y_val)
    logger.info(f"Validation metrics: {val_metrics}")

    return val_metrics

def train_and_validate_params(
        cfg: DictConfig,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict,
    ) -> tuple[dict, float]:
    model = train_batch(cfg, X_train, y_train, params)
    metrics = test_batch(cfg, model, X_val, y_val)
    score = metrics.get("f1_score", np.nan)
    return params, score


def train_model(cfg: DictConfig, window: dict, data_handler: DataHandler) -> None:
    """Train model for a single rolling window."""
    logger.info("Training model for window...")

    # ---------------------------------------------
    # Rolling window
    # ---------------------------------------------
    train_start, train_end = window["train"]
    valid_start, valid_end = window["valid"]
    test_start, test_end   = window["test"]

    logger.info(
        f"Train: {train_start.date()} → {train_end.date()} | "
        f"Valid: {valid_start.date()} → {valid_end.date()} | "
        f"Test: {test_start.date()} → {test_end.date()}"
    )

    # ---------------------------------------------
    # Data loading
    # ---------------------------------------------
    logger.debug("Cutting train data...")
    data_train = data_handler.cut_data(train_start, train_end, cfg.datasets.training.datetime_column)
    X_train, y_train = split_features_target(cfg, data_train.data)

    logger.debug("Cutting validation data...")
    data_valid = data_handler.cut_data(valid_start, valid_end, cfg.datasets.training.datetime_column)
    X_val, y_val = split_features_target(cfg, data_valid.data)

    logger.debug("Cutting test data...")
    data_test = data_handler.cut_data(test_start, test_end, cfg.datasets.training.datetime_column)
    X_test, y_test = split_features_target(cfg, data_test.data)

    # ---------------------------------------------
    # Hyperparameter tuning (on validation)
    # ---------------------------------------------
    hyperparameter_combinations = get_hyperparameter_combinations(cfg)

    n_jobs = cfg.experiments.train_parameters.get("n_jobs", -1)

    start = time.perf_counter()

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",   # multiprocessing
        verbose=5,
    )(
        delayed(train_and_validate_params)(
            cfg,
            X_train,
            y_train,
            X_val,
            y_val,
            params,
        )
        for params in hyperparameter_combinations
    )

    elapsed = time.perf_counter() - start
    logger.info(
        f"Window {train_start.date()} → {test_end.date()} | "
        f"Hyperparameter search time: {elapsed:.2f}s | "
        f"n_jobs={n_jobs}"
    )

    best_score = -np.inf
    best_params = None

    for params, score in results:
        logger.info(f"Params {params} → F1={score}")

        if np.isnan(score):
            logger.warning(f"F1 score is NaN for params {params}")
            continue

        if score > best_score:
            best_score = score
            best_params = params

    if best_params is None:
        raise RuntimeError("No valid hyperparameter combination produced a score.")

    logger.info(f"Best hyperparameters: {best_params} (F1={best_score:.4f})")

    # ---------------------------------------------
    # Retrain on train + validation
    # ---------------------------------------------
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    final_model = train_batch(cfg, X_train_full, y_train_full, best_params)

    # ---------------------------------------------
    # Final test evaluation
    # ---------------------------------------------
    train_metrics, _ = evaluate_classifier(final_model, X_train_full, y_train_full)
    logger.info(f"Train metrics: {train_metrics}")
    test_metrics, test_results_df = evaluate_classifier(final_model, X_test, y_test)
    logger.info(f"Test metrics: {test_metrics}")

    # ---------------------------------------------
    # Collect results
    # ---------------------------------------------
    results = {
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "train_start": train_start,
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "test_start": test_start,
        "test_end": test_end,
        **best_params,
    }


    return final_model, results, test_results_df

def run_backtest(cfg: DictConfig) -> list:
    """Run backtest over all rolling windows."""
    logger.info("Starting backtest...")

    # ---------------------------------------------
    # Data import and preprocessing
    # ---------------------------------------------
    data_handler = DataHandler(cfg)
    data_handler = data_handler.cut_data(cfg.experiments.experiment_parameters.start_date,
                                        cfg.experiments.experiment_parameters.end_date,
                                        cfg.datasets.training.datetime_column,
                                        )
    data_handler = data_handler.transform_data(cfg)

    # Check for NaNs in data
    nan_counts = data_handler.data.isnull().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"Found NaN values in data:\n{nan_counts[nan_counts > 0]}")
    else:
        logger.info("No NaN values found in data")

    # ---------------------------------------------
    # Rolling window backtest
    # ---------------------------------------------
    all_results = []
    all_test_results_dfs = pd.DataFrame()
    for window in rolling_windows(cfg):
        try:
            _, results, test_results_df = train_model(cfg, window, data_handler)
            all_results.append(results)
            all_test_results_dfs = pd.concat([all_test_results_dfs, test_results_df])
        except Exception as e:
            logger.error(f"Error in window {window}: {e}")

    # Compute average metrics over all windows
    if not all_test_results_dfs.empty:
        avg_metrics = compute_accuracy_f1(
            all_test_results_dfs["true_label"].to_numpy(),
            all_test_results_dfs["predicted_label"].to_numpy()
        )
        for key, value in avg_metrics.items():
            logger.info(f"Average {key} over all windows: {value}")
    else:
        avg_metrics = {}
        logger.warning("No test results to compute average metrics.")

    logger.info("Backtest completed.")
    return all_results, avg_metrics

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config_dev")
def main(cfg: DictConfig) -> None:
    logger.info("Starting experiment")
    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)

    results, metrics = run_backtest(cfg)

    if not results:
        logger.warning("No results generated")
        return

    # Save results to CSV
    OmegaConf.resolve(cfg)
    save_path = Path(cfg.results.save_path)
    if not save_path.is_absolute():
        save_path = Path(__file__).resolve().parent.parent.parent / save_path
    save_path.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path / "backtest_results.csv", index=False)
    logger.info(f"Results saved to {save_path / 'backtest_results.csv'}")

    # Save avg_accuracy and avg_f1 to a text file
    with open(save_path / "metrics.txt", "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    logger.info("Experiment finished successfully")

if __name__ == "__main__":
    main()
