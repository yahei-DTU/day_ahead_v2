#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File name: train.py
Author: Yannick Heiser
Created: 2025-09-25
Version: 1.0
Description:
    Training methods to predict imbalance state and optimize for optimal day-ahead bids.

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
import sys
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
from day_ahead_v2.optimization import ModelSurplus, ModelBalance, ModelDeficit
from day_ahead_v2.evaluate import evaluate_classifier, compute_accuracy_f1, threshold_predictions, calculate_bids, calculate_profit
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

    if datetime_column is not None and datetime_column in data.columns:
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
    logger.debug(f"Start training model for window {X_train.index.min()} to {X_train.index.max()} with hyperparameters: {hyperparameters}")
    # Check if types are correct
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(f"X_train must be a pd.DataFrame, got {type(X_train)}")
    if not isinstance(y_train, pd.Series):
        raise TypeError(f"y_train must be a pd.Series, got {type(y_train)}")

    # Instantiate model for given window and hyperparameters
    base_params = OmegaConf.to_container(cfg.models.model_parameters, resolve=True)

    if cfg.models._target_.endswith("MLPClassifier"):
        base_params["input_dim"] = X_train.shape[1]

    model = instantiate({"_target_": cfg.models._target_, **base_params, **hyperparameters})

    # Fit model
    model.fit(X_train, y_train)
    logger.info(f"Model trained with hyperparameters: {hyperparameters}")

    return model

def test_batch(cfg: DictConfig, model: object, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """Test model for a single rolling window and hyperparameter set."""
    # Predict and evaluate on validation set
    val_metrics, val_results_df = evaluate_classifier(model, X_val, y_val)
    logger.info(f"Validation metrics: {val_metrics}")

    return val_metrics, val_results_df

def train_and_validate_params(
        cfg: DictConfig,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict,
    ) -> tuple[dict, float]:
    logger.debug(f"Training and validating with params: {params}")
    model = train_batch(cfg, X_train, y_train, params)
    _, train_results_df = test_batch(cfg,model,X_train,y_train)
    metrics_test, val_results_df = test_batch(cfg, model, X_val, y_val)
    score = metrics_test.get("f1_score", np.nan)
    return model, params, score, train_results_df, val_results_df

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
    logger.debug("Splitting features and target for training data...")
    X_train, y_train = split_features_target(cfg, data_train.data)

    logger.debug("Cutting validation data...")
    data_valid = data_handler.cut_data(valid_start, valid_end, cfg.datasets.training.datetime_column)
    logger.debug("Splitting features and target for validation data...")
    X_val, y_val = split_features_target(cfg, data_valid.data)

    logger.debug("Cutting test data...")
    data_test = data_handler.cut_data(test_start, test_end, cfg.datasets.training.datetime_column)
    logger.debug("Splitting features and target for test data...")
    X_test, y_test = split_features_target(cfg, data_test.data)

    # ---------------------------------------------
    # Hyperparameter tuning (on validation)
    # ---------------------------------------------
    hyperparameter_combinations = get_hyperparameter_combinations(cfg)
    logger.info(f"Starting hyperparameter search over {len(hyperparameter_combinations)} combinations...")

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
    best_model = None
    best_train_results_df = None
    best_val_results_df = None

    for model, params, score, train_results_df, val_results_df in results:
        logger.info(f"Params {params} → F1={score}")

        if np.isnan(score):
            logger.warning(f"F1 score is NaN for params {params}")
            continue

        if score > best_score:
            best_score = score
            best_params = params
            best_model = model
            best_train_results_df = train_results_df
            best_val_results_df = val_results_df

    if best_params is None:
        raise RuntimeError("No valid hyperparameter combination produced a score.")

    logger.info(f"Best hyperparameters: {best_params} (F1={best_score:.4f})")

    # ---------------------------------------------
    # Decision threshold tuning (on validation)
    # --------------------------------------------
    alphas = cfg.experiments.experiment_parameters.get("decision_threshold_alphas", [0.0])

    best_alpha = None
    best_profit = -np.inf
    best_optimizers = None
    logger.info(f"Best model classes: {best_model.classes_}")
    model_mapping = cfg.datasets.optimization.model_mapping
    MODEL_REGISTRY = {
        "ModelSurplus": ModelSurplus,
        "ModelBalance": ModelBalance,
        "ModelDeficit": ModelDeficit,
    }
    X_train_full = pd.concat([X_train, X_val])
    best_train_val_results_df = pd.concat([best_train_results_df,best_val_results_df])
    for alpha in alphas:
        threshold_preds_df = threshold_predictions(cfg, best_model, best_train_val_results_df.filter(like="proba_class_").to_numpy(), alpha)
        X_ = X_train_full.copy()
        # Set index of threshold_preds_df to match X_
        threshold_preds_df.index = X_.index
        X_["thresholded_label"] = threshold_preds_df["thresholded_label"]
        tot_profit_alpha = 0
        optimizers_alpha = {}
        failed_alpha = False
        logger.debug(f"Optimizing for alpha={alpha} with predicted labels: {threshold_preds_df['thresholded_label'].unique()}")
        for label_ in best_model.classes_:
            logger.info(f"Optimizing for label {label_} with alpha={alpha}...")
            datetime_index = X_[X_["thresholded_label"] == label_].index
            logger.info(f"Found {len(datetime_index)} samples for predicted label {label_} in train + val set.")
            data_optimization = data_handler.data.loc[datetime_index]
            lambda_DA_hat = data_optimization[cfg.datasets.optimization.lambda_DA_hat]
            lambda_B_hat = data_optimization[cfg.datasets.optimization.lambda_B_hat]
            P_W_hat = data_optimization[cfg.datasets.optimization.P_W_hat]
            P_W_tilde = data_optimization[cfg.datasets.optimization.P_W_tilde]
            model_name = model_mapping[int(label_)]
            model_cls = MODEL_REGISTRY[model_name]
            optimizer = model_cls(
                cfg = cfg,
                lambda_DA_hat=lambda_DA_hat,
                lambda_B_hat=lambda_B_hat,
                P_W_hat=P_W_hat,
                P_W_tilde=P_W_tilde,
            )
            # Save LP file for debugging
            root = Path(__file__).resolve().parent.parent.parent
            save_path = root / "models" / "lp_files" / f"model_alpha{alpha}_label{label_}.lp"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            optimizer.model.to_file(save_path)
            try:
                optimizer.run_optimization()
            except Exception as e:
                logger.error(
                    f"Optimization failed for alpha={alpha}, label={label_}: {e}"
                )
                failed_alpha = True
                break
            if optimizer.results.status != "ok":
                logger.error(
                    f"Optimization did not converge for alpha={alpha}, label={label_}. Status: {optimizer.results.status}"
                )
                failed_alpha = True
                break
            tot_profit_alpha += optimizer.results.objective_value
            optimizers_alpha[model_name] = optimizer
        if failed_alpha:
            logger.warning(f"Skipping alpha={alpha} due to optimization failure.")
            continue
        if tot_profit_alpha > best_profit:
            best_profit = tot_profit_alpha
            best_alpha = alpha
            best_optimizers = optimizers_alpha

    logger.info(f"Best decision threshold alpha: {best_alpha} with profit: {best_profit}")

    # ---------------------------------------------
    # Retrain on train + validation
    # ---------------------------------------------
    y_train_full = pd.concat([y_train, y_val])

    final_model = train_batch(cfg, X_train_full, y_train_full, best_params)

    # ---------------------------------------------
    # Final test evaluation
    # ---------------------------------------------
    train_metrics, train_results_df = evaluate_classifier(final_model, X_train_full, y_train_full)
    logger.info(f"Train metrics: {train_metrics}")
    test_metrics, test_results_df = evaluate_classifier(final_model, X_test, y_test)
    logger.info(f"Test metrics: {test_metrics}")

    if not alpha:
        logger.error("No alpha exists. Check optimization.")
        metrics_threshold_prediction_train = {}
        metrics_threshold_prediction_test = {}
    else:
        final_preds_train_df = threshold_predictions(cfg, final_model, train_results_df.filter(like="proba_class_").to_numpy(), best_alpha)
        final_preds_test_df = threshold_predictions(cfg, final_model, test_results_df.filter(like="proba_class_").to_numpy(), best_alpha)
        # reset index of final_preds_train_df and final_preds_test_df to match train_results_df and test_results_df
        final_preds_train_df.index = train_results_df.index
        final_preds_test_df.index = test_results_df.index

        train_results_df["thresholded_label"] = final_preds_train_df["thresholded_label"]
        train_results_df["uncertain"] = final_preds_train_df["uncertain"]
        train_results_df["uncertain"] = train_results_df["uncertain"].astype(bool)
        test_results_df["thresholded_label"] = final_preds_test_df["thresholded_label"]
        test_results_df["uncertain"] = final_preds_test_df["uncertain"]
        test_results_df["uncertain"] = test_results_df["uncertain"].astype(bool)

        # Make copy of only certain predictions (uncertain=False) and compute metrics only on those
        certain_train_results_df = train_results_df[~train_results_df["uncertain"]]
        certain_test_results_df = test_results_df[~test_results_df["uncertain"]]

        metrics_threshold_prediction_train = compute_accuracy_f1(
            certain_train_results_df["true_label"],
            certain_train_results_df["thresholded_label"],
        )
        metrics_threshold_prediction_test = compute_accuracy_f1(
            certain_test_results_df["true_label"],
            certain_test_results_df["thresholded_label"],
        )

    # ---------------------------------------------
    # Calculate bids
    # ---------------------------------------------
    logger.info("Calculating bids and profits for train and test sets...")
    datetime_index_train = X_train_full.index
    datetime_index_test = X_test.index
    data_train = data_handler.data.loc[datetime_index_train]
    data_test = data_handler.data.loc[datetime_index_test]
    bids_train = calculate_bids(cfg, data_train, final_preds_train_df, best_optimizers)
    bids_test = calculate_bids(cfg, data_test, final_preds_test_df, best_optimizers)

    # Calculate profit for train and test sets
    lambda_DA_hat_train = data_train[cfg.datasets.optimization.lambda_DA_hat]
    lambda_B_hat_train = data_train[cfg.datasets.optimization.lambda_B_hat]
    P_W_hat_train = data_train[cfg.datasets.optimization.P_W_hat]
    profit_train = calculate_profit(bids_train, lambda_DA_hat_train, lambda_B_hat_train, P_W_hat_train)
    lambda_DA_hat_test = data_test[cfg.datasets.optimization.lambda_DA_hat]
    lambda_B_hat_test = data_test[cfg.datasets.optimization.lambda_B_hat]
    P_W_hat_test = data_test[cfg.datasets.optimization.P_W_hat]
    profit_test = calculate_profit(bids_test, lambda_DA_hat_test, lambda_B_hat_test, P_W_hat_test)

    # Add bids and profit to results dfs
    train_results_df["DA_bid"] = bids_train
    train_results_df["profit"] = profit_train
    test_results_df["DA_bid"] = bids_test
    test_results_df["profit"] = profit_test
    # ---------------------------------------------
    # Collect results
    # ---------------------------------------------
    results = {
        "train_profit_total": profit_train.sum(),
        "train_profit_mean": profit_train.mean(),
        "test_profit_total": profit_test.sum(),
        "test_profit_mean": profit_test.mean(),
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **{f"train_thresholded_{k}": v for k, v in metrics_threshold_prediction_train.items()},
        **{f"test_thresholded_{k}": v for k, v in metrics_threshold_prediction_test.items()},
        "train_start": train_start,
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "test_start": test_start,
        "test_end": test_end,
        "best_alpha": best_alpha,
        **best_params,
    }
    for optimizer_name, optimizer in best_optimizers.items():
        results[f"optimizer_{optimizer_name}_x"] = optimizer.results.x

    return final_model, results, train_results_df, test_results_df

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
    all_train_results_dfs = pd.DataFrame()
    all_test_results_dfs = pd.DataFrame()
    for window in rolling_windows(cfg):
        try:
            _, results, train_results_df, test_results_df = train_model(cfg, window, data_handler)
            all_results.append(results)
            all_train_results_dfs = pd.concat([all_train_results_dfs, train_results_df])
            all_test_results_dfs = pd.concat([all_test_results_dfs, test_results_df])
        except Exception as e:
            logger.error(f"Error in window {window}: {e}")

    # Compute average metrics over all windows
    if not all_train_results_dfs.empty:
        avg_metrics_train = compute_accuracy_f1(
            all_train_results_dfs["true_label"].to_numpy(),
            all_train_results_dfs["predicted_label"].to_numpy()
        )
        avg_metrics_thresholded_train = compute_accuracy_f1(
            all_train_results_dfs["true_label"].to_numpy(),
            all_train_results_dfs["thresholded_label"].to_numpy()
        )
        for key, value in avg_metrics_train.items():
            logger.info(f"Average {key} over all windows: {value}")
        for key, value in avg_metrics_thresholded_train.items():
            logger.info(f"Average thresholded {key} over all windows: {value}")
        total_profit_train = all_train_results_dfs["profit"].sum()
        mean_profit_train = all_train_results_dfs["profit"].mean()
        logger.info(f"Total profit over all train windows: {total_profit_train}")
        logger.info(f"Mean profit over all train windows: {mean_profit_train}")
    else:
        avg_metrics_train = {}
        avg_metrics_thresholded_train = {}
        total_profit_train = 0
        mean_profit_train = 0
        logger.warning("No train results to compute average metrics.")

    if not all_test_results_dfs.empty:
        avg_metrics_test = compute_accuracy_f1(
            all_test_results_dfs["true_label"].to_numpy(),
            all_test_results_dfs["predicted_label"].to_numpy()
        )
        avg_metrics_thresholded_test = compute_accuracy_f1(
            all_test_results_dfs["true_label"].to_numpy(),
            all_test_results_dfs["thresholded_label"].to_numpy()
        )
        for key, value in avg_metrics_test.items():
            logger.info(f"Average {key} over all windows: {value}")
        for key, value in avg_metrics_thresholded_test.items():
            logger.info(f"Average thresholded {key} over all windows: {value}")
        total_profit_test = all_test_results_dfs["profit"].sum()
        mean_profit_test = all_test_results_dfs["profit"].mean()
        logger.info(f"Total profit over all test windows: {total_profit_test}")
        logger.info(f"Mean profit over all test windows: {mean_profit_test}")
    else:
        avg_metrics_test = {}
        avg_metrics_thresholded_test = {}
        total_profit_test = 0
        mean_profit_test = 0
        logger.warning("No test results to compute average metrics.")

    avg_metrics = {
        "train_accuracy": avg_metrics_train.get("accuracy", np.nan),
        "train_f1_score": avg_metrics_train.get("f1_score", np.nan),
        "test_accuracy": avg_metrics_test.get("accuracy", np.nan),
        "test_f1_score": avg_metrics_test.get("f1_score", np.nan),
    }
    avg_metrics_thresholded = {
        "train_accuracy": avg_metrics_thresholded_train.get("accuracy", np.nan),
        "train_f1_score": avg_metrics_thresholded_train.get("f1_score", np.nan),
        "test_accuracy": avg_metrics_thresholded_test.get("accuracy", np.nan),
        "test_f1_score": avg_metrics_thresholded_test.get("f1_score", np.nan),
        "train_profit_total": total_profit_train,
        "train_profit_mean": mean_profit_train,
        "test_profit_total": total_profit_test,
        "test_profit_mean": mean_profit_test,
    }
    logger.info("Backtest completed.")
    return all_results, avg_metrics, avg_metrics_thresholded

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config_dev")
def main(cfg: DictConfig) -> None:
    logger.info("Starting experiment")
    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)

    results, metrics, metrics_thresholded = run_backtest(cfg)

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
    with open(save_path / "allwindows_metrics_pure.txt", "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    with open(save_path / "allwindows_metrics_thresholded.txt", "w") as f:
        for key, value in metrics_thresholded.items():
            f.write(f"{key}: {value}\n")

    logger.info("Experiment finished successfully")

if __name__ == "__main__":
    main()
