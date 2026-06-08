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

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import logging
import random
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import copy
from datetime import datetime, timedelta
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from joblib import Parallel, delayed
from sklearn.feature_selection import VarianceThreshold
import lightgbm as lgb
import shap
from day_ahead_v2.data import PandasHandler, DataHandler
from day_ahead_v2.optimization import (
    ModelClassPolicyHPP, ModelAllOrNothingHPP,
    ModelClassPolicyWindOnly, ModelAllOrNothingWindOnly,
    ModelHindsightPrimeHPP, ModelHindsightPrimeWindOnly
)
from day_ahead_v2.evaluate import evaluate_classifier, compute_accuracy_f1, threshold_predictions, calculate_profit, cvar_profit, mean_profit
from day_ahead_v2.utils.sanitize_names import sanitize_column_names
from day_ahead_v2.utils import electrolyzer_efficiency

_OPTIMIZER_CLASSES = {
    ("class_policy",   "hpp"):       ModelClassPolicyHPP,
    ("all_or_nothing", "hpp"):       ModelAllOrNothingHPP,
    ("class_policy",   "wind_only"): ModelClassPolicyWindOnly,
    ("all_or_nothing", "wind_only"): ModelAllOrNothingWindOnly,
    ("hindsight",      "hpp"):       ModelHindsightPrimeHPP,
    ("hindsight",      "wind_only"): ModelHindsightPrimeWindOnly,
}

logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "cvar_path",
    lambda use_cvar, limit: f"/CVaR_{limit}" if use_cvar else ""
)


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

def split_features_target(cfg: DictConfig, data: pd.DataFrame, sanitize_names: bool = False) -> tuple[pd.DataFrame, pd.Series]:
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
    feature_columns = list(cfg.datasets.training.feature_columns_fix) + list(cfg.datasets.training.feature_columns_flex)

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
    if sanitize_names:
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

def feature_selection(cfg: DictConfig, data: pd.DataFrame, start: datetime, end: datetime) -> list[str]:
    """
    Perform features selection using:
    1. Variance threshold
    2. LightGBM + SHAP values
    """
    logger.info("Performing feature selection...")

    # Cut training data for feature selection
    data_train = data.loc[(data.index >= start) & (data.index < end)].copy()
    X_train, y_train = split_features_target(cfg, data_train)
    if cfg.datasets.training.fallback_class is not None:
        mask = y_train != cfg.datasets.training.fallback_class
        X_train = X_train[mask]
        y_train = y_train[mask]
    features = cfg.datasets.training.feature_columns_flex
    if not set(features).issubset(set(X_train.columns)):
        missing = set(features) - set(X_train.columns)
        logger.warning(f"Some feature columns specified in config are missing from training data: {missing}")
        features = [col for col in features if col in X_train.columns]
    X_train = X_train[features]

    logger.info(f"Initial number of features: {X_train.shape[1]}")

    # Variance threshold
    vt = VarianceThreshold(threshold=0.01)
    _ = vt.fit_transform(X_train)
    selected_features_var = X_train.columns[vt.get_support()].tolist()
    logger.info(f"Selected {len(selected_features_var)} features after variance thresholding.")
    X_train = X_train[selected_features_var]

    # LightGBM + SHAP values — regularized for stable importance estimates, not peak accuracy
    lgb_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=50,
        reg_lambda=1.0,
        random_state=cfg.seed,
        n_jobs=-1,
    )

    lgb_model.fit(X_train, y_train)

    # SHAP values
    logger.debug("Calculating SHAP values for feature importance...")
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_train)
    if isinstance(shap_values, list): # For multi-class classification, shap_values is a list of arrays (one per class)
        shap_array = np.mean(
            [np.abs(class_shap) for class_shap in shap_values],
            axis=0
        )
    else:
        shap_array = np.abs(shap_values)
    mean_abs_shap = shap_array.mean(axis=0)

    shap_importance = pd.Series(mean_abs_shap, index=X_train.columns).sort_values(ascending=False)

    # Cumulative importance selection
    cumulative_importance = shap_importance.cumsum() / shap_importance.sum()

    shap_threshold = cfg.experiments.feature_selection_parameters.get("shap_cumulative_importance_threshold", 0.9)

    plot_flag = cfg.experiments.feature_selection_parameters.get(
        "plot_shap_importance", False
    )

    if plot_flag:
        logger.debug("Plotting SHAP feature importance...")
        save_path = Path(cfg.results.save_path)
        if not save_path.is_absolute():
            save_path = Path(__file__).resolve().parent.parent.parent / save_path
        save_path.mkdir(parents=True, exist_ok=True)
        # Plot top N features (avoid overcrowded plot)
        top_n = min(30, len(shap_importance))
        shap_top = shap_importance.head(top_n)

        plt.figure(figsize=(10, 8))
        shap_top.sort_values().plot(kind="barh")
        plt.xlabel("Mean |SHAP value|")
        plt.title("SHAP Feature Importance (Top {})".format(top_n))
        plt.tight_layout()

        plot_file = save_path / "shap_feature_importance.pdf"
        plt.savefig(plot_file)
        plt.close()
        logger.info(f"SHAP importance plot saved to {plot_file}")

        plt.figure(figsize=(10, 8))
        cumulative_importance.plot()
        plt.axhline(y=shap_threshold, color="r", linestyle="--")
        plt.xlabel("Features (ordered by importance)")
        plt.ylabel("Cumulative Importance")
        plt.title("Cumulative SHAP Feature Importance")
        plt.tight_layout()
        cumulative_plot_file = save_path / "shap_cumulative_importance.pdf"
        plt.savefig(cumulative_plot_file)
        plt.close()
        logger.info(f"Cumulative SHAP importance plot saved to {cumulative_plot_file}")

    selected_features_shap = cumulative_importance[cumulative_importance <= shap_threshold].index.tolist()

    # Ensure at least 5 features are selected
    if len(selected_features_shap) < 5:
        selected_features_shap = shap_importance.head(5).index.tolist()
        logger.warning(f"Less than 5 features selected by SHAP. Automatically selecting top 5 features: {selected_features_shap}")

    logger.info(f"Selected {len(selected_features_shap)} features based on SHAP cumulative importance threshold of {shap_threshold}.")

    logger.info(f"Final number of selected features_flex: {len(selected_features_shap)}")
    return selected_features_shap

def train_batch(cfg: DictConfig, X_train: pd.DataFrame, y_train: pd.Series, hyperparameters: dict, sample_weight: pd.Series | None = None) -> object:
    """Train model for a single rolling window and hyperparameter set."""
    logger.debug(f"Start training model for window {X_train.index.min()} to {X_train.index.max()} with hyperparameters: {hyperparameters}")
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(f"X_train must be a pd.DataFrame, got {type(X_train)}")
    if not isinstance(y_train, pd.Series):
        raise TypeError(f"y_train must be a pd.Series, got {type(y_train)}")

    base_params = OmegaConf.to_container(cfg.models.model_parameters, resolve=True)

    if cfg.models._target_.endswith("MLPClassifier"):
        base_params["input_dim"] = X_train.shape[1]

    model = instantiate({"_target_": cfg.models._target_, **base_params, **hyperparameters})

    # Filter Balance class from training data to have binary classification (Deficit vs Surplus)
    if cfg.datasets.training.fallback_class is not None:
        logger.debug(f"Filtering out fallback class '{cfg.datasets.training.fallback_class}' from training data for binary classification.")
        mask = y_train != cfg.datasets.training.fallback_class
        X_train = X_train[mask]
        y_train = y_train[mask]
        if sample_weight is not None:
            sample_weight = sample_weight[mask]

    # Fit model
    model.fit(X_train, y_train, sample_weight=sample_weight)
    logger.info(f"Model trained with hyperparameters: {hyperparameters}")

    return model

def test_batch(cfg: DictConfig, model: object, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """Test model for a single rolling window and hyperparameter set."""
    # Predict and evaluate on validation set
    val_metrics, val_results_df = evaluate_classifier(model, X_val, y_val, fallback_class=cfg.datasets.training.fallback_class)
    logger.info(f"Validation metrics: {val_metrics}")

    return val_metrics, val_results_df

def train_and_validate_params(
        cfg: DictConfig,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict,
        sample_weight: pd.Series | None = None,
    ) -> tuple[dict, float]:
    logger.debug(f"Training and validating with params: {params}")
    model = train_batch(cfg, X_train, y_train, params, sample_weight=sample_weight)
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
    X_train, y_train = split_features_target(cfg, data_train.data, sanitize_names=True)
    logger.info(f"Target counts in training data:\n{y_train.value_counts()}")
    use_sample_weighting = cfg.experiments.train_parameters.get("sample_weighting", False)
    if use_sample_weighting:
        sample_weight_train = pd.Series(
            np.abs(
                data_train.data[cfg.datasets.optimization.lambda_DA_hat].values -
                data_train.data[cfg.datasets.optimization.lambda_B_hat].values
            ),
            index=X_train.index,
        )
    else:
        sample_weight_train = None

    logger.debug("Cutting validation data...")
    data_valid = data_handler.cut_data(valid_start, valid_end, cfg.datasets.training.datetime_column)
    logger.debug("Splitting features and target for validation data...")
    X_val, y_val = split_features_target(cfg, data_valid.data, sanitize_names=True)
    logger.info(f"Target counts in validation data:\n{y_val.value_counts()}")

    logger.debug("Cutting test data...")
    data_test = data_handler.cut_data(test_start, test_end, cfg.datasets.training.datetime_column)
    logger.debug("Splitting features and target for test data...")
    X_test, y_test = split_features_target(cfg, data_test.data, sanitize_names=True)
    logger.info(f"Target counts in test data:\n{y_test.value_counts()}")

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
            sample_weight=sample_weight_train,
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
    portfolio = cfg.experiments.optimization_parameters.portfolio
    is_hpp = portfolio == "hpp"

    alphas = cfg.experiments.experiment_parameters.get("decision_threshold_alphas", [0.5])

    metric_name = cfg.experiments.experiment_parameters.get('threshold_alpha_tuning', 'mean_profit')
    best_alpha = None
    best_metric_profit = -np.inf
    logger.info(f"Best model classes: {best_model.classes_}")

    if metric_name == 'accuracy':
        fallback_class = cfg.datasets.training.fallback_class
        val_proba = best_val_results_df.filter(like="proba_class_").to_numpy()
        mask = y_val.values != fallback_class
        for alpha in alphas:
            threshold_preds_df_val = threshold_predictions(cfg, val_proba, alpha)
            threshold_preds_df_val.index = X_val.index
            acc = (threshold_preds_df_val["thresholded_label"].values[mask] == y_val.values[mask]).mean()
            logger.info(f"Alpha {alpha} → accuracy on validation set: {acc:.4f}")
            if acc > best_metric_profit:
                best_metric_profit = acc
                best_alpha = alpha
        logger.info(f"Best decision threshold alpha: {best_alpha} with accuracy={best_metric_profit:.4f} on validation set")
    else:
        for alpha in alphas:
            threshold_preds_df_train = threshold_predictions(cfg, best_train_results_df.filter(like="proba_class_").to_numpy(), alpha)
            threshold_preds_df_val = threshold_predictions(cfg, best_val_results_df.filter(like="proba_class_").to_numpy(), alpha)
            X_ = X_train.copy()
            # Set index of threshold_preds_df to match X_
            threshold_preds_df_train.index = X_.index
            X_["thresholded_label"] = threshold_preds_df_train["thresholded_label"]
            X_["uncertain"] = threshold_preds_df_train["uncertain"]
            X_["predicted_proba"] = threshold_preds_df_train["predicted_proba"]
            failed_alpha = False
            logger.debug(f"Optimizing for alpha={alpha} with predicted labels: {threshold_preds_df_train['thresholded_label'].unique()}")
            del threshold_preds_df_train
            # Check if any count is below threshold
            label_counts = X_["thresholded_label"].value_counts()
            logger.debug(f"Predicted label counts for alpha={alpha}:\n{label_counts}")
            for label_, count in label_counts.items():
                logger.info(f"Found {count} samples for predicted label {label_} in validation set.")
            filtered_counts = label_counts[label_counts.index != cfg.datasets.training.fallback_class]
            rare_labels = filtered_counts[filtered_counts < 50].index.tolist()
            if rare_labels:
                logger.warning(
                    f"Labels {rare_labels} have fewer than 50 samples for alpha={alpha}. Reassigning to fallback_class."
                )
                X_.loc[X_["thresholded_label"].isin(rare_labels), "thresholded_label"] = cfg.datasets.training.fallback_class
            datetime_index = X_.index
            data_optimization = data_handler.data.loc[datetime_index]
            X_features = pd.concat([X_train.loc[datetime_index], X_.loc[datetime_index][["predicted_proba"]]], axis=1)
            lambda_DA_hat = data_optimization[cfg.datasets.optimization.lambda_DA_hat]
            lambda_B_hat = data_optimization[cfg.datasets.optimization.lambda_B_hat]
            P_W_hat = data_optimization[cfg.datasets.optimization.P_W_hat]
            P_W_tilde = data_optimization[cfg.datasets.optimization.P_W_tilde]
            optimizer_cls = _OPTIMIZER_CLASSES[(cfg.experiments.optimization_parameters.model, portfolio)]
            optimizer = optimizer_cls(
                cfg = cfg,
                lambda_DA_hat = lambda_DA_hat,
                lambda_B_hat = lambda_B_hat,
                P_W_hat = P_W_hat,
                P_W_tilde = P_W_tilde,
                X_features = X_features,
                pred_labels= X_["thresholded_label"]
            )
            optimizer.build_model()
            # Save LP file for debugging
            root = Path(cfg.project_root)
            save_path = root / "models" / "lp_files" / f"model_alpha{alpha}.lp"
            try:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                optimizer.model.to_file(save_path)
            except OSError as e:
                logger.warning(f"Could not save LP file to {save_path}: {e}")
            try:
                optimizer.run_optimization()
            except Exception as e:
                logger.error(
                    f"Optimization failed for alpha={alpha}: {e}"
                )
                failed_alpha = True
                break
            if optimizer.results.status != "ok":
                logger.error(
                    f"Optimization did not converge for alpha={alpha}. Status: {optimizer.results.status}"
                )
                failed_alpha = True
                break
            if failed_alpha:
                logger.warning(f"Skipping alpha={alpha} due to optimization failure.")
                continue

            # Calculate profit on validation set using optimized bids
            X_ = X_val.copy()
            threshold_preds_df_val.index = X_.index
            if rare_labels:
                threshold_preds_df_val.loc[threshold_preds_df_val["thresholded_label"].isin(rare_labels), "thresholded_label"] = cfg.datasets.training.fallback_class
                logger.info(f"Validation: reassigned rare labels {rare_labels} to fallback_class to match training.")
            X_["thresholded_label"] = threshold_preds_df_val["thresholded_label"]
            X_["uncertain"] = threshold_preds_df_val["uncertain"]
            X_["predicted_proba"] = threshold_preds_df_val["predicted_proba"]
            label_counts = X_["thresholded_label"].value_counts()
            logger.debug(f"Predicted label counts for alpha={alpha}:\n{label_counts}")
            datetime_index = X_.index
            data_optimization = data_handler.data.loc[datetime_index]
            X_features = pd.concat([X_val.loc[datetime_index], X_.loc[datetime_index][["predicted_proba"]]], axis=1)
            val_bids = optimizer.calculate_bids(cfg, data_optimization, X_features, threshold_preds_df_val)
            p_DA_val, p_B_val = val_bids[0], val_bids[1]
            h_val = val_bids[3] if is_hpp else pd.Series(0.0, index=p_DA_val.index)
            lambda_DA_hat_val = data_optimization[cfg.datasets.optimization.lambda_DA_hat]
            lambda_B_hat_val = data_optimization[cfg.datasets.optimization.lambda_B_hat]
            h2_price = cfg.experiments.optimization_parameters.hydrogen_price if is_hpp else 0.0
            profit_val = calculate_profit(p_DA_val, h_val, p_B_val, lambda_DA_hat_val, h2_price, lambda_B_hat_val)
            profit_metric_val = getattr(sys.modules[__name__], metric_name)(profit_val)
            logger.info(f"Alpha {alpha} → {metric_name} on validation set: {profit_metric_val:.2f}")

            if profit_metric_val > best_metric_profit:
                best_metric_profit = profit_metric_val
                best_alpha = alpha

        logger.info(f"Best decision threshold alpha: {best_alpha} with {metric_name}={best_metric_profit:.2f} on validation set")

    # ---------------------------------------------
    # Retrain on train + validation
    # ---------------------------------------------
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    if use_sample_weighting:
        sample_weight_val = pd.Series(
            np.abs(
                data_valid.data[cfg.datasets.optimization.lambda_DA_hat].values -
                data_valid.data[cfg.datasets.optimization.lambda_B_hat].values
            ),
            index=X_val.index,
        )
        sample_weight_full = pd.concat([sample_weight_train, sample_weight_val])
    else:
        sample_weight_full = None

    final_model = train_batch(cfg, X_train_full, y_train_full, best_params, sample_weight=sample_weight_full)

    try:
        booster = final_model.booster_
        fig, ax = plt.subplots(figsize=(10, max(6, len(booster.feature_name()) // 3)))
        lgb.plot_importance(booster, importance_type="gain", ax=ax, title="Feature importance (gain)")
        plt.tight_layout()
        fi_path = Path(cfg.project_root) / "models" / "feature_importance.png"
        fi_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fi_path)
        plt.close(fig)
        logger.info(f"Feature importance plot saved to {fi_path}")
    except Exception as e:
        logger.warning(f"Could not save feature importance plot: {e}")

    # ---------------------------------------------
    # Final test evaluation
    # ---------------------------------------------
    train_metrics, train_results_df = evaluate_classifier(final_model, X_train_full, y_train_full, fallback_class=cfg.datasets.training.fallback_class)
    logger.info(f"Train metrics: {train_metrics}")
    test_metrics, test_results_df = evaluate_classifier(final_model, X_test, y_test, fallback_class=cfg.datasets.training.fallback_class)
    logger.info(f"Test metrics: {test_metrics}")

    if best_alpha is None:
        logger.error("No alpha exists. Check optimization.")
        metrics_threshold_prediction_train = {}
        metrics_threshold_prediction_test = {}
    else:
        final_preds_train_df = threshold_predictions(cfg, train_results_df.filter(like="proba_class_").to_numpy(), best_alpha)
        final_preds_test_df = threshold_predictions(cfg, test_results_df.filter(like="proba_class_").to_numpy(), best_alpha)
        # reset index of final_preds_train_df and final_preds_test_df to match train_results_df and test_results_df
        final_preds_train_df.index = train_results_df.index
        final_preds_test_df.index = test_results_df.index

        train_results_df["thresholded_label"] = final_preds_train_df["thresholded_label"]
        train_results_df["uncertain"] = final_preds_train_df["uncertain"]
        train_results_df["predicted_proba"] = final_preds_train_df["predicted_proba"]
        train_results_df["uncertain"] = train_results_df["uncertain"].astype(bool)
        test_results_df["thresholded_label"] = final_preds_test_df["thresholded_label"]
        test_results_df["uncertain"] = final_preds_test_df["uncertain"]
        test_results_df["predicted_proba"] = final_preds_test_df["predicted_proba"]
        test_results_df["uncertain"] = test_results_df["uncertain"].astype(bool)

        # Make copy of only certain predictions (uncertain=False) and compute metrics only on those
        certain_train_results_df = train_results_df[~train_results_df["uncertain"]]
        certain_test_results_df = test_results_df[~test_results_df["uncertain"]]

        metrics_threshold_prediction_train = compute_accuracy_f1(
            certain_train_results_df["true_label"],
            certain_train_results_df["thresholded_label"],
            y_score=certain_train_results_df.get("proba_class_1"),
        )
        metrics_threshold_prediction_test = compute_accuracy_f1(
            certain_test_results_df["true_label"],
            certain_test_results_df["thresholded_label"],
            y_score=certain_test_results_df.get("proba_class_1"),
        )

    # ---------------------------------------------
    # Final policy and profit calculation
    # ---------------------------------------------
    logger.info("Calculating bids and profits for train and test sets...")
    datetime_index_train = X_train_full.index
    datetime_index_test = X_test.index
    data_train = data_handler.data.loc[datetime_index_train]
    data_test = data_handler.data.loc[datetime_index_test]

    # Reassign rare labels (< 50 samples) to fallback_class in final train predictions
    final_label_counts = final_preds_train_df["thresholded_label"].value_counts()
    final_filtered_counts = final_label_counts[final_label_counts.index != cfg.datasets.training.fallback_class]
    final_rare_labels = final_filtered_counts[final_filtered_counts < 50].index.tolist()
    if final_rare_labels:
        logger.warning(f"Final train: labels {final_rare_labels} have fewer than 50 samples. Reassigning to fallback_class.")
        final_preds_train_df.loc[final_preds_train_df["thresholded_label"].isin(final_rare_labels), "thresholded_label"] = cfg.datasets.training.fallback_class

    X_features = pd.concat([X_train_full.loc[datetime_index_train], final_preds_train_df.loc[datetime_index_train][["predicted_proba"]]], axis=1)
    lambda_DA_hat = data_train[cfg.datasets.optimization.lambda_DA_hat]
    lambda_B_hat = data_train[cfg.datasets.optimization.lambda_B_hat]
    P_W_hat = data_train[cfg.datasets.optimization.P_W_hat]
    P_W_tilde = data_train[cfg.datasets.optimization.P_W_tilde]
    optimizer_cls = _OPTIMIZER_CLASSES[(cfg.experiments.optimization_parameters.model, portfolio)]
    optimizer_final = optimizer_cls(
        cfg = cfg,
        lambda_DA_hat = lambda_DA_hat,
        lambda_B_hat = lambda_B_hat,
        P_W_hat = P_W_hat,
        P_W_tilde = P_W_tilde,
        X_features = X_features,
        pred_labels= final_preds_train_df["thresholded_label"]
    )
    optimizer_final.build_model()
    # Save LP file for debugging
    root = Path(cfg.project_root)
    save_path = root / "models" / "lp_files" / f"model_alpha{alpha}.lp"
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        optimizer_final.model.to_file(save_path)
    except OSError as e:
        logger.warning(f"Could not save LP file to {save_path}: {e}")
    try:
        optimizer_final.run_optimization()
    except Exception as e:
        logger.error(f"Final optimization failed: {e}")
        raise
    if optimizer_final.results.status != "ok":
        raise RuntimeError(
            f"Final optimization did not converge for training data in window {window}. Status: {optimizer_final.results.status}"
        )
    p_DA_train = optimizer_final.results.p_DA
    p_B_train = optimizer_final.results.p_B
    p_H_train = optimizer_final.results.p_H if is_hpp else pd.Series(0.0, index=p_DA_train.index)
    h_train = optimizer_final.results.h if is_hpp else pd.Series(0.0, index=p_DA_train.index)

    if final_rare_labels:
        final_preds_test_df.loc[final_preds_test_df["thresholded_label"].isin(final_rare_labels), "thresholded_label"] = cfg.datasets.training.fallback_class
        logger.info(f"Final test: reassigned rare labels {final_rare_labels} to fallback_class to match final training.")
    X_features = pd.concat([X_test.loc[datetime_index_test], final_preds_test_df.loc[datetime_index_test][["predicted_proba"]]], axis=1)
    test_bids = optimizer_final.calculate_bids(cfg, data_test, X_features, final_preds_test_df)
    p_DA_test, p_B_test = test_bids[0], test_bids[1]
    p_H_test = test_bids[2] if is_hpp else pd.Series(0.0, index=p_DA_test.index)
    h_test = test_bids[3] if is_hpp else pd.Series(0.0, index=p_DA_test.index)

    # Calculate profit for train and test sets
    lambda_DA_hat_train = data_train[cfg.datasets.optimization.lambda_DA_hat]
    lambda_B_hat_train = data_train[cfg.datasets.optimization.lambda_B_hat]
    h2_price = cfg.experiments.optimization_parameters.hydrogen_price if is_hpp else 0.0
    profit_train = calculate_profit(p_DA_train, h_train, p_B_train, lambda_DA_hat_train, h2_price, lambda_B_hat_train)
    lambda_DA_hat_test = data_test[cfg.datasets.optimization.lambda_DA_hat]
    lambda_B_hat_test = data_test[cfg.datasets.optimization.lambda_B_hat]
    profit_test = calculate_profit(p_DA_test, h_test, p_B_test, lambda_DA_hat_test, h2_price, lambda_B_hat_test)
    # Add bids and profit to results dfs
    train_results_df["p_DA"] = p_DA_train
    train_results_df["p_B"] = p_B_train
    train_results_df["p_H"] = p_H_train
    train_results_df["h"] = h_train
    train_results_df["profit"] = profit_train
    train_results_df["lambda_DA_hat"] = lambda_DA_hat_train
    train_results_df["lambda_B_hat"] = lambda_B_hat_train
    train_results_df["P_W_hat"] = data_train[cfg.datasets.optimization.P_W_hat]
    train_results_df["P_W_tilde"] = data_train[cfg.datasets.optimization.P_W_tilde]
    test_results_df["p_DA"] = p_DA_test
    test_results_df["p_B"] = p_B_test
    test_results_df["p_H"] = p_H_test
    test_results_df["h"] = h_test
    test_results_df["profit"] = profit_test
    test_results_df["lambda_DA_hat"] = lambda_DA_hat_test
    test_results_df["lambda_B_hat"] = lambda_B_hat_test
    test_results_df["P_W_hat"] = data_test[cfg.datasets.optimization.P_W_hat]
    test_results_df["P_W_tilde"] = data_test[cfg.datasets.optimization.P_W_tilde]
    # ---------------------------------------------
    # Collect results
    # ---------------------------------------------
    results = {
        "train_profit_total": profit_train.sum(),
        "train_profit_mean": profit_train.mean(),
        "train_profit_std": profit_train.std(),
        "train_profit_cvar": profit_train[profit_train <= np.percentile(profit_train, 5)].mean(),
        "test_profit_total": profit_test.sum(),
        "test_profit_mean": profit_test.mean(),
        "test_profit_std": profit_test.std(),
        "test_profit_cvar": profit_test[profit_test <= np.percentile(profit_test, 5)].mean(),
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

    return final_model, results, train_results_df, test_results_df

def run_backtest(cfg: DictConfig) -> list:
    """Run backtest over all rolling windows."""
    logger.info("Starting backtest...")

    # ---------------------------------------------
    # Data import and preprocessing
    # ---------------------------------------------
    raw_data_handler = PandasHandler(cfg)
    raw_data_handler = raw_data_handler.cut_data(cfg.experiments.experiment_parameters.start_date,
                                        cfg.experiments.experiment_parameters.end_date,
                                        cfg.datasets.training.datetime_column,
                                        )

    # ----------------------------------------------
    # Initialize the electrolyzer (HPP only)
    # ----------------------------------------------
    if cfg.experiments.optimization_parameters.portfolio == "hpp":
        electrolyzer_efficiency.initiate_HYP_L(cfg)

    windows = list(rolling_windows(cfg))
    logger.info(f"Running backtest over {len(windows)} windows...")

    # ---------------------------------------------
    # Rolling window backtest
    # ---------------------------------------------
    all_results = []
    all_train_results_dfs = pd.DataFrame()
    all_test_results_dfs = pd.DataFrame()

    # Create a snapshot of cfg to ensure each window starts with the same configuration (important if feature selection modifies cfg)
    cfg_snapshot = copy.deepcopy(cfg)

    for window in windows:
        try:
            # Make a deep copy of cfg for current window to avoid side effects
            cfg = copy.deepcopy(cfg_snapshot)

            # Transform data for current window
            data_handler = raw_data_handler.transform_data(cfg)

            # Check for NaNs in data
            nan_counts = data_handler.data.isnull().sum()
            if nan_counts.sum() > 0:
                logger.warning(f"Found NaN values in data:\n{nan_counts[nan_counts > 0]}")
            else:
                logger.info("No NaN values found in data")

            # Feature selection (optional, only on first window to avoid data leakage)
            if cfg.experiments.train_parameters.get("feature_selection", False) and window == windows[0]:
                selected_features = feature_selection(cfg, data_handler.data, start=window["train"][0], end=window["train"][1])
                cfg.datasets.training.feature_columns_flex = selected_features
                cfg_snapshot.datasets.training.feature_columns_flex = selected_features

            # Train model and evaluate on test set for current window
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
        all_train_results_dfs_certain = all_train_results_dfs[~all_train_results_dfs["uncertain"]]
        avg_metrics_thresholded_train = compute_accuracy_f1(
            all_train_results_dfs_certain["true_label"].to_numpy(),
            all_train_results_dfs_certain["thresholded_label"].to_numpy()
        )
        for key, value in avg_metrics_train.items():
            logger.info(f"Average {key} over all windows: {value}")
        for key, value in avg_metrics_thresholded_train.items():
            logger.info(f"Average thresholded {key} over all windows: {value}")
        total_profit_train = all_train_results_dfs["profit"].sum()
        mean_profit_train = all_train_results_dfs["profit"].mean()
        std_profit_train = all_train_results_dfs["profit"].std()
        profit_train = all_train_results_dfs["profit"]
        cvar95_train = cvar_profit(profit_train, 0.05)
        logger.info(f"Total profit over all train windows: {total_profit_train}")
        logger.info(f"Mean profit over all train windows: {mean_profit_train}")
        logger.info(f"CVaR 95% over all train windows: {cvar95_train}")
    else:
        avg_metrics_train = {}
        avg_metrics_thresholded_train = {}
        total_profit_train = 0
        mean_profit_train = 0
        std_profit_train = np.nan
        cvar95_train = np.nan
        logger.warning("No train results to compute average metrics.")

    if not all_test_results_dfs.empty:
        avg_metrics_test = compute_accuracy_f1(
            all_test_results_dfs["true_label"].to_numpy(),
            all_test_results_dfs["predicted_label"].to_numpy()
        )
        all_test_results_dfs_certain = all_test_results_dfs[~all_test_results_dfs["uncertain"]]
        avg_metrics_thresholded_test = compute_accuracy_f1(
            all_test_results_dfs_certain["true_label"].to_numpy(),
            all_test_results_dfs_certain["thresholded_label"].to_numpy()
        )
        for key, value in avg_metrics_test.items():
            logger.info(f"Average {key} over all windows: {value}")
        for key, value in avg_metrics_thresholded_test.items():
            logger.info(f"Average thresholded {key} over all windows: {value}")
        total_profit_test = all_test_results_dfs["profit"].sum()
        mean_profit_test = all_test_results_dfs["profit"].mean()
        std_profit_test = all_test_results_dfs["profit"].std()
        profit_test = all_test_results_dfs["profit"]
        cvar95_test = cvar_profit(profit_test, 0.05)
        logger.info(f"Total profit over all test windows: {total_profit_test}")
        logger.info(f"Mean profit over all test windows: {mean_profit_test}")
        logger.info(f"CVaR 95% over all test windows: {cvar95_test}")
    else:
        avg_metrics_test = {}
        avg_metrics_thresholded_test = {}
        total_profit_test = 0
        mean_profit_test = 0
        std_profit_test = np.nan
        cvar95_test = np.nan
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
        "train_profit_mean": f"{mean_profit_train:.2f} ± {std_profit_train:.2f}",
        "train_profit_cvar": cvar95_train,
        "test_profit_total": total_profit_test,
        "test_profit_mean": f"{mean_profit_test:.2f} ± {std_profit_test:.2f}",
        "test_profit_cvar": cvar95_test,
    }
    logger.info("Backtest completed.")
    return all_results, avg_metrics, avg_metrics_thresholded, all_train_results_dfs, all_test_results_dfs, cfg_snapshot

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config_dev")
def main(cfg: DictConfig) -> None:
    logger.info(f"Starting experiment {cfg.experiments.experiment_name} with dataset {cfg.datasets.dataset_name} and model {cfg.models.model_name}")
    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)

    results, metrics, metrics_thresholded, all_train_results_dfs, all_test_results_dfs, cfg = run_backtest(cfg)

    if not results:
        logger.warning("No results generated")
        return

    # Save results to CSV
    OmegaConf.resolve(cfg)
    project_root = Path(cfg.project_root)
    save_path = Path(cfg.results.save_path)
    if not save_path.is_absolute():
        save_path = project_root / save_path
    save_path.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, save_path / "config.yaml")
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

    # Save all train and test results to CSV files
    all_train_results_dfs.to_csv(save_path / "all_train_results_hourly.csv", index=True)
    logger.info(f"All train results saved to {save_path / 'all_train_results_hourly.csv'}")
    all_test_results_dfs.to_csv(save_path / "all_test_results_hourly.csv", index=True)
    logger.info(f"All test results saved to {save_path / 'all_test_results_hourly.csv'}")

    logger.info("Experiment finished successfully")

if __name__ == "__main__":
    main()
