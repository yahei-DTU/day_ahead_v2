#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File name: wasserstein.py
Author: Yannick Heiser
Created: 2026-06-17
Version: 1.0
Description:
    Lightweight backtest that mirrors the rolling-window setup of ``train.py`` but,
    instead of training any model, only measures distribution drift between the
    final training set (X_train + X_val) and the test set of each rolling window.

    Two distances are reported per window:
      - feature_wasserstein: drift of the (multivariate) feature distribution.
        By default the *sliced* Wasserstein distance is used, which averages the
        1-D Wasserstein distance over many random 1-D projections and therefore
        captures joint/correlation structure, not just per-feature marginals.
        Set ``feature_drift_method: marginal`` to instead average the per-feature
        1-D Wasserstein distances (more interpretable, ignores correlations).
      - target_wasserstein: drift of the *continuous* target distribution
        (full 1-D Wasserstein distance).

    Both distances are computed in standardized units. Features and target are
    z-scored using statistics estimated on the training set only (X_train + X_val),
    so the numbers are comparable across features and across windows and no test
    information leaks into the scaling.

Contact: yahei@dtu.dk
Dependencies:
    - pandas
    - numpy
    - scipy
    - hydra
    - src.day_ahead_v2.data (custom module)
    - src.day_ahead_v2.train (custom module, reuses rolling_windows / split_features_target)
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import copy
import logging
import random
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scipy.stats import wasserstein_distance

from day_ahead_v2.data import PandasHandler, DataHandler
from day_ahead_v2.train import rolling_windows, split_features_target, feature_selection

logger = logging.getLogger(__name__)

# Number of random projections used for the sliced Wasserstein distance.
DEFAULT_N_PROJECTIONS = 200


def _standardize(reference: pd.DataFrame, compare: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Z-score ``reference`` and ``compare`` using statistics from ``reference`` only.

    Columns are aligned to the intersection of both frames, zero-variance columns
    are dropped (they carry no distributional information), and any remaining NaN
    values are filled with 0 (the training mean in standardized space).

    Args:
        reference (pd.DataFrame): Training features (X_train + X_val).
        compare (pd.DataFrame): Test features.

    Returns:
        tuple[np.ndarray, np.ndarray]: Standardized (reference, compare) arrays
        with identical column ordering.
    """
    common = [c for c in reference.columns if c in compare.columns]
    dropped = set(reference.columns).symmetric_difference(compare.columns)
    if dropped:
        logger.warning(f"Dropping {len(dropped)} non-shared feature columns for drift: {sorted(dropped)}")
    reference = reference[common]
    compare = compare[common]

    mu = reference.mean()
    sigma = reference.std()

    # Drop zero-variance columns (sigma == 0 or NaN) -> nothing to compare.
    keep = sigma[(sigma > 0) & sigma.notna()].index
    if len(keep) < len(common):
        logger.warning(f"Dropping {len(common) - len(keep)} zero-variance feature columns for drift.")
    reference = reference[keep]
    compare = compare[keep]
    mu = mu[keep]
    sigma = sigma[keep]

    ref_z = ((reference - mu) / sigma).fillna(0.0).to_numpy()
    cmp_z = ((compare - mu) / sigma).fillna(0.0).to_numpy()
    return ref_z, cmp_z


def sliced_wasserstein_distance(
    reference: np.ndarray, compare: np.ndarray, n_projections: int, rng: np.random.Generator
) -> float:
    """
    Sliced Wasserstein distance between two multivariate sample sets.

    Projects both sample sets onto ``n_projections`` random unit directions and
    averages the closed-form 1-D Wasserstein distance over the projections. This
    approximates the true multivariate optimal-transport distance while remaining
    cheap, and is sensitive to changes in the joint (correlation) structure.

    Args:
        reference (np.ndarray): Standardized reference samples, shape (n_ref, d).
        compare (np.ndarray): Standardized comparison samples, shape (n_cmp, d).
        n_projections (int): Number of random 1-D projections.
        rng (np.random.Generator): Seeded RNG for reproducible projections.

    Returns:
        float: Mean 1-D Wasserstein distance over the random projections.
    """
    d = reference.shape[1]
    if d == 0:
        return float("nan")
    projections = rng.standard_normal(size=(d, n_projections))
    projections /= np.linalg.norm(projections, axis=0, keepdims=True)
    ref_proj = reference @ projections
    cmp_proj = compare @ projections
    distances = [wasserstein_distance(ref_proj[:, k], cmp_proj[:, k]) for k in range(n_projections)]
    return float(np.mean(distances))


def marginal_wasserstein_distance(reference: np.ndarray, compare: np.ndarray) -> float:
    """
    Mean per-feature (marginal) 1-D Wasserstein distance.

    Ignores correlations between features; reports the average over the
    standardized feature columns.

    Args:
        reference (np.ndarray): Standardized reference samples, shape (n_ref, d).
        compare (np.ndarray): Standardized comparison samples, shape (n_cmp, d).

    Returns:
        float: Mean of the per-feature 1-D Wasserstein distances.
    """
    d = reference.shape[1]
    if d == 0:
        return float("nan")
    distances = [wasserstein_distance(reference[:, k], compare[:, k]) for k in range(d)]
    return float(np.mean(distances))


def target_drift_distance(y_reference: pd.Series, y_compare: pd.Series) -> float:
    """
    1-D Wasserstein distance between two continuous-target distributions.

    The target is z-scored using the reference (training) mean and std so the
    result is in the same standardized units as the feature drift.

    Args:
        y_reference (pd.Series): Continuous target for X_train + X_val.
        y_compare (pd.Series): Continuous target for the test set.

    Returns:
        float: Standardized 1-D Wasserstein distance of the target.
    """
    ref = np.asarray(y_reference, dtype=float)
    cmp = np.asarray(y_compare, dtype=float)
    ref = ref[~np.isnan(ref)]
    cmp = cmp[~np.isnan(cmp)]
    if ref.size == 0 or cmp.size == 0:
        return float("nan")
    mu = ref.mean()
    sigma = ref.std()
    if sigma == 0 or np.isnan(sigma):
        sigma = 1.0
    return float(wasserstein_distance((ref - mu) / sigma, (cmp - mu) / sigma))


def compute_window_drift(
    cfg: DictConfig,
    window: dict,
    data_handler: DataHandler,
    n_projections: int,
    rng: np.random.Generator,
    n_features_selected: int | None = None,
) -> dict:
    """
    Compute feature and continuous-target Wasserstein drift for a single window.

    The reference distribution is the final training set (X_train + X_val), and
    the comparison distribution is the test set, mirroring how ``train.py`` fits
    the final model on train+validation before evaluating on test.

    Args:
        cfg (DictConfig): Configuration object.
        window (dict): Rolling window with "train", "valid", "test" date ranges.
        data_handler (DataHandler): Transformed data handler for this window.
        n_projections (int): Number of projections for the sliced distance.
        rng (np.random.Generator): Seeded RNG for reproducible projections.
        n_features_selected (int | None): Number of features the drift is computed
            on. Falls back to the configured feature count when selection is off.

    Returns:
        dict: Window date bounds, ``n_features``, ``feature_wasserstein`` and
        ``target_wasserstein``.
    """
    train_start, train_end = window["train"]
    valid_start, valid_end = window["valid"]
    test_start, test_end = window["test"]

    logger.info(
        f"Drift | Train: {train_start.date()} → {train_end.date()} | "
        f"Valid: {valid_start.date()} → {valid_end.date()} | "
        f"Test: {test_start.date()} → {test_end.date()}"
    )

    datetime_column = cfg.datasets.training.datetime_column
    target_column = cfg.datasets.training.target_wasserstein

    data_train = data_handler.cut_data(train_start, train_end, datetime_column)
    data_valid = data_handler.cut_data(valid_start, valid_end, datetime_column)
    data_test = data_handler.cut_data(test_start, test_end, datetime_column)

    X_train, _ = split_features_target(cfg, data_train.data, sanitize_names=True)
    X_val, _ = split_features_target(cfg, data_valid.data, sanitize_names=True)
    X_test, _ = split_features_target(cfg, data_test.data, sanitize_names=True)

    # Final training distribution = train + validation, exactly as the final model sees it.
    X_train_full = pd.concat([X_train, X_val])

    # Continuous target is pulled straight from the (engineered) data, not the
    # binary classification target used by the model.
    y_train_full = pd.concat([data_train.data[target_column], data_valid.data[target_column]])
    y_test = data_test.data[target_column]

    ref_z, cmp_z = _standardize(X_train_full, X_test)

    method = cfg.experiments.experiment_parameters.get("feature_drift_method", "sliced")
    if method == "marginal":
        feature_distance = marginal_wasserstein_distance(ref_z, cmp_z)
    else:
        feature_distance = sliced_wasserstein_distance(ref_z, cmp_z, n_projections, rng)

    target_distance = target_drift_distance(y_train_full, y_test)

    if n_features_selected is None:
        n_features_selected = X_train_full.shape[1]

    logger.info(
        f"Drift | n_features={n_features_selected} | feature_wasserstein ({method})={feature_distance:.4f} | "
        f"target_wasserstein={target_distance:.4f}"
    )

    return {
        "train_start": train_start,
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "test_start": test_start,
        "test_end": test_end,
        "n_features": n_features_selected,
        "feature_wasserstein": feature_distance,
        "target_wasserstein": target_distance,
    }


def run_drift_backtest(cfg: DictConfig) -> pd.DataFrame:
    """
    Run the drift backtest over all rolling windows.

    Replicates the data loading, per-window deep-copy of cfg, and data transform
    of ``train.run_backtest`` so the windows and feature engineering match, but
    only computes the Wasserstein distances (no model training/optimization).

    Args:
        cfg (DictConfig): Configuration object.

    Returns:
        pd.DataFrame: One row per window with date bounds and both distances.
    """
    logger.info("Starting Wasserstein drift backtest...")

    raw_data_handler = PandasHandler(cfg)
    raw_data_handler = raw_data_handler.cut_data(
        cfg.experiments.experiment_parameters.start_date,
        cfg.experiments.experiment_parameters.end_date,
        cfg.datasets.training.datetime_column,
    )

    windows = list(rolling_windows(cfg))
    logger.info(f"Running drift backtest over {len(windows)} windows...")

    n_projections = cfg.experiments.experiment_parameters.get("n_projections", DEFAULT_N_PROJECTIONS)
    rng = np.random.default_rng(cfg.seed)

    # Snapshot cfg so each window starts from the same configuration (feature
    # selection in train.py mutates cfg; we keep the same guard for parity).
    cfg_snapshot = copy.deepcopy(cfg)

    rows = []
    for window in windows:
        try:
            cfg = copy.deepcopy(cfg_snapshot)
            data_handler = raw_data_handler.transform_data(cfg)

            nan_counts = data_handler.data.isnull().sum()
            if nan_counts.sum() > 0:
                logger.warning(f"Found NaN values in data:\n{nan_counts[nan_counts > 0]}")

            # Feature selection (per window, using the training period) so that the
            # feature drift is measured only on the features the model actually uses
            # in this window — mirrors train.run_backtest.
            n_features_selected = None
            if cfg.experiments.train_parameters.get("feature_selection", False):
                selected_features = feature_selection(
                    cfg, data_handler.data, start=window["train"][0], end=window["train"][1]
                )
                cfg.datasets.training.feature_columns_flex = selected_features
                n_features_selected = len(cfg.datasets.training.feature_columns_fix) + len(selected_features)
                logger.info(f"Number of features after selection: {n_features_selected}")

            rows.append(
                compute_window_drift(cfg, window, data_handler, n_projections, rng, n_features_selected)
            )
        except Exception as e:
            logger.error(f"Error computing drift for window {window}: {e}")

    logger.info("Wasserstein drift backtest completed.")
    return pd.DataFrame(rows)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config_dev")
def main(cfg: DictConfig) -> None:
    logger.info(
        f"Starting Wasserstein drift backtest for experiment {cfg.experiments.experiment_name} "
        f"with dataset {cfg.datasets.dataset_name}"
    )
    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)

    results_df = run_drift_backtest(cfg)

    if results_df.empty:
        logger.warning("No drift results generated")
        return

    # Save under reports/<experiment>/<model>/<dataset>/, same as train.py.
    OmegaConf.resolve(cfg)
    project_root = Path(cfg.project_root)
    save_path = Path(cfg.results.save_path)
    if not save_path.is_absolute():
        save_path = project_root / save_path
    save_path.mkdir(parents=True, exist_ok=True)

    out_file = save_path / "wasserstein.csv"
    results_df.to_csv(out_file, index=False)
    logger.info(f"Wasserstein drift results saved to {out_file}")


if __name__ == "__main__":
    main()
