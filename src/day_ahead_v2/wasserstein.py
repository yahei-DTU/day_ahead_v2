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
      - joint_wasserstein: drift of the *joint* feature+target distribution
        P(X, y). The standardized target is appended as one extra dimension to
        the standardized feature matrix and the sliced Wasserstein distance is
        computed over the augmented space. Because P(X, y) = P(X)·P(y|X), this
        captures covariate shift *and* concept drift (changes in the
        feature->target relationship) in a single number — unlike the two
        marginal distances above. The target dimension can be re-weighted via
        ``joint_target_weight`` (default 1.0 treats it as one more standardized
        feature; set it to ~sqrt(n_features) to balance the target against the
        whole feature block, which otherwise dilutes it across many projections).

    All distances are computed in standardized units. Features and target are
    z-scored using statistics estimated on the training set only (X_train + X_val),
    so the numbers are comparable across features and across windows and no test
    information leaks into the scaling.

    Alongside the drift, each window also reports the strength of the
    feature->target relationship, separately for the training and test sets:
      - {train,test}_feature_target_corr: mean absolute Pearson correlation over
        the shared features, i.e. the average *linear* feature->target dependence.
      - {train,test}_feature_target_mi: mean mutual information over the shared
        features, which additionally captures *non-linear* dependence. Comparing
        the train and test columns shows whether the feature->target relationship
        the model relies on still holds out of sample.

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
from sklearn.feature_selection import mutual_info_regression

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


def _standardize_target_aligned(
    y_reference: pd.Series, y_compare: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """
    Z-score the target keeping every row (NaN -> 0), for row-aligned stacking.

    Unlike :func:`target_drift_distance`, which drops NaN rows before measuring
    the 1-D marginal, this helper preserves the row count so the standardized
    target can be appended as an extra column to the standardized feature matrix
    (which also keeps all rows and fills NaN with 0). The reference mean/std are
    used for both arrays so no test information leaks into the scaling.

    Args:
        y_reference (pd.Series): Continuous target for X_train + X_val.
        y_compare (pd.Series): Continuous target for the test set.

    Returns:
        tuple[np.ndarray, np.ndarray]: Standardized (reference, compare) target
        arrays with the same length as their inputs.
    """
    ref = np.asarray(y_reference, dtype=float)
    cmp = np.asarray(y_compare, dtype=float)
    mu = np.nanmean(ref) if np.isfinite(ref).any() else 0.0
    sigma = np.nanstd(ref)
    if sigma == 0 or np.isnan(sigma):
        sigma = 1.0
    ref_z = np.nan_to_num((ref - mu) / sigma, nan=0.0)
    cmp_z = np.nan_to_num((cmp - mu) / sigma, nan=0.0)
    return ref_z, cmp_z


def joint_wasserstein_distance(
    feature_ref: np.ndarray,
    feature_cmp: np.ndarray,
    target_ref: np.ndarray,
    target_cmp: np.ndarray,
    target_weight: float,
    n_projections: int,
    rng: np.random.Generator,
) -> float:
    """
    Sliced Wasserstein distance over the joint feature+target distribution.

    The standardized target is appended (after multiplying by ``target_weight``)
    as one extra column to the standardized feature matrix, and the sliced
    Wasserstein distance is computed over the augmented space. This measures drift
    in P(X, y) and therefore captures covariate shift *and* concept drift in a
    single number. The joint metric is always computed via random projections
    (the ``marginal`` feature method has no meaningful joint analogue, since the
    whole point of the joint distance is the cross feature-target structure).

    Args:
        feature_ref (np.ndarray): Standardized reference features, shape (n_ref, d).
        feature_cmp (np.ndarray): Standardized comparison features, shape (n_cmp, d).
        target_ref (np.ndarray): Standardized reference target, shape (n_ref,).
        target_cmp (np.ndarray): Standardized comparison target, shape (n_cmp,).
        target_weight (float): Scaling applied to the target column before
            stacking. 1.0 treats the target as one more standardized feature;
            larger values counteract its dilution across random projections.
        n_projections (int): Number of random 1-D projections.
        rng (np.random.Generator): Seeded RNG for reproducible projections.

    Returns:
        float: Sliced Wasserstein distance over the augmented feature+target space.
    """
    ref_aug = np.column_stack([feature_ref, target_weight * target_ref])
    cmp_aug = np.column_stack([feature_cmp, target_weight * target_cmp])
    return sliced_wasserstein_distance(ref_aug, cmp_aug, n_projections, rng)


def _common_feature_columns(reference: pd.DataFrame, compare: pd.DataFrame) -> list:
    """
    Columns shared by both frames that have non-zero variance in the reference.

    Mirrors the column selection used by :func:`_standardize` so the
    feature->target dependence metrics are measured on the same feature set as
    the feature drift.

    Args:
        reference (pd.DataFrame): Training features (X_train + X_val).
        compare (pd.DataFrame): Test features.

    Returns:
        list: Column names present in both frames with reference variance > 0.
    """
    common = [c for c in reference.columns if c in compare.columns]
    sigma = reference[common].std()
    return list(sigma[(sigma > 0) & sigma.notna()].index)


def feature_target_correlation(X: pd.DataFrame, y: pd.Series) -> float:
    """
    Mean absolute Pearson correlation between each feature and the target.

    Captures the average strength of the *linear* feature->target relationship.
    Correlations are computed per column over rows where both the feature and the
    target are finite, then averaged in absolute value (signed correlations would
    cancel across features and understate the dependence). Columns with no
    variance over the valid rows contribute nothing and are skipped.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Continuous target, row-aligned with ``X``.

    Returns:
        float: Mean of the per-feature absolute Pearson correlations, or NaN when
        no column yields a defined correlation.
    """
    Xv = np.asarray(X, dtype=float)
    yv = np.asarray(y, dtype=float)
    if Xv.shape[0] < 2 or Xv.ndim != 2 or Xv.shape[1] == 0:
        return float("nan")

    corrs = []
    for k in range(Xv.shape[1]):
        col = Xv[:, k]
        mask = np.isfinite(col) & np.isfinite(yv)
        if mask.sum() < 2:
            continue
        c = col[mask]
        t = yv[mask]
        if c.std() == 0 or t.std() == 0:
            continue
        corrs.append(abs(np.corrcoef(c, t)[0, 1]))

    if not corrs:
        return float("nan")
    return float(np.mean(corrs))


def feature_target_mutual_information(X: pd.DataFrame, y: pd.Series, seed: int) -> float:
    """
    Mean mutual information between each feature and the continuous target.

    Unlike Pearson correlation, mutual information also captures *non-linear*
    dependence. MI is estimated per feature with sklearn's nearest-neighbour
    estimator (:func:`mutual_info_regression`), which returns one non-negative
    score per feature (in nats); the scores are averaged. Rows whose target is
    non-finite are dropped, remaining non-finite feature entries are filled with
    their column mean, and zero-variance columns are excluded.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Continuous target, row-aligned with ``X``.
        seed (int): Random seed for the MI estimator's k-NN tie-breaking, for
            reproducible scores.

    Returns:
        float: Mean mutual information over the feature columns, or NaN when the
        estimate is undefined (too few rows / no usable columns).
    """
    Xv = np.asarray(X, dtype=float)
    yv = np.asarray(y, dtype=float)
    if Xv.ndim != 2 or Xv.shape[1] == 0:
        return float("nan")

    row_mask = np.isfinite(yv)
    Xv = Xv[row_mask]
    yv = yv[row_mask]
    if Xv.shape[0] < 3:
        return float("nan")

    # Fill remaining non-finite feature entries with their column mean.
    col_means = np.nanmean(np.where(np.isfinite(Xv), Xv, np.nan), axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    bad = ~np.isfinite(Xv)
    Xv[bad] = np.take(col_means, np.nonzero(bad)[1])

    keep = Xv.std(axis=0) > 0
    Xv = Xv[:, keep]
    if Xv.shape[1] == 0:
        return float("nan")

    mi = mutual_info_regression(Xv, yv, random_state=seed)
    return float(np.mean(mi))


def _to_utc(ts) -> pd.Timestamp:
    """Coerce a timestamp(-like) value to a tz-aware UTC ``pd.Timestamp``."""
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def load_deadband_quantiles(cfg: DictConfig) -> dict | None:
    """
    Load the per-window deadband quantile chosen in a previous backtest.

    Reads the CSV at ``experiment_parameters.deadband_results_path`` (resolved
    relative to ``cfg.project_root`` when not absolute) and builds a lookup from
    each window's ``(train_start, test_start)`` to its ``deadband_quantile``.
    Timestamps are normalized to UTC so they match the windows produced by
    :func:`rolling_windows`.

    Args:
        cfg (DictConfig): Configuration object.

    Returns:
        dict | None: Mapping ``(train_start, test_start) -> deadband_quantile``,
        or ``None`` when no path is configured / the file is missing / it lacks
        the required columns (in which case drift is computed on the full
        train+validation set, as before).
    """
    path = cfg.experiments.experiment_parameters.get("deadband_results_path", None)
    if not path:
        return None

    results_path = Path(path)
    if not results_path.is_absolute():
        results_path = Path(cfg.project_root) / results_path
    if not results_path.exists():
        logger.warning(
            f"deadband_results_path '{results_path}' not found; computing drift on the full "
            f"train+validation set without deadband reconstruction."
        )
        return None

    df = pd.read_csv(results_path)
    required = {"train_start", "test_start", "deadband_quantile"}
    missing = required - set(df.columns)
    if missing:
        logger.warning(
            f"deadband_results_path '{results_path}' is missing columns {sorted(missing)}; "
            f"skipping deadband reconstruction."
        )
        return None

    lookup = {
        (_to_utc(row["train_start"]), _to_utc(row["test_start"])): float(row["deadband_quantile"])
        for _, row in df.iterrows()
    }
    logger.info(f"Loaded deadband quantiles for {len(lookup)} windows from {results_path}")
    return lookup


def apply_training_deadband(
    X_train_full: pd.DataFrame,
    y_continuous_full: pd.Series,
    y_class_full: pd.Series,
    price_spread_full: pd.Series,
    deadband_quantile: float,
    fallback_class,
) -> tuple[pd.DataFrame, pd.Series, int]:
    """
    Reconstruct the rows a model is actually trained on for a single window.

    Mirrors the filtering order in :func:`train.train_batch`: first drop the
    fallback class (so the classifier is binary), then drop the lowest-spread
    fraction of the *remaining* samples (the deadband). The spread quantile is
    therefore taken over the post-fallback samples, exactly as in training, so
    the retained set matches the samples the final model saw. Masking is
    positional to stay robust to duplicate datetime indices.

    Args:
        X_train_full (pd.DataFrame): Train+validation features.
        y_continuous_full (pd.Series): Continuous target, row-aligned with X.
        y_class_full (pd.Series): Classification target, row-aligned with X.
        price_spread_full (pd.Series): ``|lambda_DA_hat - lambda_B_hat|``,
            row-aligned with X.
        deadband_quantile (float): Quantile of lowest-spread samples to drop;
            ``0.0`` drops nothing (but fallback filtering still applies).
        fallback_class: Class label to exclude before training, or ``None``.

    Returns:
        tuple[pd.DataFrame, pd.Series, int]: Filtered features, filtered
        continuous target, and the number of retained training samples.
    """
    if fallback_class is not None:
        keep = (y_class_full != fallback_class).to_numpy()
        X_train_full = X_train_full[keep]
        y_continuous_full = y_continuous_full[keep]
        price_spread_full = price_spread_full[keep]

    if deadband_quantile and deadband_quantile > 0.0:
        threshold = price_spread_full.quantile(deadband_quantile)
        keep = (price_spread_full > threshold).to_numpy()
        X_train_full = X_train_full[keep]
        y_continuous_full = y_continuous_full[keep]
        logger.info(
            f"Deadband (q={deadband_quantile:.2f}, threshold={threshold:.2f}): "
            f"{int(keep.sum())} / {len(keep)} training samples retained."
        )

    return X_train_full, y_continuous_full, len(X_train_full)


def compute_window_drift(
    cfg: DictConfig,
    window: dict,
    data_handler: DataHandler,
    n_projections: int,
    rng: np.random.Generator,
    n_features_selected: int | None = None,
    deadband_quantile: float | None = None,
) -> dict:
    """
    Compute feature and continuous-target Wasserstein drift for a single window.

    The reference distribution is the final training set (X_train + X_val), and
    the comparison distribution is the test set, mirroring how ``train.py`` fits
    the final model on train+validation before evaluating on test.

    When ``deadband_quantile`` is provided, the reference set is reduced to the
    samples the model is actually trained on for this window — the fallback class
    is dropped and the lowest-spread fraction is removed (see
    :func:`apply_training_deadband`) — so the drift measures the distance from the
    test set to the *used* training samples rather than the raw train+validation
    set. When it is ``None`` the full train+validation set is used (original
    behaviour).

    Args:
        cfg (DictConfig): Configuration object.
        window (dict): Rolling window with "train", "valid", "test" date ranges.
        data_handler (DataHandler): Transformed data handler for this window.
        n_projections (int): Number of projections for the sliced distance.
        rng (np.random.Generator): Seeded RNG for reproducible projections.
        n_features_selected (int | None): Number of features the drift is computed
            on. Falls back to the configured feature count when selection is off.
        deadband_quantile (float | None): Per-window deadband quantile to apply to
            the training set, or ``None`` to skip the reconstruction.

    Returns:
        dict: Window date bounds, ``n_features``, ``n_train_samples``,
        ``deadband_quantile``, ``feature_wasserstein``, ``target_wasserstein``,
        ``joint_wasserstein``, and the feature->target dependence on the training
        and test windows: ``train_feature_target_corr`` /
        ``test_feature_target_corr`` (mean absolute Pearson correlation, linear)
        and ``train_feature_target_mi`` / ``test_feature_target_mi`` (mean mutual
        information, also non-linear).
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

    X_train, yclass_train = split_features_target(cfg, data_train.data, sanitize_names=True)
    X_val, yclass_val = split_features_target(cfg, data_valid.data, sanitize_names=True)
    X_test, _ = split_features_target(cfg, data_test.data, sanitize_names=True)

    # Final training distribution = train + validation, exactly as the final model sees it.
    X_train_full = pd.concat([X_train, X_val])

    # Continuous target is pulled straight from the (engineered) data, not the
    # binary classification target used by the model. Re-index onto the feature
    # index so it can be filtered row-for-row alongside the features.
    y_train_full = pd.concat(
        [
            pd.Series(data_train.data[target_column].to_numpy(), index=X_train.index),
            pd.Series(data_valid.data[target_column].to_numpy(), index=X_val.index),
        ]
    )
    y_test = data_test.data[target_column]

    # Reduce the reference set to the rows the model is actually trained on for
    # this window (fallback-class filtering + deadband), mirroring train.train_batch.
    if deadband_quantile is not None:
        lambda_da = cfg.datasets.optimization.lambda_DA_hat
        lambda_b = cfg.datasets.optimization.lambda_B_hat
        price_spread_full = pd.concat(
            [
                pd.Series(
                    np.abs(data_train.data[lambda_da].to_numpy() - data_train.data[lambda_b].to_numpy()),
                    index=X_train.index,
                ),
                pd.Series(
                    np.abs(data_valid.data[lambda_da].to_numpy() - data_valid.data[lambda_b].to_numpy()),
                    index=X_val.index,
                ),
            ]
        )
        yclass_full = pd.concat([yclass_train, yclass_val])
        X_train_full, y_train_full, n_train_samples = apply_training_deadband(
            X_train_full,
            y_train_full,
            yclass_full,
            price_spread_full,
            deadband_quantile,
            cfg.datasets.training.fallback_class,
        )
    else:
        n_train_samples = len(X_train_full)

    ref_z, cmp_z = _standardize(X_train_full, X_test)

    method = cfg.experiments.experiment_parameters.get("feature_drift_method", "sliced")
    if method == "marginal":
        feature_distance = marginal_wasserstein_distance(ref_z, cmp_z)
    else:
        feature_distance = sliced_wasserstein_distance(ref_z, cmp_z, n_projections, rng)

    target_distance = target_drift_distance(y_train_full, y_test)

    # Joint feature+target drift: append the standardized target as an extra
    # dimension and measure the sliced distance over the augmented space.
    target_weight = cfg.experiments.experiment_parameters.get("joint_target_weight", 1.0)
    y_ref_z, y_cmp_z = _standardize_target_aligned(y_train_full, y_test)
    joint_distance = joint_wasserstein_distance(
        ref_z, cmp_z, y_ref_z, y_cmp_z, target_weight, n_projections, rng
    )

    # Feature->target dependence, measured on the same shared, non-zero-variance
    # feature set as the drift, separately for the training and test windows.
    # Pearson correlation captures linear dependence; mutual information also
    # captures non-linear dependence.
    common_columns = _common_feature_columns(X_train_full, X_test)
    X_train_common = X_train_full[common_columns]
    X_test_common = X_test[common_columns]
    seed = int(cfg.seed)
    train_corr = feature_target_correlation(X_train_common, y_train_full)
    test_corr = feature_target_correlation(X_test_common, y_test)
    train_mi = feature_target_mutual_information(X_train_common, y_train_full, seed)
    test_mi = feature_target_mutual_information(X_test_common, y_test, seed)

    if n_features_selected is None:
        n_features_selected = X_train_full.shape[1]

    logger.info(
        f"Drift | n_features={n_features_selected} | n_train_samples={n_train_samples} | "
        f"deadband_q={deadband_quantile} | feature_wasserstein ({method})={feature_distance:.4f} | "
        f"target_wasserstein={target_distance:.4f} | "
        f"joint_wasserstein (w={target_weight})={joint_distance:.4f} | "
        f"train_corr={train_corr:.4f} | test_corr={test_corr:.4f} | "
        f"train_mi={train_mi:.4f} | test_mi={test_mi:.4f}"
    )

    return {
        "train_start": train_start,
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "test_start": test_start,
        "test_end": test_end,
        "n_features": n_features_selected,
        "n_train_samples": n_train_samples,
        "deadband_quantile": deadband_quantile,
        "feature_wasserstein": feature_distance,
        "target_wasserstein": target_distance,
        "joint_wasserstein": joint_distance,
        "train_feature_target_corr": train_corr,
        "test_feature_target_corr": test_corr,
        "train_feature_target_mi": train_mi,
        "test_feature_target_mi": test_mi,
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

    # Per-window deadband quantiles from a previous backtest. When configured,
    # the drift is measured against the actually-used training samples instead of
    # the raw train+validation set.
    deadband_lookup = load_deadband_quantiles(cfg)

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

            # Look up this window's deadband quantile (matched by train/test start).
            deadband_quantile = None
            if deadband_lookup is not None:
                key = (_to_utc(window["train"][0]), _to_utc(window["test"][0]))
                deadband_quantile = deadband_lookup.get(key)
                if deadband_quantile is None:
                    logger.warning(
                        f"No deadband result for window train={key[0].date()} test={key[1].date()}; "
                        f"using 0.0 (fallback filtering still applied)."
                    )
                    deadband_quantile = 0.0

            rows.append(
                compute_window_drift(
                    cfg, window, data_handler, n_projections, rng, n_features_selected, deadband_quantile
                )
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
