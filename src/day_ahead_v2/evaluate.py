import numpy as np
import pandas as pd
import linopy
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score
from typing import Dict, Tuple
from types import SimpleNamespace
import xarray as xr
import logging
logger = logging.getLogger(__name__)

def evaluate_classifier(model, X: pd.DataFrame = None, y: pd.Series = None, fallback_class: int = 2) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluate a classifier on the binary (class 0 vs class 1) task.

    Samples whose true label is the fallback/abstain class are excluded before
    computing any metric, so all metrics describe the binary task.

    Args:
        model: Trained model with predict and predict_proba methods.
        X (pd.DataFrame): Feature data for validation/testing.
        y (pd.Series): True labels (may include the fallback class).
        fallback_class (int): Label of the fallback/abstain class to exclude from metrics.

    Returns:
        Dict[str, float]: Dictionary with accuracy, ROC-AUC (binary, on P(class 1)),
            and F1 score (macro averaged over classes 0 and 1).
    """
    # Check types of X and y
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a pd.DataFrame, got {type(X)}")
    if not isinstance(y, pd.Series):
        raise TypeError(f"y must be a pd.Series, got {type(y)}")


    preds = model.predict(X)  # shape: (n_samples,)
    proba = model.predict_proba(X)  # shape: (n_samples, n_classes)

    # Filter fallback class predictions for metrics calculation
    mask = y.values != fallback_class
    if mask.sum() == 0:
        logger.warning("All samples belong to the fallback class. Metrics will be set to NaN.")
        return {"accuracy": np.nan, "roc_auc": np.nan, "f1_score": np.nan}, pd.DataFrame()
    y_filtered = y[mask]
    preds_filtered = preds[mask]
    proba_filtered = proba[mask]

    # Accuracy
    try:
        accuracy = accuracy_score(y_filtered.values, preds_filtered)
    except ValueError as e:
        accuracy = np.nan
        logger.warning(f"Accuracy could not be computed: {e}")

    # ROC-AUC
    try:
        auc = roc_auc_score(y_filtered.values, proba_filtered[:, 1])
    except ValueError as e:
        auc = np.nan
        logger.warning(f"ROC-AUC could not be computed: {e}")
    # F1 score (macro over the two real classes)
    try:
        f1 = f1_score(y_filtered.values, preds_filtered, average="macro", labels=[0, 1])
    except ValueError as e:
        f1 = np.nan
        logger.warning(f"F1 score could not be computed: {e}")

    metrics = {"accuracy": accuracy, "roc_auc": auc, "f1_score": f1}

    results_df = pd.DataFrame({
        "true_label": y,
        "predicted_label": preds,
    }, index=X.index)

    for i, class_label in enumerate(model.classes_):
        results_df[f"proba_class_{class_label}"] = proba[:, i]

    return metrics, results_df

def compute_accuracy_f1(y_true: pd.Series, y_pred: pd.Series, y_score: pd.Series | None = None, fallback_class: int = 2) -> Dict[str, float]:
    """
    Compute classification metrics: accuracy, F1 score (macro), and optionally ROC-AUC.

    Args:
        y_true (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.
        y_score (pd.Series, optional): Predicted probabilities for the positive class (for ROC-AUC).
        fallback_class (int): Label of the fallback/abstain class. Samples whose true
            label is this class are excluded from every metric, since they all describe
            the binary (class 0 vs class 1) task.

    Returns:
        Dict[str, float]: Dictionary with accuracy, F1 score, and optionally ROC-AUC.
    """
    metrics = {}

    # All metrics describe the binary (class 0 vs class 1) task, so drop
    # fallback/abstain-class samples up front to keep them mutually consistent
    # (and to keep ROC-AUC from ever seeing a multiclass target).
    binary_mask = np.asarray(y_true) != fallback_class
    y_true_bin = np.asarray(y_true)[binary_mask]
    y_pred_bin = np.asarray(y_pred)[binary_mask]

    try:
        metrics["accuracy"] = accuracy_score(y_true_bin, y_pred_bin)
    except ValueError:
        metrics["accuracy"] = np.nan
        logger.warning("Accuracy could not be computed due to a ValueError.")

    try:
        metrics["f1_score"] = f1_score(y_true_bin, y_pred_bin, average="macro", labels=[0, 1])
    except ValueError:
        metrics["f1_score"] = np.nan
        logger.warning("F1 score could not be computed due to a ValueError.")

    try:
        per_class_recall = recall_score(y_true_bin, y_pred_bin, average=None, labels=[0, 1], zero_division=np.nan)
        metrics["accuracy_class_0"] = per_class_recall[0]
        metrics["accuracy_class_1"] = per_class_recall[1]
    except ValueError:
        metrics["accuracy_class_0"] = np.nan
        metrics["accuracy_class_1"] = np.nan
        logger.warning("Per-class accuracy could not be computed due to a ValueError.")

    if y_score is not None:
        y_score_bin = np.asarray(y_score)[binary_mask]
        try:
            metrics["roc_auc"] = roc_auc_score(y_true_bin, y_score_bin)
        except ValueError as e:
            metrics["roc_auc"] = np.nan
            logger.warning(f"ROC-AUC could not be computed: {e}")

    return metrics

def threshold_predictions(cfg, proba: np.ndarray, alpha_0: float, alpha_1: float) -> pd.DataFrame:
    """
    Make predictions using asymmetric decision thresholds.

    For binary probabilities (column 1 = P(class 1)):
        - predict class 1 if P(class 1) >= alpha_1,
        - predict class 0 if P(class 1) <= alpha_0,
        - otherwise mark the sample uncertain (-> fallback_class).

    The symmetric single-threshold case is recovered with
    alpha_0 = 1 - alpha and alpha_1 = alpha.

    Args:
        cfg: Configuration object.
        proba (np.ndarray): Predicted probabilities. Shape (n_samples, 2), columns
            ordered by ascending class label so column 1 is P(class 1).
        alpha_0 (float): Lower threshold in [0, 0.5]; predict class 0 if P(class 1) <= alpha_0.
        alpha_1 (float): Upper threshold in [0.5, 1.0]; predict class 1 if P(class 1) >= alpha_1.

    Returns:
        pd.DataFrame: DataFrame with predicted class labels and uncertainty flag.
    """
    logger.info(f"Applying asymmetric decision thresholds alpha_0={alpha_0}, alpha_1={alpha_1} for predictions.")
    if proba.shape[1] != 2:
        raise ValueError(
            f"Asymmetric thresholding expects binary probabilities (2 columns), got {proba.shape[1]}."
        )
    n_samples = proba.shape[0]
    p1 = proba[:, 1]

    fallback_class = cfg.datasets.training.fallback_class
    preds = np.full(n_samples, fallback_class, dtype=int)
    preds[p1 >= alpha_1] = 1
    preds[p1 <= alpha_0] = 0
    uncertain = (p1 > alpha_0) & (p1 < alpha_1)
    logger.info(f"Total uncertain predictions: {uncertain.sum()} out of {n_samples} ({(uncertain.sum() / n_samples) * 100:.2f}%) -  (alpha_0={alpha_0}, alpha_1={alpha_1})")

    # return df with preds and uncertain flag
    return pd.DataFrame({
        "thresholded_label": preds,
        "predicted_proba": proba.max(axis=1),
        "uncertain": uncertain
    }, index=range(n_samples))

def calculate_hydrogen_balancing_bids(cfg, p_DA: pd.Series, lambda_B_hat: pd.Series, P_W_hat: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    parameters = SimpleNamespace()
    constants = SimpleNamespace()
    results = SimpleNamespace()

    assert lambda_B_hat.index.equals(p_DA.index)
    assert P_W_hat.index.equals(p_DA.index)

    # Constants
    constants.T = lambda_B_hat.index
    constants.D = lambda_B_hat.index.floor("D").to_numpy()
    constants.DAY_INDEX, constants.DAY_VALUES = pd.factorize(
        lambda_B_hat.index.normalize()
    )
    constants.DAYS = range(len(constants.DAY_VALUES))
    constants.P_W_BAR = cfg.experiments.optimization_parameters.wind_capacity
    constants.P_H_BAR = cfg.experiments.optimization_parameters.electrolyzer_capacity
    constants.H2_PRICE = cfg.experiments.optimization_parameters.hydrogen_price
    constants.ELECTROLYZER_LOAD_MIN = cfg.experiments.optimization_parameters.electrolyzer_load_min*cfg.experiments.optimization_parameters.electrolyzer_capacity
    constants.A = cfg.experiments.optimization_parameters.a # List of linear segments for hydrogen production function (kg/MWh)
    constants.B = cfg.experiments.optimization_parameters.b # List of intercepts for hydrogen production function (kg)
    constants.SEGMENTS = range(len(constants.A))
    constants.H2_MIN = cfg.experiments.optimization_parameters.minimum_daily_hydrogen

    # Parameters
    parameters.p_DA = xr.DataArray(
        p_DA,
        dims=["datetime"],
        coords={
            "datetime": constants.T,
            "day": ("datetime", constants.D)
            }
    )
    parameters.lambda_B_hat = xr.DataArray(
        lambda_B_hat,
        dims=["datetime"],
        coords={
            "datetime": constants.T,
            "day": ("datetime", constants.D)
        }
    )
    parameters.P_W_hat = xr.DataArray(
        P_W_hat,
        dims=["datetime"],
        coords={
            "datetime": constants.T,
            "day": ("datetime", constants.D)
        }
    )
    parameters.A = xr.DataArray(
        constants.A,
        dims=["segment"],
        coords={"segment": constants.SEGMENTS}
    )
    parameters.B = xr.DataArray(
        constants.B,
        dims=["segment"],
        coords={"segment": constants.SEGMENTS}
    )

    # Create optimization model
    model = linopy.Model()

    # Add decision variables
    model.add_variables(
        dims=["datetime"],
        coords={"datetime": constants.T,
                "day": ("datetime", constants.D)},
        name="p_B"
    )
    model.add_variables(
        lower=constants.ELECTROLYZER_LOAD_MIN,
        upper=constants.P_H_BAR,
        dims=["datetime"],
        coords={"datetime": constants.T,
                "day": ("datetime", constants.D)},
        name="p_H"
    )
    model.add_variables(
        lower=0,
        dims=["datetime"],
        coords={"datetime": constants.T,
                "day": ("datetime", constants.D)},
        name="h"
    )

    # Add objective
    model.add_objective(
        (model.variables["p_B"] * parameters.lambda_B_hat).sum()
        + (model.variables["h"] * constants.H2_PRICE).sum(),
        sense="max"
    )

    # Add constraints
    model.add_constraints(
        model.variables["p_B"] + model.variables["p_H"] == parameters.P_W_hat - parameters.p_DA,
        name="power_balance"
    )
    model.add_constraints(
        model.variables["h"]
        <= parameters.A * model.variables["p_H"]
        + parameters.B,
        name="hydrogen_production"
    )
    for d in constants.DAYS:
        mask = constants.DAY_INDEX == d
        model.add_constraints(
            model.variables["h"][mask].sum() >= constants.H2_MIN,
            name=f"daily_hydrogen_{d}"
        )

    # Log model properties for debugging
    logger.debug("Built model with following properties:")
    # Variables
    for var_name in model.variables:
        var = model.variables[var_name]
        logger.debug(f"\tVariable: {var_name}, dims={var.dims}, shape={var.shape}")
    # Constraints
    for con_name, con in model.constraints.items():
        # Use .sizes if available
        sizes = getattr(con, "sizes", None)
        if sizes is not None:
            dims_str = ", ".join(f"{k}:{v}" for k, v in sizes.items())
        else:
            dims_str = "unknown"
        logger.debug(f"\tConstraint: {con_name}, dims={dims_str}")

    # Solve model
    model.solve(solver_name="highs")

    # Save results
    results.status = model.status
    if results.status != "ok":
        logger.error(f"Optimization of hydrogen and balancing bids did not solve successfully. Status: {results.status}")
    logger.info(f"Optimization status: {results.status}")
    results.p_B = model.variables["p_B"].solution.to_pandas()
    results.p_H = model.variables["p_H"].solution.to_pandas()
    results.h = model.variables["h"].solution.to_pandas()

    return results.p_B, results.p_H, results.h


def calculate_profit(p_DA: pd.Series, h: pd.Series, p_B: pd.Series, lambda_DA_hat: pd.Series, lambda_H: pd.Series | float, lambda_B_hat: pd.Series) -> pd.Series:
    """
    Calculate profit based on realized day-ahead and imbalance prices, realized wind production.
    """
    profit = p_DA * lambda_DA_hat + h * lambda_H + p_B * lambda_B_hat
    # Check for NaN values in profit
    if profit.isna().any():
        logger.error(f"{profit.isna().sum()} profit values are NaN. Check calculate_profit method.")
        profit.fillna(0, inplace=True)
    return profit

def cvar_profit(profit: pd.Series, alpha: float = 0.05) -> float:
    """
    Calculate CVaR (Conditional Value at Risk) of the profit distribution at the given alpha level.
    CVaR is the expected profit in the worst alpha% of cases.
    """
    if profit.isna().any():
        logger.error("Profit series contains NaN values. Check cvar_profit method.")
        profit = profit.dropna()
    var_threshold = np.percentile(profit, alpha * 100)
    cvar = profit[profit <= var_threshold].mean()
    return cvar

def mean_profit(profit: pd.Series) -> float:
    """
    Calculate the mean profit.
    """
    if profit.isna().any():
        logger.error("Profit series contains NaN values. Check mean_profit method.")
        profit = profit.dropna()
    return profit.mean()
