import numpy as np
import pandas as pd
import linopy
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from typing import Dict, Tuple
from types import SimpleNamespace
import xarray as xr
import logging
logger = logging.getLogger(__name__)

def evaluate_classifier(model, X: pd.DataFrame = None, y: pd.Series = None, fallback_class: int = 2) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluate a multi-class model.

    Args:
        model: Trained model with predict_proba method.
        X (pd.DataFrame): Feature data for validation/testing.
        y (pd.Series): True labels (multi-class series).

    Returns:
        Dict[str, float]: Dictionary with accuracy, ROC-AUC (macro averaged), and F1 score (macro averaged).
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
    # F1 score (macro)
    try:
        f1 = f1_score(y_filtered.values, preds_filtered)
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

def compute_accuracy_f1(y_true: pd.Series, y_pred: pd.Series, y_score: pd.Series | None = None) -> Dict[str, float]:
    """
    Compute classification metrics: accuracy, F1 score (macro), and optionally ROC-AUC.

    Args:
        y_true (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.
        y_score (pd.Series, optional): Predicted probabilities for the positive class (for ROC-AUC).

    Returns:
        Dict[str, float]: Dictionary with accuracy, F1 score, and optionally ROC-AUC.
    """
    metrics = {}
    try:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
    except ValueError:
        metrics["accuracy"] = np.nan
        logger.warning("Accuracy could not be computed due to a ValueError.")

    try:
        metrics["f1_score"] = f1_score(y_true, y_pred, average="macro")
    except ValueError:
        metrics["f1_score"] = np.nan
        logger.warning("F1 score could not be computed due to a ValueError.")

    if y_score is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        except ValueError as e:
            metrics["roc_auc"] = np.nan
            logger.warning(f"ROC-AUC could not be computed: {e}")

    return metrics

def threshold_predictions(cfg, proba: np.ndarray, alpha: float) -> pd.DataFrame:
    """
    Make predictions using the trained model.

    Args:
        cfg: Configuration object.
        model: Trained model with predict method.
        proba (np.ndarray): Predicted probabilities for each class. Shape (n_samples, n_classes).
        alpha (float): Decision threshold for assigning class labels.

    Returns:
        pd.DataFrame: DataFrame with predicted class labels and uncertainty flag.
    """
    logger.info(f"Applying decision threshold alpha={alpha} for predictions.")
    preds = np.zeros_like(proba, dtype=int)
    n_samples = proba.shape[0]

    uncertain = np.zeros(n_samples, dtype=bool)
    for i in range(n_samples):
        argmax = proba[i].argmax()
        preds[i, argmax] = 1  # Assign predicted class based on highest probability
        if proba[i, argmax] < alpha:
            uncertain[i] = True

    preds = preds.argmax(axis=1)
    # if uncertain set to fallback class
    fallback_class = cfg.datasets.training.fallback_class
    preds[uncertain] = fallback_class
    logger.info(f"Total uncertain predictions: {uncertain.sum()} out of {n_samples} ({(uncertain.sum() / n_samples) * 100:.2f}%) -  (alpha={alpha})")

    # return df with preds and uncertain flag
    return pd.DataFrame({
        "thresholded_label": preds,
        "predicted_proba": proba.max(axis=1),
        "uncertain": uncertain
    }, index=range(len(preds)))

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
