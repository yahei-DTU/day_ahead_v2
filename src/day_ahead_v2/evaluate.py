import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def evaluate_classifier(model, X: pd.DataFrame = None, y: pd.Series = None) -> Tuple[Dict[str, float], pd.DataFrame]:
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

    # Accuracy
    try:
        accuracy = accuracy_score(y.values, preds)
    except ValueError as e:
        accuracy = np.nan
        logger.warning(f"Accuracy could not be computed: {e}")

    # ROC-AUC
    try:
        auc = roc_auc_score(y.values, proba[:, 1])
    except ValueError as e:
        auc = np.nan
        logger.warning(f"ROC-AUC could not be computed: {e}")
    # F1 score (macro)
    try:
        f1 = f1_score(y.values, preds)
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

def compute_accuracy_f1(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Compute classification metrics: accuracy, and F1 score (macro).

    Args:
        y_true (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.

    Returns:
        Dict[str, float]: Dictionary with accuracy and F1 score.
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
        "uncertain": uncertain
    }, index=range(len(preds)))

def calculate_bids(cfg, data_df: pd.DataFrame, features: pd.DataFrame, preds_df: pd.DataFrame, optimizers: dict) -> pd.Series:
    """Calculate the bids for best optimization parameters"""
    lambda_max = 1000 # max lambda for bidding curve
    lambda_step = 2  # step size of bids
    P_W_tilde = data_df[cfg.datasets.optimization.P_W_tilde]
    if P_W_tilde.isna().any():
        logger.error(f"{P_W_tilde.isna().sum()} values in P_W_tilde are NaN. Check data preprocessing.")
        # P_W_tilde.fillna(0, inplace=True)
    P_W_hat = data_df[cfg.datasets.optimization.P_W_hat]
    lambda_DA_hat = data_df[cfg.datasets.optimization.lambda_DA_hat]
    lambda_B_hat = data_df[cfg.datasets.optimization.lambda_B_hat]
    features = features.copy()
    features["intercept"] = 1.0
    model_mapping = cfg.datasets.optimization.model_mapping
    logger.info(f"optimizers: {optimizers}")
    DA_bids = pd.Series(index=data_df.index, dtype=float)
    IM_bids = pd.Series(index=data_df.index, dtype=float)
    if preds_df["uncertain"].dtype != bool:
        preds_df["uncertain"] = preds_df["uncertain"].astype(bool)
    for label in np.unique(preds_df["thresholded_label"]):
        label = int(label)  # Ensure label is int for mapping
        model_name = model_mapping[label]
        optimizer = optimizers[model_name]
        if label == cfg.datasets.training.fallback_class:
            preds_label = preds_df[(preds_df["thresholded_label"] == label) | (preds_df["uncertain"])]
        else:
            preds_label = preds_df[
                (preds_df["thresholded_label"] == label) &
                (~preds_df["uncertain"])
            ]
        if not preds_label.empty:
            bid_lambda_DA = np.arange(0,lambda_max,lambda_step) # Discretization for bidding curve, actual bids follow from curve
            bid_p_DA = np.zeros((len(preds_label),len(bid_lambda_DA)))
            bid_p_DA = pd.DataFrame(bid_p_DA, index=preds_label.index, columns=bid_lambda_DA)
            if label != cfg.datasets.training.fallback_class:
                q = optimizer.results.q.copy()
                q = q.drop('lambda_DA_hat')
            for t in preds_label.index:
                # Check for negative prices and handle accordingly
                if lambda_DA_hat.loc[t] < 0:
                    logger.warning(f"Negative price detected at time {t}: lambda_DA_hat={lambda_DA_hat.loc[t]}. Setting DA bid to 0.")
                    DA_bids.loc[t] = 0.0
                    IM_bids.loc[t] = P_W_hat.loc[t] - DA_bids.loc[t]
                    if lambda_B_hat.loc[t] < 0:
                        logger.warning(f"Negative imbalance price detected at time {t}: lambda_B_hat={lambda_B_hat.loc[t]}. Setting IM bid to 0.")
                        IM_bids.loc[t] = 0.0
                    continue
                # Loop through discretized steps of bidding curve to find optimal bid
                if label == cfg.datasets.training.fallback_class:
                    DA_bids.loc[t] = P_W_tilde.loc[t]
                    IM_bids.loc[t] = P_W_hat.loc[t] - DA_bids.loc[t]
                    continue
                for k in bid_p_DA.columns:
                    a_DA = optimizer.results.q["lambda_DA_hat"]
                    b_DA = features.loc[t,:] @ q
                    if bid_lambda_DA[k] > lambda_DA_hat.loc[t]:
                        DA_bids.loc[t] = bid_p_DA.loc[t,k-lambda_step]
                        IM_bids.loc[t] = P_W_hat.loc[t] - DA_bids.loc[t]
                        break
                    bid_p_DA.loc[t,k] = P_W_tilde.loc[t] + np.maximum(np.minimum(a_DA * bid_lambda_DA[k] + b_DA,cfg.experiments.optimization_parameters.wind_capacity),-cfg.experiments.optimization_parameters.electrolyzer_capacity)

    # Check if any DA_bids are still NaN (e.g., due to all predictions being uncertain or fallback class)
    if DA_bids.isna().any():
        logger.error(f"{DA_bids.isna().sum()} DA bids are NaN. Check calculate_bids method.")
        DA_bids.fillna(P_W_tilde, inplace=True)
    if IM_bids.isna().any():
        logger.error(f"{IM_bids.isna().sum()} IM bids are NaN. Check calculate_bids method.")
        IM_bids.fillna(0, inplace=True)

    return DA_bids, IM_bids


def calculate_profit(DA_bids: pd.Series, IM_bids: pd.Series, lambda_DA_hat: pd.Series, lambda_B_hat: pd.Series) -> pd.Series:
    """
    Calculate profit based on realized day-ahead and imbalance prices, realized wind production.
    """
    profit = DA_bids * lambda_DA_hat + IM_bids * lambda_B_hat
    # Check for NaN values in profit
    if profit.isna().any():
        logger.error(f"{profit.isna().sum()} profit values are NaN. Check calculate_profit method.")
        profit.fillna(0, inplace=True)
    return profit
