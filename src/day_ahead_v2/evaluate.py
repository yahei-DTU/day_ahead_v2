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
        auc = roc_auc_score(y.values, proba, average="macro", multi_class="ovr")
    except ValueError as e:
        auc = np.nan
        logger.warning(f"ROC-AUC could not be computed: {e}")
    # F1 score (macro)
    try:
        f1 = f1_score(y.values, preds, average="macro")
    except ValueError as e:
        f1 = np.nan
        logger.warning(f"F1 score could not be computed: {e}")

    metrics = {"accuracy": accuracy, "roc_auc": auc, "f1_score": f1}

    results_df = pd.DataFrame({
        "true_label": y,
        "predicted_label": preds,
    }, index=X.index)

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

def make_predictions(model, X: pd.DataFrame) -> pd.Series:
    """
    Make predictions using the trained model.

    Args:
        model: Trained model with predict method.
        X (pd.DataFrame): Feature data for prediction.

    Returns:
        pd.Series: Predicted class labels.
    """
    proba = model.predict_proba(X)  # shape: (n_samples, n_classes)
    preds = np.zeros_like(proba, dtype=int)
    n_samples = proba.shape[0]
    # Determine fallback index
    fallback_class = '0'  # Fallback to class '0' (Balanced)
    if fallback_class not in model.classes_:
        raise ValueError(f"Fallback class {fallback_class} not in model.classes_: {model.classes_}")
    fallback_idx = np.where(model.classes_ == fallback_class)[0][0]

    uncertain = np.zeros(n_samples, dtype=bool)
    for i in range(n_samples):
        argmax = proba[i].argmax()
        if proba[i, argmax] >= alpha:
            preds[i, argmax] = 1
        else:
            preds[i, fallback_idx] = 1  # safe fallback to '0'
            uncertain[i] = True
