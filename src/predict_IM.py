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
"""

import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    roc_auc_score,
    roc_curve
)
from tabulate import tabulate
from src.data_handler import DataHandler
from matplotlib.colors import TwoSlopeNorm


class ImbalancePredictor:
    """
    Base class for system imbalance prediction.
    Subclasses must implement `_load_model()`.
    """

    def __init__(self, **model_params):
        self.model_params = model_params
        self.model = self._load_model()
        self.pca = None

    def _load_model(self):
        """
        Subclasses must override this method to load the specific model.
        """
        raise NotImplementedError("Subclasses must implement _load_model()")

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        """
        Fit the model to the provided features and target.
        """
        if self.model is None:
            raise ValueError("Model is not loaded.")
        if features.empty or target.empty:
            raise ValueError("Features or target DataFrame is empty.")
        self.model.fit(features, target)

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        """
        Run predictions using the loaded model.
        """
        if self.model is None:
            raise ValueError("Model is not loaded.")
        if features.empty:
            raise ValueError("Features DataFrame is empty.")
        return pd.Series(
            self.model.predict_proba(features)[:, 1],
            index=features.index,
        )

    def predict(self, predict_proba: pd.Series, alpha: float) -> pd.Series:
        """
        Convert predicted probabilities to class labels.
        """
        return predict_proba.apply(
            lambda p: 1 if p >= alpha else (-1 if p <= 1 - alpha else 0)
        )

    def preprocess(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features before prediction.
        Standardize features and fill NaNs by interpolation.
        """
        if features.empty:
            raise ValueError("Features DataFrame is empty.")
        
        # Separate numeric and categorical features
        num_features = features.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        cat_features = features.select_dtypes(
            exclude=[np.number]
        ).columns.tolist()
        features_num = features[num_features].copy()
        features_cat = features[cat_features].copy()

        # Numeric: interpolate
        if not features_num.empty:
            features_num = features_num.interpolate(
                method="linear", limit_direction="both"
            )
            features_num = features_num.fillna(features_num.mean())

        # Categorical: fill with mode
        if not features_cat.empty:
            for col in features_cat.columns:
                mode_series = features_cat[col].mode()
                mode_val = (
                    mode_series.iloc[0] if not mode_series.empty else "missing"
                )
                features_cat[col] = features_cat[col].fillna(mode_val)

        # Standardize numeric features
        if not features_num.empty:
            scaler = StandardScaler()
            features_num = pd.DataFrame(
                scaler.fit_transform(features_num),
                columns=num_features,
                index=features.index
            )

        # One-hot encode categorical features
        if not features_cat.empty:
            encoder = OneHotEncoder(drop="first", sparse_output=False)
            encoded = encoder.fit_transform(features_cat)
            cat_columns = encoder.get_feature_names_out(cat_features)
            features_cat = pd.DataFrame(encoded, columns=cat_columns,
                                        index=features.index)

        # Concatenate processed numeric and categorical features
        if not features_num.empty and not features_cat.empty:
            features_processed = pd.concat([features_num, features_cat],
                                           axis=1)
        elif not features_num.empty:
            features_processed = features_num
        else:
            features_processed = features_cat

        return features_processed

    def apply_pca(self, features: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
        """
        Reduce dimensionality of standardized numeric features using PCA.
        
        Parameters
        ----------
        features : pd.DataFrame
            The preprocessed (standardized + encoded) features.
        n_components : int, default=2
            Number of principal components to retain.

        Returns
        -------
        pd.DataFrame
            Transformed feature set with principal components.
        """
        if features.empty:
            raise ValueError("Features DataFrame is empty.")

        # Fit PCA if not already fitted
        self.pca = PCA(n_components=n_components)
        reduced = self.pca.fit_transform(features)

        # Create a DataFrame with principal components
        pca_df = pd.DataFrame(
            reduced,
            columns=[f"PC{i+1}" for i in range(n_components)],
            index=features.index,
        )

        explained_var = self.pca.explained_variance_ratio_.sum()
        print(f"[INFO] PCA reduced features to {n_components} components "
              f"(explaining {explained_var:.2%} of variance)")

        return pca_df
    
    def evaluate_classifier(
        self, y_pred: pd.Series, y_test: pd.Series, y_proba: pd.Series,
        print_metrics: bool = True, alpha: float = 0.7
    ) -> Dict[str, Any]:
        """
        Evaluate classifier predictions and plot a confusion matrix.

        Args:
            y_pred: The predicted labels.
            y_test: The true labels.
            y_proba: The predicted probabilities.
            print_metrics: Whether to print the evaluation metrics.

        Raises:
            ValueError: If y_pred or y_test is None.

        Returns:
            A dictionary containing evaluation metrics and the confusion matrix.
        """
        if y_pred is None or y_test is None:
            raise ValueError("y_pred and y_test must not be None")

        # Ensure aligned indices
        y_test = y_test.loc[y_pred.index]

        # Keep only confident predictions (exclude 0)
        mask = y_pred != 0
        if not mask.any():
            raise ValueError(
                "All predictions are 0 (uncertain); nothing to evaluate."
            )
        y_pred_filtered = y_pred[mask]
        y_test_filtered = y_test[mask]

        # Map labels {-1,1} -> {0,1}
        y_pred_bin = y_pred_filtered.replace({-1: 0, 1: 1})
        y_test_bin = y_test_filtered.replace({-1: 0, 1: 1})

        # Metrics
        acc = accuracy_score(y_test_bin, y_pred_bin)
        f1 = f1_score(y_test_bin, y_pred_bin, average="weighted")
        if y_proba is not None:
            auc = roc_auc_score(y_test_bin, y_proba.loc[y_pred_bin.index])
        else:
            auc = None

        #########################################################################
        # Confusion matrix (with original label names for display)
        cm = confusion_matrix(y_test_bin, y_pred_bin, labels=[0, 1])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["-1", "1"]
        )
        disp.plot(cmap="Blues")
        plt.title(r"Confusion Matrix ($\alpha={}$)".format(alpha))
        plt.tight_layout()
        # Path to project root (parent of src)
        project_root = Path(__file__).resolve().parents[1]
        figures_dir = project_root / "figures"
        figures_dir.mkdir(exist_ok=True)
        plt.savefig(figures_dir / f"confusion_matrix_{alpha}.pdf")
        plt.close()

        #######################################################################
        # ROC curve
        if y_proba is not None:

            # Align scores with evaluated (non-zero) predictions
            y_scores = y_proba.loc[y_pred_bin.index]

            # Compute ROC
            fpr, tpr, _ = roc_curve(y_test_bin, y_scores)

            # Plot ROC
            plt.figure()
            plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
            plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(r"ROC Curve ($\alpha={}$)".format(alpha))
            plt.legend(loc="lower right")
            plt.tight_layout()

            # Save ROC figure next to confusion matrix
            plt.savefig(figures_dir / f"roc_curve_{alpha}.pdf")
            plt.close()


        #######################################################################
        # Compile metrics
        metrics = {
            "accuracy": acc,
            "f1": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "support": {
                "n_evaluated": int(len(y_test_bin)),
                "n_dropped_uncertain": int((~mask).sum()),
            },
        }

        if print_metrics:
            table = [
                ["Accuracy", f"{acc:.4f}"],
                ["F1 Score", f"{f1:.4f}"],
                ["ROC AUC", f"{auc:.4f}" if auc is not None else "N/A"],
            ]
            print(tabulate(table,
                           headers=["Metric", "Value"],
                           tablefmt="github"))
            print("Confusion Matrix:\n", pd.DataFrame(cm))
            print("Support:", metrics["support"])
        return metrics


class LogisticRegressionPredictor(ImbalancePredictor):
    """
    Imbalance predictor using Logistic Regression.
    """
    def _load_model(self):
        return LogisticRegression(**self.model_params)


class BartPredictor(ImbalancePredictor):
    """Imbalance predictor using BART (placeholder)."""

    def _load_model(self):  # pragma: no cover - placeholder
        return None


if __name__ == "__main__":
    # Import features
    imbalance_data = DataHandler("imbalance_data.parquet",
                                 "data/processed")

    #########################################################################
    # DK1 example
    #########################################################################

    imbalance_data.set_data(
        imbalance_data.data[
            (imbalance_data.data['datetime'] < '2025-01-03') &
            (imbalance_data.data['datetime'] >= '2023-01-03')
        ]
    )

    # Drop rows where ImbalanceDirection_DK1 is 0
    imbalance_data.set_data(
        imbalance_data.data[imbalance_data.data['ImbalanceDirection_DK1'] != 0]
    )

    imbalance_data = imbalance_data.transform_data(
            drop_missing_threshold=0.1)
    
    # _ = imbalance_data.validate_data(print_report=True)

    #########################################################################
    # Example of DK1 prediction
    #########################################################################

    # No NaNs or zeros in target
    dk1_data = imbalance_data.data[imbalance_data.data['ImbalanceDirection_DK1'].notnull()]
    dk1_data = dk1_data[dk1_data['ImbalanceDirection_DK1'] != 0]
    # Set index to datetime
    dk1_data = dk1_data.set_index('datetime')
    # Set features and target
    X = DataHandler()
    X.set_data(dk1_data)
    X = X.transform_data(drop_columns=['ImbalancePriceEUR_DK1',
                                       'ImbalanceMWh_DK1',
                                       'ImbalancePriceEUR_DK2',
                                       'ImbalanceMWh_DK2',
                                       'ImbalanceDirection_DK1',
                                       'ImbalanceDirection_DK2'])
    y = dk1_data['ImbalanceDirection_DK1']

    del dk1_data  # Free memory

    # predictor
    predictor = LogisticRegressionPredictor(penalty='l1',
                                            max_iter=10000,
                                            solver='saga')

    # Preprocess features
    X_processed = predictor.preprocess(X.data)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=42,
        stratify=y,
    )

    # Fit model
    predictor.fit(X_train, y_train)

    # Predict
    prediction_proba = predictor.predict_proba(X_test)

    alpha = 0.9
    predictions = predictor.predict(prediction_proba, alpha=alpha)

    # Evaluate and plot confusion matrix
    metrics = predictor.evaluate_classifier(predictions, y_test, prediction_proba, alpha=alpha)

    #########################################################################
    # PCA and Visualization of Decision Boundary
    #########################################################################
    # Apply PCA to test set
    X_test_pca = predictor.apply_pca(X_test, n_components=2)
    print(X_test_pca.head())

    # Visualize decision boundary
    plt.figure(figsize=(10, 6))
    # Align predictions with PCA dataframe index
    preds_aligned = predictions.loc[X_test_pca.index]
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    sc = plt.scatter(
        X_test_pca.iloc[:, 0],
        X_test_pca.iloc[:, 1],
        c=preds_aligned,
        cmap="coolwarm",
        norm=norm,
        alpha=0.7,
        edgecolor="black",
    )
    plt.title(r"Decision Boundary ($\alpha={}$)".format(alpha))
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    cbar = plt.colorbar(sc, ticks=[-1, 0, 1])
    cbar.set_label("Predicted Class")
    # Path to project root (parent of src)
    project_root = Path(__file__).resolve().parents[1]
    figures_dir = project_root / "figures"
    figures_dir.mkdir(exist_ok=True)
    plt.savefig(figures_dir / f"decision_boundary_{alpha}.pdf")
    plt.tight_layout()
    plt.close()