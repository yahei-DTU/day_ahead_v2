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

import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import RFE
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
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib.lines import Line2D
import yaml
from src.data_handler import DataHandler


###########################################################################
# Plotting settings
x_length = 10
golden_ratio = (1 + 5 ** 0.5) / 2
plt.rcParams['figure.figsize'] = (x_length, x_length / golden_ratio)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

color_palette_1 = {
    'black': (0, 0, 0),
    'orange': (230, 159, 0),
    'sky_blue': (86, 180, 233),
    'bluish_green': (0, 158, 115),
    'yellow': (240, 228, 66),
    'blue': (0, 114, 178),
    'vermillion': (213, 94, 0),
    'reddish_purple': (204, 121, 167)
}
# Normalize to 0-1 range for matplotlib
color_palette_1 = {name: (r/255, g/255, b/255)
                   for name, (r, g, b) in color_palette_1.items()}
# Color palette 2 (colorblind-friendly) from S. Bolognani
color_palette_2 = {
    'blue': (68, 119, 170),
    'cyan': (102, 204, 238),
    'green': (34, 136, 51),
    'yellow': (204, 187, 68),
    'red': (238, 102, 119),
    'purple': (170, 51, 119),
    'grey': (187, 187, 187)
}
color_palette_2 = {name: (r/255, g/255, b/255)
                   for name, (r, g, b) in color_palette_2.items()}
############################################################################


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

    def rfe(self, features: pd.DataFrame,
            target: pd.Series, n_features_to_select: int) -> pd.DataFrame:
        """
        Perform Recursive Feature Elimination (RFE) to select top features.

        Args:
            features: DataFrame with the preprocessed features.
            target: Series with the target labels.
            n_features_to_select: Number of top features to select.

        Returns:
            DataFrame with selected top features.
        """
        if self.model is None:
            raise ValueError("Model is not loaded.")
        if features.empty or target.empty:
            raise ValueError("Features or target DataFrame is empty.")

        rfe = RFE(estimator=self.model,
                  n_features_to_select=n_features_to_select)
        rfe.fit(features, target)

        selected_features = features.columns[rfe.support_]
        print(f"[INFO] Selected top {n_features_to_select} features via RFE.")

        return features[selected_features]

    def apply_pca(self, features: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
        """
        Reduce dimensionality of standardized numeric features using PCA.
        
        Args:
            features: DataFrame with the preprocessed features.
            n_components: Number of principal components to retain.

        Returns:
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
    
    def plot_decision_boundary(self, X_pca: pd.DataFrame,
                               y_pred: pd.Series, y_true: pd.Series,
                               alpha: float, show_misclassified: bool,
                               y_magnitude: pd.Series = None) -> None:
        """
        Visualize decision boundary in PCA-reduced feature space.

        Args:
            X_pca: DataFrame with PCA-reduced features (2 components).
            y_pred: Series with predicted labels.
            y_true: Series with true labels.
            alpha: Confidence threshold used for predictions.
            show_misclassified: Whether to highlight misclassified points.
            y_magnitude: Series with imbalance magnitudes for marker sizing.
        """
        plt.style.use('seaborn-v0_8-colorblind')
        fig, ax = plt.subplots()
        # Align predictions with PCA dataframe index
        preds_aligned = y_pred.loc[X_pca.index]
        true_aligned = y_true.loc[X_pca.index]
        
        # Color normalization: -1 (shortage), 0 (uncertain), 1 (surplus)
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        
        # Use first 3 colors defined at module level
        cmap_colors = [color_palette_1["reddish_purple"],
                       color_palette_1["orange"],
                       color_palette_1["bluish_green"]]
        custom_cmap = ListedColormap(cmap_colors)
        
        # Identify mismatches (excluding uncertain predictions)
        mismatch = (preds_aligned != true_aligned) & (preds_aligned != 0)
        n_mismatch = int(mismatch.sum())
        n_total = int((preds_aligned != 0).sum())
        n_correct = n_total - n_mismatch
        accuracy = n_correct / n_total if n_total > 0 else 0
        
        # Separate correct and incorrect predictions
        correct_mask = ~mismatch
        
        # Calculate marker sizes based on magnitude if provided
        if y_magnitude is not None:
            # Align magnitude with predictions
            magnitude_aligned = y_magnitude.loc[X_pca.index].abs()
            # Scale to reasonable marker size range (10 to 200)
            min_size, max_size = 10, 200
            mag_min, mag_max = magnitude_aligned.min(), magnitude_aligned.max()
            if mag_max > mag_min:
                sizes = min_size + (magnitude_aligned - mag_min) / \
                        (mag_max - mag_min) * (max_size - min_size)
            else:
                sizes = pd.Series(50, index=magnitude_aligned.index)
        else:
            sizes = None
        
        # Plot based on show_misclassified flag
        if show_misclassified:
            # Plot correct predictions
            if correct_mask.any():
                sc_correct = ax.scatter(
                    X_pca.loc[correct_mask].iloc[:, 0],
                    X_pca.loc[correct_mask].iloc[:, 1],
                    c=preds_aligned[correct_mask],
                    cmap=custom_cmap,
                    norm=norm,
                    alpha=0.6,
                    s=sizes[correct_mask] if sizes is not None else 50,
                    edgecolors="white",
                    linewidths=0.5,
                    label="Correct",
                )
            
            # Plot mismatches with prominent marker
            if mismatch.any():
                sc_mismatch = ax.scatter(
                    X_pca.loc[mismatch].iloc[:, 0],
                    X_pca.loc[mismatch].iloc[:, 1],
                    c=preds_aligned[mismatch],
                    cmap=custom_cmap,
                    norm=norm,
                    alpha=0.9,
                    s=sizes[mismatch] if sizes is not None else 80,
                    edgecolors="black",
                    linewidths=1,
                    marker="X",
                    label=f"Misclassified ({n_mismatch})",
                )
        else:
            # Plot all predictions together without distinguishing misclassified
            sc_all = ax.scatter(
                X_pca.iloc[:, 0],
                X_pca.iloc[:, 1],
                c=preds_aligned,
                cmap=custom_cmap,
                norm=norm,
                alpha=0.7,
                s=sizes if sizes is not None else 50,
                edgecolors="white",
                linewidths=0.5,
            )
        
        # Title with accuracy
        title = (
            r"Decision Boundary in PCA Space "
            rf"($\alpha={alpha:.2f}$, Accuracy={accuracy:.1%})"
        )
        ax.set_title(title, pad=15)
        ax.set_xlabel("First Principal Component")
        ax.set_ylabel("Second Principal Component")
                
        # Colorbar with custom labels
        if show_misclassified:
            scatter_ref = sc_correct if correct_mask.any() else sc_mismatch
        else:
            scatter_ref = sc_all
            
        cbar = plt.colorbar(
            scatter_ref,
            ax=ax,
            ticks=[-1, 0, 1]
        )
        cbar.ax.set_yticklabels(
            ["Deficit (-1)", "Uncertain (0)", "Surplus (1)"]
        )
        
        # Custom legend - only show if misclassified points are displayed
        if show_misclassified:
            legend_handles = [
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='gray',
                       markersize=8, linestyle='None',
                       label=f'Correctly Classified ({n_correct})'),
                Line2D([0], [0], marker='X', color='w',
                       markerfacecolor='gray',
                       markeredgecolor='black', markersize=8,
                       linestyle='None',
                       label=f'Misclassified ({n_mismatch})')
            ]
            ax.legend(handles=legend_handles, loc="best",
                      framealpha=0.9, fontsize=10)
        
        # Save figure
        project_root = Path(__file__).resolve().parents[1]
        figures_dir = project_root / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        plt.tight_layout()
        plt.savefig(
            figures_dir / f"decision_boundary_{alpha}.pdf",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
    
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
    # Load configuration
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config" / "config_dev.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

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
    dk1_data = imbalance_data.data[
        imbalance_data.data['ImbalanceDirection_DK1'].notnull()]
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
    y_label = dk1_data['ImbalanceDirection_DK1']
    y_magnitude = dk1_data['ImbalanceMWh_DK1']

    del dk1_data  # Free memory

    # predictor
    predictor = LogisticRegressionPredictor(penalty='l1',
                                            max_iter=10000,
                                            solver='saga',
                                            class_weight='balanced')

    # Preprocess features
    print("[INFO] Preprocessing features...")
    X_processed = predictor.preprocess(X.data)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y_label,
        test_size=0.5,
        shuffle=False,
        random_state=42,
        # stratify=y_label,
    )

    # Optional: Feature selection via RFE
    if config.get('feature_selection', {}).get('enabled', False):
        method = config['feature_selection'].get('method', 'RFE')
        n_features = config['feature_selection'].get('n_features_to_select', 20)
        if method == 'RFE':
            print(f"[INFO] Performing feature selection using RFE...")
            X_processed = predictor.rfe(
                X_train, y_train, n_features_to_select=n_features
            )
        else:
            print(f"[WARNING] Feature selection method '{method}' not recognized. Skipping feature selection.")

    # Split magnitude with same indices
    y_magnitude_train = y_magnitude.loc[y_train.index]
    y_magnitude_test = y_magnitude.loc[y_test.index]

    # Fit model
    print("[INFO] Fitting model...")    
    predictor.fit(X_train, y_train)

    # Predict
    print("[INFO] Making predictions...")
    prediction_proba = predictor.predict_proba(X_test)

    # Convert probabilities to class labels
    alpha = config['parameters']['alpha']
    print(f"[INFO] Using decision threshold alpha = {alpha}")
    predictions = predictor.predict(prediction_proba, alpha=alpha)

    # Evaluate and plot confusion matrix
    metrics = predictor.evaluate_classifier(predictions, y_test,
                                            prediction_proba, alpha=alpha)

    #########################################################################
    # PCA and Visualization of Decision Boundary
    #########################################################################
    # Apply PCA to test set
    X_test_pca = predictor.apply_pca(X_test, n_components=2)

    # Plot decision boundary
    predictor.plot_decision_boundary(X_test_pca, predictions,
                                     y_test,
                                     alpha=alpha,
                                     show_misclassified=False,
                                     y_magnitude=y_magnitude_test)
