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
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from src.data_handler import DataHandler

class ImbalancePredictor:
    """
    Base class for system imbalance prediction.
    Subclasses must implement `_load_model()`.
    """

    def __init__(self, **model_params):
        self.model_params = model_params
        self.model = self._load_model()

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

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Run predictions using the loaded model.
        """
        if self.model is None:
            raise ValueError("Model is not loaded.")
        if features.empty:
            raise ValueError("Features DataFrame is empty.")
        return pd.Series(self.model.predict(features), index=features.index)
    
    def fit_predict(self, features: pd.DataFrame, target: pd.Series) -> pd.Series:
        """
        Fit the model and run predictions.
        """
        if self.model is None:
            raise ValueError("Model is not loaded.")
        if features.empty or target.empty:
            raise ValueError("Features or target DataFrame is empty.")
        self.model.fit(features, target)
        return pd.Series(self.model.predict(features), index=features.index)

    def preprocess(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features before prediction.
        Standardize features and fill NaNs by interpolation.
        """
        if features.empty:
            raise ValueError("Features DataFrame is empty.")
        
        # Separate numeric and categorical features
        num_features = features.select_dtypes(include=[np.number]).columns.tolist()
        cat_features = features.select_dtypes(exclude=[np.number]).columns.tolist()
        features_num = features[num_features].copy()
        features_cat = features[cat_features].copy()

        # Numeric: interpolate
        if not features_num.empty:
            features_num = features_num.interpolate(method="linear", limit_direction="both")
            features_num = features_num.fillna(features_num.mean())

        # Categorical: fill with mode
        if not features_cat.empty:
            for col in features_cat.columns:
                mode_val = features_cat[col].mode().iloc[0] if not features_cat[col].mode().empty else "missing"
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

    def evaluate_classifier(self, X_test, y_test):
        """Evaluate a fitted classifier on test data.

        Parameters
        ----------
        X_test : pd.DataFrame or np.ndarray
            Test feature matrix.
        y_test : pd.Series or np.ndarray
            True labels.

        Returns
        -------
        metrics : Dict[str, Any]
            Dictionary containing accuracy, classification_report (string),
            confusion_matrix (ndarray), optional roc_auc (float), and y_pred.

        Notes
        -----
        - Handles binary and multiclass problems.
        - Tries to extract class probabilities using predict_proba; if not
          available, falls back to decision_function where possible.
        - For multiclass ROC AUC, computes macro one-vs-rest if probabilities
          are available.
        """
        if self.model is None:
            raise ValueError("Model not loaded.")
        if X_test is None or y_test is None:
            raise ValueError("X_test and y_test must not be None.")
        if len(y_test) == 0:
            raise ValueError("y_test is empty.")

        y_pred = self.model.predict(X_test)

        # Core metrics
        acc = accuracy_score(y_test, y_pred)
        report_text = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(f"Accuracy: {acc:.4f}\n")
        print("Classification Report:")
        print(report_text)

        # Confusion matrix plot
        try:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues")
            plt.title("Confusion Matrix")
            plt.show()
        except Exception as e:  # noqa: BLE001
            print(f"Warning: could not plot confusion matrix ({e}).")

        unique_classes = np.unique(y_test)
        n_classes = len(unique_classes)
        metrics: Dict[str, Any] = {
            "accuracy": acc,
            "classification_report": report_text,
            "confusion_matrix": cm,
            "y_pred": y_pred,
        }

        # Probability / score extraction
        y_score = None
        proba_matrix = None
        if hasattr(self.model, "predict_proba"):
            try:
                proba_matrix = self.model.predict_proba(X_test)
            except Exception as e:  # noqa: BLE001
                print(f"predict_proba failed: {e}")
        if proba_matrix is None and hasattr(self.model, "decision_function"):
            try:
                decision = self.model.decision_function(X_test)
                # decision_function shape handling
                if decision.ndim == 1:
                    # Binary case; map to probability-like scores via sigmoid
                    y_score = 1 / (1 + np.exp(-decision))
                else:
                    # Multiclass raw scores -> softmax approximation
                    expd = np.exp(
                        decision - decision.max(axis=1, keepdims=True)
                    )
                    proba_matrix = expd / expd.sum(axis=1, keepdims=True)
            except Exception as e:  # noqa: BLE001
                print(f"decision_function failed: {e}")

        # Binary metrics (ROC / PR) or multiclass ROC AUC
        try:
            if n_classes == 2:
                if y_score is None:
                    # Derive positive class proba from proba_matrix
                    if proba_matrix is not None:
                        classes = getattr(
                            self.model, "classes_", unique_classes
                        )
                        # Choose max label as positive (e.g. {-1,1})
                        classes_list = list(classes)
                        positive_class = max(classes_list)
                        pos_index = classes_list.index(positive_class)
                        y_score = proba_matrix[:, pos_index]
                if y_score is not None:
                    auc = roc_auc_score(y_test, y_score)
                    metrics["roc_auc"] = auc
                    print(f"ROC AUC: {auc:.4f}\n")
                    try:
                        RocCurveDisplay.from_predictions(y_test, y_score)
                        plt.title("ROC Curve")
                        plt.show()
                    except Exception as e:  # noqa: BLE001
                        print(f"Cannot plot ROC curve: {e}")
                    try:
                        PrecisionRecallDisplay.from_predictions(
                            y_test, y_score
                        )
                        plt.title("Precision-Recall Curve")
                        plt.show()
                    except Exception as e:  # noqa: BLE001
                        print(f"Cannot plot Precision-Recall curve: {e}")
            elif n_classes > 2 and proba_matrix is not None:
                try:
                    # macro OVR ROC AUC
                    multi_auc = roc_auc_score(
                        y_test,
                        proba_matrix,
                        multi_class="ovr",
                        average="macro",
                    )
                    metrics["roc_auc_macro_ovr"] = multi_auc
                    print(f"Macro OVR ROC AUC: {multi_auc:.4f}\n")
                except Exception as e:  # noqa: BLE001
                    print(f"Multiclass ROC AUC failed: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"AUC computation failed: {e}")

        # Always expose y_score (may be None if not derivable)
        metrics["y_score"] = y_score
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

    # Import weather coordinates
    # coordinates = DataHandler("dk_mesh_points.csv", "data/processed")
    # latitudes = coordinates.data['latitude'].tolist()
    # longitudes = coordinates.data['longitude'].tolist()

    # weather_data = OpenMeteoHandler(latitude=latitudes,
    #                                 longitude=longitudes,
    #                                 hourly=["temperature_2m",
    #                                         "relative_humidity_2m",
    #                                         "cloud_cover",
    #                                         "wind_speed_10m",
    #                                         "wind_speed_80m",
    #                                         "wind_speed_120m",
    #                                         "wind_speed_180m",
    #                                         "wind_direction_10m",
    #                                         "wind_direction_80m",
    #                                         "wind_direction_120m",
    #                                         "wind_direction_180m",
    #                                         "surface_pressure",
    #                                         "visibility"],
    #                                 models="dmi_harmonie_arome_europe",
    #                                 start_date="2023-01-01",
    #                                 end_date="2024-12-31"
    #                                 )
    # _ = weather_data.validate_data(print_report=True)
    # print(weather_data.head())
    # print(weather_data.tail())

    # DK2 example
    imbalance_data.set_data(
        imbalance_data.data[
            (imbalance_data.data['datetime'] < '2025-01-03') &
            (imbalance_data.data['datetime'] >= '2023-01-03')
        ]
    )
    imbalance_data = imbalance_data.transform_data(
            drop_missing_threshold=0.1)
    
    # Add target to imbalance_data
    imbalance_data.set_data(imbalance_data.data.assign(
        target_DK1=imbalance_data.data['ImbalanceMWh_DK1'].apply(
            lambda x: 1 if x < 0 else (-1 if x > 0 else 0)
        )
    ))
    imbalance_data.set_data(imbalance_data.data.assign(
        target_DK2=imbalance_data.data['ImbalanceMWh_DK2'].apply(
            lambda x: 1 if x < 0 else (-1 if x > 0 else 0)
        )
    ))

    # _ = imbalance_data.validate_data(print_report=True)

    #########################################################################
    # Example of DK2 prediction

    # No NaNs or zeros in target
    dk2_data = imbalance_data.data[imbalance_data.data['target_DK2'].notnull()]
    dk2_data = dk2_data[dk2_data['target_DK2'] != 0]
    # Set index to datetime
    dk2_data = dk2_data.set_index('datetime')
    # Set features and target
    X = DataHandler()
    X.set_data(dk2_data)
    X = X.transform_data(drop_columns=['ImbalancePriceEUR_DK1',
                                       'ImbalanceMWh_DK1',
                                       'ImbalancePriceEUR_DK2',
                                       'ImbalanceMWh_DK2',
                                       'target_DK1',
                                       'target_DK2'])
    y = dk2_data['target_DK2']

    del dk2_data  # Free memory

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
        test_size=0.5,
        shuffle=True,
        random_state=42,
        stratify=y,
    )

    # Fit model
    predictor.fit(X_train, y_train)

    # Predict
    predictions = predictor.predict(X_test)

    # Evaluate accuracy
    metrics = predictor.evaluate_classifier(X_test, y_test)
    print(metrics)


