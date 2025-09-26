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

from statistics import correlation
import pandas as pd
import os
import numpy as np
from typing import Dict, Any, Union, Optional
import sys
from sklearn.linear_model import LogisticRegression
from data_handler import DataHandler

class ImbalancePredictor:
    """
    Base class for system imbalance prediction.
    Subclasses must implement `_load_model()`.
    """

    def __init__(self, model_params: Dict[str, Any]):
        self.model_params = model_params
        self.model = self._load_model()

    def analyze_features(self, features: pd.DataFrame, **kwargs: Any) -> None:
        """
        Analyze and preprocess features before prediction.
        """
        if features.empty:
            raise ValueError("Features DataFrame is empty.")
        
        # Calculate cross correlation between all features
        print("Analyzing feature correlations...")
        corr_matrix = features.corr(method='pearson')
        print("Feature cross-correlation matrix:\n", corr_matrix)

    def feature_selection(self, features: pd.DataFrame, target: pd.Series, model=None, threshold: float = 0.1) -> pd.DataFrame:
        """
        Select features based on mean absolute SHAP values.
        Requires a fitted model and SHAP installed.
        """
        if "shap" not in sys.modules:
            import shap
        if features.empty or target.empty:
            raise ValueError("Features or target DataFrame is empty.")
        if model is None:
            raise ValueError("A fitted model must be provided for SHAP feature selection.")
        print("Selecting features based on SHAP values...")
        # Use TreeExplainer for tree models, otherwise KernelExplainer
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = shap.KernelExplainer(model.predict, features)
        shap_values = explainer.shap_values(features)
        # If shap_values is a list (multi-class), take the mean across classes
        if isinstance(shap_values, list):
            shap_values = np.mean(np.abs(shap_values), axis=0)
        else:
            shap_values = np.abs(shap_values)
        mean_shap = np.mean(shap_values, axis=0)
        selected_features = [features.columns[i] for i, val in enumerate(mean_shap) if val >= threshold]
        for col, val in zip(features.columns, mean_shap):
            if val >= threshold:
                print(f"Selected feature: {col} with mean(|SHAP|): {val}")
        return features[selected_features]


    def _load_model(self):
        """
        Subclasses must override this method to load the specific model.
        """
        raise NotImplementedError("Subclasses must implement _load_model()")

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Run predictions using the loaded model.
        """
        if self.model is None:
            raise ValueError("Model is not loaded.")
        # In real code: return pd.Series(self.model.predict(features), index=features.index)
        return pd.Series(np.random.randn(len(features)), index=features.index)
    
    def preprocess(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features before prediction.
        """
        if features.empty:
            raise ValueError("Features DataFrame is empty.")
        # Example preprocessing: fill NaNs with column means
        features = features.fillna(features.mean())
        return features


class LogisticRegressionPredictor(ImbalancePredictor):
    """
    Imbalance predictor using Logistic Regression.
    """
    def _load_model(self):
        return LogisticRegression(**self.model_params)


class BartPredictor(ImbalancePredictor):
    """
    Imbalance predictor using BART.
    """
    def _load_model(self):
        # Replace with actual BART model initialization
        # Example: return BartRegressor(**self.model_params)
        return None
    
if __name__ == "__main__":
    # Example usage
    print("A")
    model_params = {"C": 1.0, "max_iter": 100}
    predictor = LogisticRegressionPredictor(model_params)
    
    # Import features
    imbalance_data = DataHandler("imbalance_data.parquet",
                                  "../data/processed")
    print("B")
    imbalance_data.preview()

    sys.exit()  # Temporary exit to avoid running incomplete code below

    # DK2 example
    features = imbalance_data.data
    features = features[features['datetime'] <= '2024-12-31']
    print("Number of rows with NaN values:", features.isna().any(axis=1).sum())
    print("Number of columns with NaN values:", features.isna().sum().gt(0).sum())

    print("NaN percentage per column:")
    for col in features.columns:
        nan_percent = features[col].isna().mean() * 100
        print(f"{col}: {nan_percent:.2f}% NaN")

    

    sys.exit()  # Temporary exit to avoid running incomplete code below

    target = features['ImbalanceMWh_DK2'].apply(lambda x: 1 if x < 0 else (-1 if x > 0 else 0))
    features = features.drop(columns=['datetime',
                                      'ImbalanceMWh_DK1',
                                      'ImbalancePriceEUR_DK1',
                                      'ImbalanceMWh_DK2',
                                      'ImbalancePriceEUR_DK2'])
    print("Length of target:", len(target))
    print("Unique values in target:", target.unique())
    print("Value counts in target:")
    print(target.value_counts())

    print("Rows where target == 0:")
    print(target[target == 0])

    sys.exit()  # Temporary exit to avoid running incomplete code below

    
    predictor.analyze_features(features)
    selected_features = predictor.feature_selection(features, target, model=predictor.model, threshold=0.1)
    predictions = predictor.predict(selected_features)
    print(predictions.head())