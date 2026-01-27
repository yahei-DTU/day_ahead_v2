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
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.feature_selection import RFE
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
# from sklearn.metrics import (
#     accuracy_score,
#     confusion_matrix,
#     ConfusionMatrixDisplay,
#     f1_score,
#     roc_auc_score,
#     roc_curve
# )
import hydra
from omegaconf import DictConfig
# from tabulate import tabulate
# from matplotlib.colors import TwoSlopeNorm, ListedColormap
# from matplotlib.lines import Line2D
# import yaml
# from src.day_ahead_v2.data_handler import DataHandler
# from utils.plot_settings import color_palette_1, color_palette_2, apply_plot_settings


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Your training code here
    print(cfg)

if __name__ == "__main__":
    sys.exit(main())

    # Load configuration
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs"
    @hydra.main(config_name="config_dev.yaml", config_path=config_path)
    def main(config: Dict[str, Any]) -> None:
        print("[INFO] Starting imbalance prediction...")


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
    predictor = LogisticRegressionPredictor(l1_ratio=1,
                                            max_iter=10000,
                                            solver='saga',
                                            class_weight='balanced')

    # Preprocess features
    print("[INFO] Preprocessing features...")
    X_processed = predictor.preprocess(X.data)

    print(X_processed.head())
    print(y_label.head())

    sys.exit(0)

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
