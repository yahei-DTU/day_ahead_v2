#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File name: descriptive_analysis.py
Author: Yannick Heiser
Created: 2025-11-27
Version: 1.0
Description:
    This script applies descriptive statistical analysis on a given dataset.
    It computes measures such as mean, median, mode, standard deviation, variance, and range.

Contact: yahei@dtu.dk
Dependencies: pandas, typing, pathlib, data_handler
"""

from pathlib import Path
from typing import Any, Sequence
import hydra
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from tabulate import tabulate
from day_ahead_v2.utils.plot_settings import color_palette_1, color_palette_2
from day_ahead_v2.data import DataHandler

logger = logging.getLogger(__name__)

class DescriptiveAnalysis:
    """
    A class to perform descriptive statistical analysis on a dataset.
    Attributes:
        data (pd.DataFrame): The dataset on which to perform the analysis.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initializes the DescriptiveAnalysis class with the provided dataset.
        Args:
            data (pd.DataFrame): The dataset to analyze.
        """
        self.data: pd.DataFrame = data

    def compute_numerical_statistics(self, cols: Sequence[str] | None = None) -> dict[str, Any]:
        """
        Computes descriptive statistics for the numerical columns in the dataset.

        Args:
            cols (Sequence[str] | None): A list of columns to include in the analysis.

        Returns:
            statistics (dict[str, Any]): A dictionary containing the computed statistics.
        """
        statistics = {}
        numerical_columns = self.data.select_dtypes(include=[np.number]).columns

        if cols is not None:
            if not all(isinstance(c, str) for c in cols):
                raise TypeError("All elements in 'cols' must be of type str")
            numerical_columns = [col for col in cols if col in numerical_columns]

        for col in numerical_columns:
            col_series = self.data[col].dropna()
            if col_series.empty:
                statistics[col] = {
                    'mean': None,
                    'median': None,
                    'mode': None,
                    'std_dev': None,
                    'variance': None,
                    'min': None,
                    'max': None
                }
                continue

            try:
                mode_val = col_series.mode().iloc[0]
            except Exception:
                mode_val = None

            statistics[col] = {
                'mean': col_series.mean(),
                'median': col_series.median(),
                'mode': mode_val,
                'std_dev': col_series.std(),
                'variance': col_series.var(),
                'min': col_series.min(),
                'max': col_series.max()
            }

        return statistics

    def compute_categorical_statistics(self, cols: Sequence[str] | None = None) -> dict[str, Any]:
        """
        Computes descriptive statistics for categorical columns in the dataset.

        Args:
            cols (Sequence[str] | None): A list of columns to include in the analysis.

        Returns:
            statistics (dict[str, Any]): A dictionary containing the computed statistics for categorical columns.
        """
        categorical_stats = {}
        categorical_columns = self.data.select_dtypes(include=['object', 'category', 'string']).columns

        if cols is not None:
            if not all(isinstance(c, str) for c in cols):
                raise TypeError("All elements in 'cols' must be of type str")
            categorical_columns = [col for col in cols if col in categorical_columns]

        for col in categorical_columns:
            col_series = self.data[col].dropna()
            if col_series.empty:
                categorical_stats[col] = {
                    'mode': None,
                    'unique_values': None,
                    'value_counts': None
                }
                continue

            try:
                mode_val = col_series.mode().iloc[0]
            except Exception:
                mode_val = None

            categorical_stats[col] = {
                'mode': mode_val,
                'unique_values': col_series.unique().tolist(),
                'value_counts': col_series.value_counts().to_dict()
            }

        return categorical_stats

    def anova_test(self, target_col: str, feature_col: Sequence[str] | None = None) -> pd.DataFrame:
        """
        Performs a one-way ANOVA test per feature to determine whether feature means differ significantly across the target categories.

        NaN values are dropped per feature (not globally).

        Args:
            target_col (str): The target categorical column.
            feature_col (Sequence[str] | None): Optional subset of numerical
                feature columns. If None, all numerical columns are tested.

        Raises:
            ValueError: If the target column is not found in the dataset.
            TypeError: If elements in feature_col are not strings.

        Returns:
            pd.DataFrame: A DataFrame with columns:
                - 'feature'
                - 'F_statistic'
                - 'p_value'
                - 'eta_squared'
                - 'n_samples'
        """

        if target_col not in self.data.columns:
            raise ValueError(f"Target column '{target_col}' not found in the dataset.")

        # Find numerical columns
        numerical_columns = self.data.select_dtypes(include=[np.number]).columns

        # Restrict to subset if provided
        if feature_col is not None:
            if not all(isinstance(c, str) for c in feature_col):
                raise TypeError("All elements in 'feature_col' must be strings.")
            numerical_columns = pd.Index([col for col in feature_col if col in numerical_columns])

        unique_classes = self.data[target_col].dropna().unique()

        results = []

        # Run ANOVA per feature
        for col in numerical_columns:
            groups = []
            total_n_samples = 0

            # Build per-class numerical groups, drop NaNs
            for cls in unique_classes:
                values = self.data.loc[self.data[target_col] == cls, col].dropna()
                if len(values) > 0:
                    groups.append(values)
                    total_n_samples += len(values)

            # Must have at least 2 valid groups to perform ANOVA
            if len(groups) < 2:
                results.append({
                    "feature": col,
                    "F_statistic": np.nan,
                    "p_value": np.nan,
                    "eta_squared": np.nan,
                    "n_samples": total_n_samples
                })
                continue

            # Perform ANOVA
            F, p = f_oneway(*groups)

            # Compute eta squared: SS_between / SS_total
            # SS_between = sum(n_i * (mean_i - grand_mean)^2)
            # SS_total = sum((x - grand_mean)^2)
            all_values = np.concatenate(groups)
            grand_mean = np.mean(all_values)
            ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
            ss_total = sum((x - grand_mean) ** 2 for x in all_values)
            eta_squared = ss_between / ss_total if ss_total > 0 else np.nan

            results.append({
                "feature": col,
                "F_statistic": F,
                "p_value": p,
                "eta_squared": eta_squared,
                "n_samples": total_n_samples
            })

        return pd.DataFrame(results)

    def correlation_matrix(self, cols: Sequence[str] | None = None) -> pd.DataFrame:
        """
        Computes the correlation matrix for the dataset.

        Args:
            cols (Sequence[str] | None): A list of columns to include in the correlation matrix.
                If None, includes all numerical columns.

        Raises:
            TypeError: If elements in 'cols' are not strings.

        Returns:
            pd.DataFrame: The correlation matrix.
        """
        numerical_columns = self.data.select_dtypes(include=[np.number]).columns

        if cols is not None:
            if not all(isinstance(c, str) for c in cols):
                raise TypeError("All elements in 'cols' must be of type str")
            numerical_columns = pd.Index([col for col in cols if col in numerical_columns])

        return self.data[numerical_columns].corr()

    def mutual_information(
        self,
        target_col: str,
        feature_col: Sequence[str] | None = None,
        is_classification: bool = True,
        random_state: int | None = 42
    ) -> pd.DataFrame:
        """
        Computes mutual information between numerical features and a target.

        If is_classification=True -> uses mutual_info_classif (categorical target).
        If is_classification=False -> uses mutual_info_regression (numeric target).

        NaN values are dropped per feature (not globally).

        Args:
            target_col (str): The target column.
            feature_col (Sequence[str] | None): Optional subset of feature columns.
                If None, all numerical columns are used.
            is_classification (bool): Choose classification or regression MI.
            random_state (int | None): Random state for MI reproducibility.

        Returns:
            pd.DataFrame: A DataFrame with columns:
                - 'feature'
                - 'mutual_information'
                - 'n_samples'
        """

        if target_col not in self.data.columns:
            raise ValueError(f"Target column '{target_col}' not found in the dataset.")

        numerical_columns = self.data.select_dtypes(include=[np.number]).columns

        if feature_col is not None:
            if not all(isinstance(c, str) for c in feature_col):
                raise TypeError("All elements in 'feature_col' must be strings.")
            numerical_columns = pd.Index([col for col in feature_col if col in numerical_columns])

        results = []

        for col in numerical_columns:
            df_valid = self.data[[col, target_col]].dropna()
            n_samples = len(df_valid)

            if n_samples < 2:
                results.append({
                    "feature": col,
                    "mutual_information": np.nan,
                    "n_samples": n_samples
                })
                continue

            X = df_valid[col].values.reshape(-1, 1)
            y = df_valid[target_col].to_numpy().ravel()

            if is_classification:
                # Encode target if needed
                if not pd.api.types.is_numeric_dtype(df_valid[target_col]):
                    y = LabelEncoder().fit_transform(y)

                mi = mutual_info_classif(
                    X,
                    y,
                    discrete_features=False,
                    random_state=random_state
                )[0]
            else:
                mi = mutual_info_regression(
                    X,
                    y,
                    discrete_features=False,
                    random_state=random_state
                )[0]

            results.append({
                "feature": col,
                "mutual_information": mi,
                "n_samples": n_samples
            })

        return pd.DataFrame(results)

    def plot_correlation_heatmap(self, output_dir: str | None, cols: Sequence[str] | None = None) -> None:
        """
        Plots a heatmap of the correlation matrix.

        Args:
            output_dir (str | None): The directory where the plot will be saved.
                If None, saves in the same directory as the script.
            cols (Optional[list]): A list of columns to include in the heatmap.
                If None, includes all numerical columns.
        """
        # Resolve save directory
        if output_dir is None:
            save_path = Path(__file__).parent
        else:
            save_path = Path(output_dir)

        if not save_path.is_absolute():
            save_path = Path(__file__).resolve().parent.parent / save_path

        save_path.mkdir(parents=True, exist_ok=True)

        if cols is not None:
            corr_matrix = self.correlation_matrix().loc[cols, cols]
        else:
            corr_matrix = self.correlation_matrix()

        plt.figure()
        plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
        plt.colorbar()
        plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=90)
        plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(save_path / 'correlation_heatmap.pdf', dpi=300, bbox_inches="tight")
        plt.close()

    def plot_boxplots(self, output_dir: str | None, cols: Sequence[str] | None = None) -> None:
        """
        Plots boxplots for each numerical column in the dataset.

        Args:
            output_dir (str | None): The directory where the plots will be saved.
                If None, saves in the same directory as the script.
            cols (Optional[Sequence[str]]): A list of columns to include in the boxplots.
        """
        # Resolve save directory
        if output_dir is None:
            save_path = Path(__file__).parent
        else:
            save_path = Path(output_dir)

        if not save_path.is_absolute():
            save_path = Path(__file__).resolve().parent.parent / save_path

        save_path.mkdir(parents=True, exist_ok=True)

        for column in self.data.select_dtypes(include=[np.number]).columns:
            if cols is not None and column not in cols:
                continue
            plt.figure()
            self.data.boxplot(column=column)
            plt.title(f'Boxplot of {column}')
            plt.ylabel(column)
            plt.grid(False)
            plt.savefig(save_path / f'{column}_boxplot.png',
                        dpi=300,
                        bbox_inches="tight")
            plt.close()

    def plot_histograms(self, output_dir: str | None, cols: Sequence[str] | None = None) -> None:
        """
        Plots histograms for each numerical column in the dataset.

        Args:
            output_dir (str | None): The directory where the plots will be saved.
                If None, saves in the same directory as the script.
            cols (Optional[Sequence[str]]): A list of columns to include in the histograms.
        """
        # Resolve save directory
        if output_dir is None:
            save_path = Path(__file__).parent
        else:
            save_path = Path(output_dir)

        if not save_path.is_absolute():
            save_path = Path(__file__).resolve().parent.parent / save_path

        save_path.mkdir(parents=True, exist_ok=True)

        for column in self.data.select_dtypes(include=[np.number]).columns:
            if cols is not None and column not in cols:
                continue
            plt.figure()
            self.data[column].hist(bins=30, color=color_palette_1['blue'], edgecolor='black')
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.grid(False)
            plt.tight_layout()
            plt.savefig(
                save_path / f'{column}_histogram.pdf',
                dpi=300,
                bbox_inches="tight")
            plt.close()


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config_dev.yaml")
def main(cfg):
    logger.info("Starting main function")
    # Import imbalance dataset
    imbalance_data = DataHandler(cfg.datasets)

    # Filter data between specific dates
    imbalance_data.set_data(
        imbalance_data.data[
            (imbalance_data.data['datetime'] < cfg.experiments.experiment_parameters.end_date) &
            (imbalance_data.data['datetime'] >= cfg.experiments.experiment_parameters.start_date)
            ]
        )

    # Initialize DescriptiveAnalysis
    descriptive_analyzer = DescriptiveAnalysis(imbalance_data.data)

    # Value counts for imbalance direction
    categorial_stats = descriptive_analyzer.compute_categorical_statistics(
        cols=["ImbalanceDirection_DK1"]
    )
    logger.info("Categorical Statistics for ImbalanceDirection_DK1:")
    for col, stat in categorial_stats.items():
        logger.info(f"{col}: {stat}")

    # Compute numerical statistics
    numerical_stats = descriptive_analyzer.compute_numerical_statistics(
        cols=["ImbalanceMWh_DK1", "ImbalancePriceEUR_DK1"]
        )
    logger.info("Numerical Statistics:")
    for col, stats in numerical_stats.items():
        logger.info(f"{col}: {stats}")

    # Plot histograms
    descriptive_analyzer.plot_histograms(
        output_dir="reports/figures/descriptive_analysis", cols=["ImbalanceMWh_DK1",
                                                         "ImbalancePriceEUR_DK1"]
        )

    # Separate surplus and deficit data
    surplus_data = imbalance_data.data[imbalance_data.data['ImbalanceDirection_DK1'] == "1"]
    neutral_data = imbalance_data.data[imbalance_data.data['ImbalanceDirection_DK1'] == "0"]
    deficit_data = imbalance_data.data[imbalance_data.data['ImbalanceDirection_DK1'] == "-1"]

    # Initialize DescriptiveAnalysis for surplus and deficit
    surplus_analyzer = DescriptiveAnalysis(surplus_data)
    neutral_analyzer = DescriptiveAnalysis(neutral_data)
    deficit_analyzer = DescriptiveAnalysis(deficit_data)

    # Compute numerical statistics for surplus and deficit
    numerical_stats_surplus = surplus_analyzer.compute_numerical_statistics(
        cols=["ImbalanceMWh_DK1", "ImbalancePriceEUR_DK1"]
        )
    numerical_stats_neutral = neutral_analyzer.compute_numerical_statistics(
        cols=["ImbalanceMWh_DK1", "ImbalancePriceEUR_DK1"]
        )
    numerical_stats_deficit = deficit_analyzer.compute_numerical_statistics(
        cols=["ImbalanceMWh_DK1", "ImbalancePriceEUR_DK1"]
        )

    logger.info("\nNumerical Statistics for Surplus:")
    for col, stats in numerical_stats_surplus.items():
        logger.info(f"{col}: {stats}")

    logger.info("\nNumerical Statistics for Neutral:")
    for col, stats in numerical_stats_neutral.items():
        logger.info(f"{col}: {stats}")

    logger.info("\nNumerical Statistics for Deficit:")
    for col, stats in numerical_stats_deficit.items():
        logger.info(f"{col}: {stats}")

    # Compute and print correlation matrix
    corr_matrix = descriptive_analyzer.correlation_matrix()

    # Print the 30 columns with highest absolute correlation to ImbalanceDirection_DK1
    if "ImbalanceMWh_DK1" not in corr_matrix.index:
        logger.info("ImbalanceMWh_DK1 not found in correlation matrix.")
    else:
        top_abs = (
            corr_matrix.loc["ImbalanceMWh_DK1"]
            .drop(labels=["ImbalanceMWh_DK1"], errors="ignore")
            .abs()
            .sort_values(ascending=False)
            .head(30)
        )
        top30_original = corr_matrix.loc["ImbalanceMWh_DK1", top_abs.index]
        logger.info("Top 30 columns by absolute correlation with ImbalanceMWh_DK1:")
        # Build a table and print it using tabulate
        top30_df = top30_original.rename_axis('column').reset_index(name='correlation')
        logger.info(tabulate(
            top30_df.values, headers=top30_df.columns, tablefmt='github', floatfmt=".6f"
            ))

    # Compute ANOVA test
    anova_stats = descriptive_analyzer.anova_test(
        target_col="ImbalanceDirection_DK1"
        )
    logger.info("\nANOVA Test Results:")
    df_sorted = anova_stats.sort_values('eta_squared', ascending=False)

    table_str = tabulate(
        df_sorted.values,       # only the values, not the DataFrame object
        headers=df_sorted.columns,
        tablefmt="github",
            floatfmt=".6f"
        )
    logger.info("\n" + table_str)

    # Compute correlations for extreme events: lowest 10% and highest 90% of ImbalanceMWh_DK1
    p10 = imbalance_data.data['ImbalanceMWh_DK1'].quantile(0.05)
    p90 = imbalance_data.data['ImbalanceMWh_DK1'].quantile(0.95)
    extreme_data = imbalance_data.data[
        (imbalance_data.data['ImbalanceMWh_DK1'] <= p10) |
        (imbalance_data.data['ImbalanceMWh_DK1'] >= p90)
    ]

    extreme_analyzer = DescriptiveAnalysis(extreme_data)

    extreme_corr_matrix = extreme_analyzer.correlation_matrix()

    # Print the 30 columns with highest absolute correlation to ImbalanceDirection_DK1
    if "ImbalanceMWh_DK1" not in extreme_corr_matrix.index:
        logger.info("ImbalanceMWh_DK1 not found in correlation matrix.")
    else:
        top_abs = (
            extreme_corr_matrix.loc["ImbalanceMWh_DK1"]
            .drop(labels=["ImbalanceMWh_DK1"], errors="ignore")
            .abs()
            .sort_values(ascending=False)
            .head(30)
        )
        top30_original = extreme_corr_matrix.loc["ImbalanceMWh_DK1", top_abs.index]
        logger.info("Top 30 columns by absolute correlation with ImbalanceMWh_DK1:")
        # Build a table and print it using tabulate
        top30_df = top30_original.rename_axis('column').reset_index(name='correlation')
        logger.info(tabulate(
            top30_df.values, headers=top30_df.columns, tablefmt='github', floatfmt=".6f"
            ))

    # Compute ANOVA test
    anova_stats = extreme_analyzer.anova_test(
        target_col="ImbalanceDirection_DK1"
        )
    logger.info("\nANOVA Test Results:")
    df_sorted = anova_stats.sort_values('eta_squared', ascending=False)

    table_str = tabulate(
        df_sorted.values,       # only the values, not the DataFrame object
        headers=df_sorted.columns,
        tablefmt="github",
        floatfmt=".6f"
    )

    logger.info("\n" + table_str)

    # Compute mutual information for IM Direction
    mi_stats = descriptive_analyzer.mutual_information(
        target_col="ImbalanceDirection_DK1",
        is_classification=True
        )
    logger.info("\nMutual Information Results:")
    df_sorted = mi_stats.sort_values('mutual_information', ascending=False)

    table_str = tabulate(
        df_sorted.values,       # only the values, not the DataFrame object
        headers=df_sorted.columns,
        tablefmt="github",
        floatfmt=".6f"
    )

    logger.info("\n" + table_str)

    # Compute mutual information for IM MWh
    mi_stats = descriptive_analyzer.mutual_information(
        target_col="ImbalanceMWh_DK1",
        is_classification=False
        )
    logger.info("\nMutual Information Results:")
    df_sorted = mi_stats.sort_values('mutual_information', ascending=False)

    table_str = tabulate(
        df_sorted.values,       # only the values, not the DataFrame object
        headers=df_sorted.columns,
        tablefmt="github",
        floatfmt=".6f"
    )

    logger.info("\n" + table_str)

if __name__ == "__main__":
    main()
