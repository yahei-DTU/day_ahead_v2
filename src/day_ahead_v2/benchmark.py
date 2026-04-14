"""
File name: benchmark.py
Author: Yannick Heiser
Created: 2026-02-08
Last modified: 2026-02-08
Version: 1.0
Description:
    Run and evaluate the benchmark models.
Contact: yahei@dtu.dk
Dependencies: pandas, os, typing, pathlib, data_validation
"""

from pathlib import Path
import numpy as np
import pandas as pd
import hydra
import logging
import copy
from day_ahead_v2.optimization import ModelHindsight, ModelSinglePolicy
from day_ahead_v2.data import PandasHandler
from day_ahead_v2.train import rolling_windows, feature_selection, split_features_target
from day_ahead_v2.evaluate import calculate_profit, calculate_hydrogen_balancing_bids
from day_ahead_v2.utils import electrolyzer_efficiency

logger = logging.getLogger(__name__)

def simulate_ar1_errors(index, phi, sigma, initial_error):
    errors = np.zeros(len(index))
    errors[0] = initial_error
    for i in range(1, len(index)):
        eps = np.random.normal(0, sigma)
        errors[i] = phi * errors[i-1] + eps
    return pd.Series(errors, index=index)

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config_dev")
def main(cfg):
    H2_PRICE = cfg.experiments.optimization_parameters.hydrogen_price
    # ---------------------------------------------
    # Import dataset
    # ---------------------------------------------
    raw_data_handler = PandasHandler(cfg)
    raw_data_handler = raw_data_handler.cut_data(cfg.experiments.experiment_parameters.start_date,
                                        cfg.experiments.experiment_parameters.end_date,
                                        cfg.datasets.training.datetime_column,
                                        )

    # ----------------------------------------------
    # Initialize the electrolyzer
    # ----------------------------------------------
    electrolyzer_efficiency.initiate_HYP_L(cfg)

    # ---------------------------------------------
    # Rolling window backtest
    # ---------------------------------------------
    windows = list(rolling_windows(cfg))
    cfg_snapshot = copy.deepcopy(cfg)

    all_results_hindsight = []
    all_results_policy = []
    all_results_bid_forecast = []
    all_train_results_dfs_hindsight = pd.DataFrame()
    all_test_results_dfs_hindsight = pd.DataFrame()
    all_train_results_dfs_policy = pd.DataFrame()
    all_test_results_dfs_policy = pd.DataFrame()
    all_train_results_dfs_bid_forecast = pd.DataFrame()
    all_test_results_dfs_bid_forecast = pd.DataFrame()

    for window in windows:

        train_start, train_end = window["train"]
        valid_start, valid_end = window["valid"]
        test_start, test_end   = window["test"]

        logger.info(
            f"Train: {train_start.date()} → {train_end.date()} | "
            f"Valid: {valid_start.date()} → {valid_end.date()} | "
            f"Test: {test_start.date()} → {test_end.date()}"
        )

        # Reset cfg and transform from raw data for each window
        cfg = copy.deepcopy(cfg_snapshot)
        data_handler = raw_data_handler.transform_data(cfg)

        # Check for NaNs in data
        nan_counts = data_handler.data.isnull().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"Found NaN values in data:\n{nan_counts[nan_counts > 0]}")
        else:
            logger.info("No NaN values found in data")

        # Feature selection (optional, only on first window)
        if cfg.experiments.train_parameters.get("feature_selection", False) and window == windows[0]:
            selected_features = feature_selection(cfg, data_handler.data, start=train_start, end=train_end)
            cfg.datasets.training.feature_columns_flex = selected_features
            cfg_snapshot.datasets.training.feature_columns_flex = selected_features

        # ---------------------------------------------
        # Data loading
        # ---------------------------------------------
        logger.debug("Cutting train data...")
        data_train = data_handler.cut_data(train_start, train_end, cfg.datasets.training.datetime_column)
        logger.debug("Splitting features and target for training data...")
        X_train, y_train = split_features_target(cfg, data_train.data, sanitize_names=True)
        logger.info(f"Target counts in training data:\n{y_train.value_counts()}")

        logger.debug("Cutting validation data...")
        data_valid = data_handler.cut_data(valid_start, valid_end, cfg.datasets.training.datetime_column)
        logger.debug("Splitting features and target for validation data...")
        X_val, y_val = split_features_target(cfg, data_valid.data, sanitize_names=True)
        logger.info(f"Target counts in validation data:\n{y_val.value_counts()}")

        logger.debug("Cutting test data...")
        data_test = data_handler.cut_data(test_start, test_end, cfg.datasets.training.datetime_column)
        logger.debug("Splitting features and target for test data...")
        X_test, y_test = split_features_target(cfg, data_test.data, sanitize_names=True)
        logger.info(f"Target counts in test data:\n{y_test.value_counts()}")

        data_train_full = pd.concat([data_train.data, data_valid.data])
        X_train_full = pd.concat([X_train, X_val])
        datetime_index_train = data_train_full.index
        datetime_index_test = data_test.data.index
        del data_train_full, data_valid
        data_train = data_handler.data.loc[datetime_index_train]
        data_test = data_handler.data.loc[datetime_index_test]
        lambda_DA_hat_train = data_train[cfg.datasets.optimization.lambda_DA_hat]
        lambda_B_hat_train = data_train[cfg.datasets.optimization.lambda_B_hat]
        P_W_hat_train = data_train[cfg.datasets.optimization.P_W_hat]
        P_W_tilde_train = data_train[cfg.datasets.optimization.P_W_tilde]
        lambda_DA_hat_test = data_test[cfg.datasets.optimization.lambda_DA_hat]
        lambda_B_hat_test = data_test[cfg.datasets.optimization.lambda_B_hat]
        P_W_hat_test = data_test[cfg.datasets.optimization.P_W_hat]
        P_W_tilde_test = data_test[cfg.datasets.optimization.P_W_tilde]
        # ------------------------------------- Model 1 - Perfect foresight -------------------------------------
        logger.debug("Running Hindsight Model...")
        hindsight_model_train = ModelHindsight(
            cfg = cfg,
            lambda_DA_hat=lambda_DA_hat_train,
            lambda_B_hat=lambda_B_hat_train,
            P_W_hat=P_W_hat_train,
        )
        hindsight_model_train.build_model()
        hindsight_model_train.run_optimization()
        if hindsight_model_train.results.status != "ok":
            logger.error(
                f"Optimization did not converge for training data in window {window}. Status: {hindsight_model_train.results.status}"
            )
        else:
            logger.debug(f"Optimization converged for training data in window {window}. Status: {hindsight_model_train.results.status}")
        hindsight_model_test = ModelHindsight(
            cfg = cfg,
            lambda_DA_hat=lambda_DA_hat_test,
            lambda_B_hat=lambda_B_hat_test,
            P_W_hat=P_W_hat_test,
        )
        hindsight_model_test.build_model()
        hindsight_model_test.run_optimization()
        if hindsight_model_test.results.status != "ok":
            logger.error(
                f"Optimization did not converge for test data in window {window}. Status: {hindsight_model_test.results.status}"
            )
        else:
            logger.debug(f"Optimization converged for test data in window {window}. Status: {hindsight_model_test.results.status}")
        p_DA_train = pd.Series(hindsight_model_train.results.p_DA, index=data_train.index)
        p_B_train = pd.Series(hindsight_model_train.results.p_B, index=data_train.index)
        p_H_train = pd.Series(hindsight_model_train.results.p_H, index=data_train.index)
        h_train = pd.Series(hindsight_model_train.results.h, index=data_train.index)
        p_DA_test = pd.Series(hindsight_model_test.results.p_DA, index=data_test.index)
        p_B_test = pd.Series(hindsight_model_test.results.p_B, index=data_test.index)
        p_H_test = pd.Series(hindsight_model_test.results.p_H, index=data_test.index)
        h_test = pd.Series(hindsight_model_test.results.h, index=data_test.index)
        profit_train = calculate_profit(p_DA_train, h_train, p_B_train, lambda_DA_hat_train, H2_PRICE, lambda_B_hat_train)
        profit_test = calculate_profit(p_DA_test, h_test, p_B_test, lambda_DA_hat_test, H2_PRICE, lambda_B_hat_test)
        # Add bids and profit to results dfs
        results_hindsight_train_df = pd.DataFrame({
            "p_DA": p_DA_train,
            "p_B": p_B_train,
            "p_H": p_H_train,
            "h": h_train,
            "profit": profit_train,
            "P_W_hat": P_W_hat_train,
            "P_W_tilde": P_W_tilde_train,
            "lambda_DA_hat": lambda_DA_hat_train,
            "lambda_B_hat": lambda_B_hat_train
        }, index=data_train.index)
        results_hindsight_test_df = pd.DataFrame({
            "p_DA": p_DA_test,
            "p_B": p_B_test,
            "p_H": p_H_test,
            "h": h_test,
            "profit": profit_test,
            "P_W_hat": P_W_hat_test,
            "P_W_tilde": P_W_tilde_test,
            "lambda_DA_hat": lambda_DA_hat_test,
            "lambda_B_hat": lambda_B_hat_test
        }, index=data_test.index)
        # ---------------------------------------------
        # Collect results
        # ---------------------------------------------
        results_hindsight = {
            "train_profit_total": profit_train.sum(),
            "train_profit_mean": profit_train.mean(),
            "test_profit_total": profit_test.sum(),
            "test_profit_mean": profit_test.mean(),
            "train_start": train_start,
            "train_end": train_end,
            "valid_start": valid_start,
            "valid_end": valid_end,
            "test_start": test_start,
            "test_end": test_end,
            "lambda_H": H2_PRICE
        }
        all_results_hindsight.append(results_hindsight)
        all_train_results_dfs_hindsight = pd.concat([all_train_results_dfs_hindsight, results_hindsight_train_df])
        all_test_results_dfs_hindsight = pd.concat([all_test_results_dfs_hindsight, results_hindsight_test_df])
        del p_DA_train, p_B_train, p_H_train, h_train, profit_train, p_DA_test, p_B_test, p_H_test, h_test, profit_test
        # ------------------------------------- Model 2 - Linear Policy -------------------------------------
        logger.debug("Running Single Linear Policy Model...")
        single_policy_model_train = ModelSinglePolicy(
            cfg = cfg,
            lambda_DA_hat=lambda_DA_hat_train,
            lambda_B_hat=lambda_B_hat_train,
            P_W_hat=P_W_hat_train,
            P_W_tilde=P_W_tilde_train,
            X_features=X_train_full
        )
        single_policy_model_train.build_model()
        single_policy_model_train.run_optimization()
        if single_policy_model_train.results.status != "ok":
            logger.error(
                f"Optimization did not converge for training data in window {window}. Status: {single_policy_model_train.results.status}"
            )
        else:
            logger.debug(f"Optimization converged for training data in window {window}. Status: {single_policy_model_train.results.status}")
        p_DA_train = pd.Series(single_policy_model_train.results.p_DA, index=data_train.index)
        p_B_train = pd.Series(single_policy_model_train.results.p_B, index=data_train.index)
        p_H_train = pd.Series(single_policy_model_train.results.p_H, index=data_train.index)
        h_train = pd.Series(single_policy_model_train.results.h, index=data_train.index)
        p_DA_test, p_B_test, p_H_test, h_test = single_policy_model_train.calculate_bids(cfg, data_test, X_test)
        profit_train = calculate_profit(p_DA_train, h_train, p_B_train, lambda_DA_hat_train, H2_PRICE, lambda_B_hat_train)
        profit_test = calculate_profit(p_DA_test, h_test, p_B_test, lambda_DA_hat_test, H2_PRICE, lambda_B_hat_test)
        # Add bids and profit to results dfs
        results_policy_train_df = pd.DataFrame({
            "p_DA": p_DA_train,
            "p_B": p_B_train,
            "p_H": p_H_train,
            "h": h_train,
            "profit": profit_train,
            "P_W_hat": P_W_hat_train,
            "P_W_tilde": P_W_tilde_train,
            "lambda_DA_hat": lambda_DA_hat_train,
            "lambda_B_hat": lambda_B_hat_train
        }, index=data_train.index)
        results_policy_test_df = pd.DataFrame({
            "p_DA": p_DA_test,
            "p_B": p_B_test,
            "p_H": p_H_test,
            "h": h_test,
            "profit": profit_test,
            "P_W_hat": P_W_hat_test,
            "P_W_tilde": P_W_tilde_test,
            "lambda_DA_hat": lambda_DA_hat_test,
            "lambda_B_hat": lambda_B_hat_test
        }, index=data_test.index)
        # ---------------------------------------------
        # Collect results
        # ---------------------------------------------
        results_policy = {
            "train_profit_total": profit_train.sum(),
            "train_profit_mean": profit_train.mean(),
            "train_profit_cvar": profit_train[profit_train <= np.percentile(profit_train, 5)].mean(),
            "test_profit_total": profit_test.sum(),
            "test_profit_mean": profit_test.mean(),
            "test_profit_cvar": profit_test[profit_test <= np.percentile(profit_test, 5)].mean(),
            "train_start": train_start,
            "train_end": train_end,
            "valid_start": valid_start,
            "valid_end": valid_end,
            "test_start": test_start,
            "test_end": test_end,
            "lambda_H": H2_PRICE,
            "CVaR": single_policy_model_train.results.CVaR if hasattr(single_policy_model_train.results, "CVaR") else np.nan,
        }
        all_results_policy.append(results_policy)
        all_train_results_dfs_policy = pd.concat([all_train_results_dfs_policy, results_policy_train_df])
        all_test_results_dfs_policy = pd.concat([all_test_results_dfs_policy, results_policy_test_df])
        del p_DA_train, p_B_train, p_H_train, h_train, profit_train, p_DA_test, p_B_test, p_H_test, h_test, profit_test

        # ------------------------------------- Model 3 - Bid Forecast Model -------------------------------------
        logger.debug("Running Bid Forecast Model...")
        p_DA_train = P_W_tilde_train.copy()
        p_DA_train[lambda_DA_hat_train < 0] = -cfg.experiments.optimization_parameters.electrolyzer_capacity
        p_B_train, p_H_train, h_train = calculate_hydrogen_balancing_bids(cfg, p_DA_train, lambda_B_hat_train, P_W_hat_train)
        p_DA_test = P_W_tilde_test.copy()
        p_DA_test[lambda_DA_hat_test < 0] = -cfg.experiments.optimization_parameters.electrolyzer_capacity
        p_B_test, p_H_test, h_test = calculate_hydrogen_balancing_bids(cfg, p_DA_test, lambda_B_hat_test, P_W_hat_test)
        profit_train = calculate_profit(p_DA_train, h_train, p_B_train, lambda_DA_hat_train, H2_PRICE, lambda_B_hat_train)
        profit_test = calculate_profit(p_DA_test, h_test, p_B_test, lambda_DA_hat_test, H2_PRICE, lambda_B_hat_test)
        # Add bids and profit to results dfs
        results_bid_forecast_train_df = pd.DataFrame({
            "p_DA": p_DA_train,
            "p_B": p_B_train,
            "p_H": p_H_train,
            "h": h_train,
            "profit": profit_train,
            "P_W_hat": P_W_hat_train,
            "P_W_tilde": P_W_tilde_train,
            "lambda_DA_hat": lambda_DA_hat_train,
            "lambda_B_hat": lambda_B_hat_train,
        }, index=data_train.index)
        results_bid_forecast_test_df = pd.DataFrame({
            "p_DA": p_DA_test,
            "p_B": p_B_test,
            "p_H": p_H_test,
            "h": h_test,
            "profit": profit_test,
            "P_W_hat": P_W_hat_test,
            "P_W_tilde": P_W_tilde_test,
            "lambda_DA_hat": lambda_DA_hat_test,
            "lambda_B_hat": lambda_B_hat_test
        }, index=data_test.index)
        # ---------------------------------------------
        # Collect results
        # ---------------------------------------------
        results_bid_forecast = {
            "train_profit_total": profit_train.sum(),
            "train_profit_mean": profit_train.mean(),
            "test_profit_total": profit_test.sum(),
            "test_profit_mean": profit_test.mean(),
            "train_start": train_start,
            "train_end": train_end,
            "valid_start": valid_start,
            "valid_end": valid_end,
            "test_start": test_start,
            "test_end": test_end,
            "lambda_H": H2_PRICE
        }
        all_results_bid_forecast.append(results_bid_forecast)
        all_train_results_dfs_bid_forecast = pd.concat([all_train_results_dfs_bid_forecast, results_bid_forecast_train_df])
        all_test_results_dfs_bid_forecast = pd.concat([all_test_results_dfs_bid_forecast, results_bid_forecast_test_df])
        del p_DA_train, p_B_train, p_H_train, h_train, profit_train, p_DA_test, p_B_test, p_H_test, h_test, profit_test

    # Save results
    total_profit_train_hindsight = all_train_results_dfs_hindsight["profit"].sum()
    mean_profit_train_hindsight = all_train_results_dfs_hindsight["profit"].mean()
    std_profit_train_hindsight = all_train_results_dfs_hindsight["profit"].std()
    total_profit_test_hindsight = all_test_results_dfs_hindsight["profit"].sum()
    mean_profit_test_hindsight = all_test_results_dfs_hindsight["profit"].mean()
    std_profit_test_hindsight = all_test_results_dfs_hindsight["profit"].std()
    total_profit_train_policy = all_train_results_dfs_policy["profit"].sum()
    mean_profit_train_policy = all_train_results_dfs_policy["profit"].mean()
    std_profit_train_policy = all_train_results_dfs_policy["profit"].std()
    profit_train_policy = all_train_results_dfs_policy["profit"]
    cvar95_train_policy = profit_train_policy[profit_train_policy <= np.percentile(profit_train_policy, 5)].mean()
    total_profit_test_policy = all_test_results_dfs_policy["profit"].sum()
    mean_profit_test_policy = all_test_results_dfs_policy["profit"].mean()
    std_profit_test_policy = all_test_results_dfs_policy["profit"].std()
    profit_test_policy = all_test_results_dfs_policy["profit"]
    cvar95_test_policy = profit_test_policy[profit_test_policy <= np.percentile(profit_test_policy, 5)].mean()
    total_profit_train_bid_forecast = all_train_results_dfs_bid_forecast["profit"].sum()
    mean_profit_train_bid_forecast = all_train_results_dfs_bid_forecast["profit"].mean()
    std_profit_train_bid_forecast = all_train_results_dfs_bid_forecast["profit"].std()
    profit_train_bid_forecast = all_train_results_dfs_bid_forecast["profit"]
    cvar95_train_bid_forecast = profit_train_bid_forecast[profit_train_bid_forecast <= np.percentile(profit_train_bid_forecast, 5)].mean()
    total_profit_test_bid_forecast = all_test_results_dfs_bid_forecast["profit"].sum()
    mean_profit_test_bid_forecast = all_test_results_dfs_bid_forecast["profit"].mean()
    std_profit_test_bid_forecast = all_test_results_dfs_bid_forecast["profit"].std()
    profit_test_bid_forecast = all_test_results_dfs_bid_forecast["profit"]
    cvar95_test_bid_forecast = profit_test_bid_forecast[profit_test_bid_forecast <= np.percentile(profit_test_bid_forecast, 5)].mean()

    avg_metrics_hindsight = {
        "train_profit_total": total_profit_train_hindsight,
        "train_profit_mean": f"{mean_profit_train_hindsight:.2f} ± {std_profit_train_hindsight:.2f}",
        "test_profit_total": total_profit_test_hindsight,
        "test_profit_mean": f"{mean_profit_test_hindsight:.2f} ± {std_profit_test_hindsight:.2f}",
    }
    avg_metrics_policy = {
        "train_profit_total": total_profit_train_policy,
        "train_profit_mean": f"{mean_profit_train_policy:.2f} ± {std_profit_train_policy:.2f}",
        "train_profit_cvar": cvar95_train_policy,
        "test_profit_total": total_profit_test_policy,
        "test_profit_mean": f"{mean_profit_test_policy:.2f} ± {std_profit_test_policy:.2f}",
        "test_profit_cvar": cvar95_test_policy,
    }
    avg_metrics_bid_forecast = {
        "train_profit_total": total_profit_train_bid_forecast,
        "train_profit_mean": f"{mean_profit_train_bid_forecast:.2f} ± {std_profit_train_bid_forecast:.2f}",
        "train_profit_cvar": cvar95_train_bid_forecast,
        "test_profit_total": total_profit_test_bid_forecast,
        "test_profit_mean": f"{mean_profit_test_bid_forecast:.2f} ± {std_profit_test_bid_forecast:.2f}",
        "test_profit_cvar": cvar95_test_bid_forecast,
    }

    results_hindsight_df = pd.DataFrame(all_results_hindsight)
    results_policy_df = pd.DataFrame(all_results_policy)
    results_bid_forecast_df = pd.DataFrame(all_results_bid_forecast)

    # Save results to CSV
    save_path = Path(__file__).resolve().parent.parent.parent / "reports" / cfg.experiments.experiment_name
    save_path_hindsight = save_path / "hindsight" / cfg.datasets.dataset_name
    save_path_hindsight.mkdir(parents=True, exist_ok=True)
    results_hindsight_df.to_csv(save_path_hindsight / "backtest_results.csv", index=False)
    with open(save_path_hindsight / "allwindows_metrics.txt", "w") as f:
            for key, value in avg_metrics_hindsight.items():
                f.write(f"{key}: {value}\n")

    save_path_policy = save_path / "single_policy" / cfg.datasets.dataset_name
    save_path_policy.mkdir(parents=True, exist_ok=True)
    results_policy_df.to_csv(save_path_policy / "backtest_results.csv", index=False)
    with open(save_path_policy / "allwindows_metrics.txt", "w") as f:
            for key, value in avg_metrics_policy.items():
                f.write(f"{key}: {value}\n")

    save_path_bid_forecast = save_path / "bid_forecast" / cfg.datasets.dataset_name
    save_path_bid_forecast.mkdir(parents=True, exist_ok=True)
    results_bid_forecast_df.to_csv(save_path_bid_forecast / "backtest_results.csv", index=False)
    with open(save_path_bid_forecast / "allwindows_metrics.txt", "w") as f:
            for key, value in avg_metrics_bid_forecast.items():
                f.write(f"{key}: {value}\n")

    # Save all test results to a CSV file
    all_test_results_dfs_hindsight.to_csv(save_path_hindsight / "all_test_results_hourly.csv", index=True)
    logger.info(f"All test results saved to {save_path_hindsight / 'all_test_results_hourly.csv'}")
    all_test_results_dfs_policy.to_csv(save_path_policy / "all_test_results_hourly.csv", index=True)
    logger.info(f"All test results saved to {save_path_policy / 'all_test_results_hourly.csv'}")
    all_test_results_dfs_bid_forecast.to_csv(save_path_bid_forecast / "all_test_results_hourly.csv", index=True)
    logger.info(f"All test results saved to {save_path_bid_forecast / 'all_test_results_hourly.csv'}")


if __name__ == "__main__":
    main()
