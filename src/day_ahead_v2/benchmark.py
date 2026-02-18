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
import pandas as pd
import hydra
import logging
from day_ahead_v2.optimization import ModelHindsight
from day_ahead_v2.data import DataHandler
from day_ahead_v2.train import rolling_windows
from day_ahead_v2.evaluate import calculate_profit

logger = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config_dev")
def main(cfg):
    # ---------------------------------------------
    # Import dataset
    # ---------------------------------------------
    data_handler = DataHandler(cfg)
    data_handler = data_handler.cut_data(cfg.experiments.experiment_parameters.start_date,
                                        cfg.experiments.experiment_parameters.end_date,
                                        cfg.datasets.training.datetime_column,
                                        )
    data_handler = data_handler.transform_data(cfg)

    # Check for NaNs in data
    nan_counts = data_handler.data.isnull().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"Found NaN values in data:\n{nan_counts[nan_counts > 0]}")
    else:
        logger.info("No NaN values found in data")

    # ---------------------------------------------
    # Rolling window backtest
    # ---------------------------------------------
    all_results_hindsight = []
    all_results_bid_forecast = []
    all_train_results_dfs_hindsight = pd.DataFrame()
    all_test_results_dfs_hindsight = pd.DataFrame()
    all_train_results_dfs_bid_forecast = pd.DataFrame()
    all_test_results_dfs_bid_forecast = pd.DataFrame()

    for window in rolling_windows(cfg):

        train_start, train_end = window["train"]
        valid_start, valid_end = window["valid"]
        test_start, test_end   = window["test"]

        logger.info(
            f"Train: {train_start.date()} → {train_end.date()} | "
            f"Valid: {valid_start.date()} → {valid_end.date()} | "
            f"Test: {test_start.date()} → {test_end.date()}"
        )
        # ---------------------------------------------
        # Data loading
        # ---------------------------------------------
        logger.debug("Cutting train data...")
        data_train = data_handler.cut_data(train_start, train_end, cfg.datasets.training.datetime_column)

        logger.debug("Cutting validation data...")
        data_valid = data_handler.cut_data(valid_start, valid_end, cfg.datasets.training.datetime_column)

        logger.debug("Cutting test data...")
        data_test = data_handler.cut_data(test_start, test_end, cfg.datasets.training.datetime_column)

        X_train_full = pd.concat([data_train.data, data_valid.data])
        datetime_index_train = X_train_full.index
        datetime_index_test = data_test.data.index
        del X_train_full, data_valid
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
        DA_bids_train = pd.Series(hindsight_model_train.results.p_DA, index=data_train.index)
        IM_bids_train = pd.Series(hindsight_model_train.results.delta_p, index=data_train.index)
        DA_bids_test = pd.Series(hindsight_model_test.results.p_DA, index=data_test.index)
        IM_bids_test = pd.Series(hindsight_model_test.results.delta_p, index=data_test.index)
        profit_train = calculate_profit(DA_bids_train, IM_bids_train, lambda_DA_hat_train, lambda_B_hat_train)
        profit_test = calculate_profit(DA_bids_test, IM_bids_test, lambda_DA_hat_test, lambda_B_hat_test)
        # Add bids and profit to results dfs
        results_hindsight_train_df = pd.DataFrame({
            "DA_bid": DA_bids_train,
            "IM_bid": IM_bids_train,
            "profit": profit_train
        }, index=data_train.index)
        results_hindsight_test_df = pd.DataFrame({
            "DA_bid": DA_bids_test,
            "IM_bid": IM_bids_test,
            "profit": profit_test
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
        }
        all_results_hindsight.append(results_hindsight)
        all_train_results_dfs_hindsight = pd.concat([all_train_results_dfs_hindsight, results_hindsight_train_df])
        all_test_results_dfs_hindsight = pd.concat([all_test_results_dfs_hindsight, results_hindsight_test_df])
        # ------------------------------------- Model 2 - Bid Forecast -------------------------------------
        logger.debug("Running Bid Forecast Model...")
        DA_bids_train = pd.Series(index=data_train.index, dtype=float)
        IM_bids_train = pd.Series(index=data_train.index, dtype=float)
        for t in data_train.index:
            # Check for negative prices and handle accordingly
            if lambda_DA_hat_train.loc[t] < 0:
                logger.warning(f"Negative price detected at time {t}: lambda_DA_hat={lambda_DA_hat_train.loc[t]}. Setting DA bid to 0.")
                DA_bids_train.loc[t] = 0.0
                IM_bids_train.loc[t] = P_W_hat_train.loc[t] - DA_bids_train.loc[t]
                if lambda_B_hat_train.loc[t] < 0:
                    logger.warning(f"Negative imbalance price detected at time {t}: lambda_B_hat={lambda_B_hat_train.loc[t]}. Setting IM bid to 0.")
                    IM_bids_train.loc[t] = 0.0
                continue
            DA_bids_train.loc[t] = P_W_tilde_train.loc[t]
            IM_bids_train.loc[t] = P_W_hat_train.loc[t] - DA_bids_train.loc[t]

        for t in data_test.index:
            # Check for negative prices and handle accordingly
            if lambda_DA_hat_test.loc[t] < 0:
                logger.warning(f"Negative price detected at time {t}: lambda_DA_hat={lambda_DA_hat_test.loc[t]}. Setting DA bid to 0.")
                DA_bids_test.loc[t] = 0.0
                IM_bids_test.loc[t] = P_W_hat_test.loc[t] - DA_bids_test.loc[t]
                if lambda_B_hat_test.loc[t] < 0:
                    logger.warning(f"Negative imbalance price detected at time {t}: lambda_B_hat={lambda_B_hat_test.loc[t]}. Setting IM bid to 0.")
                    IM_bids_test.loc[t] = 0.0
                continue
            DA_bids_test.loc[t] = P_W_tilde_test.loc[t]
            IM_bids_test.loc[t] = P_W_hat_test.loc[t] - DA_bids_test.loc[t]
        profit_train = calculate_profit(DA_bids_train, IM_bids_train, lambda_DA_hat_train, lambda_B_hat_train)
        profit_test = calculate_profit(DA_bids_test, IM_bids_test, lambda_DA_hat_test, lambda_B_hat_test)
        # Add bids and profit to results dfs
        results_bid_forecast_train_df = pd.DataFrame({
            "DA_bid": DA_bids_train,
            "IM_bid": IM_bids_train,
            "profit": profit_train,
            "P_W_tilde": P_W_tilde_train,
            "P_W_hat": P_W_hat_train,
            "lambda_DA_hat": lambda_DA_hat_train,
            "lambda_B_hat": lambda_B_hat_train
        }, index=data_train.index)
        results_bid_forecast_test_df = pd.DataFrame({
            "DA_bid": DA_bids_test,
            "IM_bid": IM_bids_test,
            "profit": profit_test,
            "P_W_tilde": P_W_tilde_test,
            "P_W_hat": P_W_hat_test,
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
        }
        all_results_bid_forecast.append(results_bid_forecast)
        all_train_results_dfs_bid_forecast = pd.concat([all_train_results_dfs_bid_forecast, results_bid_forecast_train_df])
        all_test_results_dfs_bid_forecast = pd.concat([all_test_results_dfs_bid_forecast, results_bid_forecast_test_df])

    # Save results
    total_profit_train_hindsight = all_train_results_dfs_hindsight["profit"].sum()
    mean_profit_train_hindsight = all_train_results_dfs_hindsight["profit"].mean()
    total_profit_test_hindsight = all_test_results_dfs_hindsight["profit"].sum()
    mean_profit_test_hindsight = all_test_results_dfs_hindsight["profit"].mean()
    total_profit_train_bid_forecast = all_train_results_dfs_bid_forecast["profit"].sum()
    mean_profit_train_bid_forecast = all_train_results_dfs_bid_forecast["profit"].mean()
    total_profit_test_bid_forecast = all_test_results_dfs_bid_forecast["profit"].sum()
    mean_profit_test_bid_forecast = all_test_results_dfs_bid_forecast["profit"].mean()

    avg_metrics_hindsight = {
        "train_profit_total": total_profit_train_hindsight,
        "train_profit_mean": mean_profit_train_hindsight,
        "test_profit_total": total_profit_test_hindsight,
        "test_profit_mean": mean_profit_test_hindsight,
    }
    avg_metrics_bid_forecast = {
        "train_profit_total": total_profit_train_bid_forecast,
        "train_profit_mean": mean_profit_train_bid_forecast,
        "test_profit_total": total_profit_test_bid_forecast,
        "test_profit_mean": mean_profit_test_bid_forecast,
    }

    results_hindsight_df = pd.DataFrame(all_results_hindsight)
    results_bid_forecast_df = pd.DataFrame(all_results_bid_forecast)
    # Save results to CSV
    save_path = Path(__file__).resolve().parent.parent.parent / "reports" / cfg.experiments.experiment_name
    save_path_hindsight = save_path / "hindsight" / cfg.datasets.dataset_name
    save_path_hindsight.mkdir(parents=True, exist_ok=True)
    results_hindsight_df.to_csv(save_path_hindsight / "backtest_results.csv", index=False)
    with open(save_path_hindsight / "allwindows_metrics.txt", "w") as f:
            for key, value in avg_metrics_hindsight.items():
                f.write(f"{key}: {value}\n")

    save_path_bid_forecast = save_path / "bid_forecast" / cfg.datasets.dataset_name
    save_path_bid_forecast.mkdir(parents=True, exist_ok=True)
    results_bid_forecast_df.to_csv(save_path_bid_forecast / "backtest_results.csv", index=False)
    with open(save_path_bid_forecast / "allwindows_metrics.txt", "w") as f:
            for key, value in avg_metrics_bid_forecast.items():
                f.write(f"{key}: {value}\n")

    # Save all test results to a CSV file
    all_test_results_dfs_bid_forecast.to_csv(save_path_bid_forecast / "all_test_results_hourly.csv", index=True)
    logger.info(f"All test results saved to {save_path_bid_forecast / 'all_test_results_hourly.csv'}")
    all_test_results_dfs_hindsight.to_csv(save_path_hindsight / "all_test_results_hourly.csv", index=True)
    logger.info(f"All test results saved to {save_path_hindsight / 'all_test_results_hourly.csv'}")


if __name__ == "__main__":
    main()
