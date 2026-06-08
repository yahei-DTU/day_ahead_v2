"""
File name: benchmark.py
Author: Yannick Heiser
Created: 2026-02-08
Last modified: 2026-05-07
Version: 2.0
Description:
    Run and evaluate the benchmark models for both HPP and wind-only portfolios.
Contact: yahei@dtu.dk
Dependencies: pandas, os, typing, pathlib, data_validation
"""

from pathlib import Path
import numpy as np
import pandas as pd
import hydra
import logging
import copy
from omegaconf import OmegaConf
from day_ahead_v2.optimization import (
    ModelHindsightHPP, ModelHindsightWindOnly,
    ModelSinglePolicyHPP, ModelSinglePolicyWindOnly,
)
from day_ahead_v2.data import PandasHandler
from day_ahead_v2.train import rolling_windows, feature_selection, split_features_target
from day_ahead_v2.evaluate import calculate_profit, calculate_hydrogen_balancing_bids
from day_ahead_v2.utils import electrolyzer_efficiency

logger = logging.getLogger(__name__)

_HINDSIGHT_CLASSES = {
    "hpp":       ModelHindsightHPP,
    "wind_only": ModelHindsightWindOnly,
}
_SINGLE_POLICY_CLASSES = {
    "hpp":       ModelSinglePolicyHPP,
    "wind_only": ModelSinglePolicyWindOnly,
}


def simulate_ar1_errors(index, phi, sigma, initial_error):
    errors = np.zeros(len(index))
    errors[0] = initial_error
    for i in range(1, len(index)):
        eps = np.random.normal(0, sigma)
        errors[i] = phi * errors[i-1] + eps
    return pd.Series(errors, index=index)


def _profit_summary(df):
    profit = df["profit"]
    return {
        "train_profit_total" if "train" in df.attrs.get("split", "") else "test_profit_total": profit.sum(),
    }


def _collect_results(results_df):
    profit = results_df["profit"]
    return {
        "profit_total": profit.sum(),
        "profit_mean":  profit.mean(),
        "profit_std":   profit.std(),
        "profit_cvar":  profit[profit <= np.percentile(profit, 5)].mean(),
    }


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config_dev")
def main(cfg):
    portfolio = cfg.experiments.optimization_parameters.portfolio
    is_hpp = portfolio == "hpp"
    h2_price = cfg.experiments.optimization_parameters.hydrogen_price if is_hpp else 0.0

    HindsightClass     = _HINDSIGHT_CLASSES[portfolio]
    SinglePolicyClass  = _SINGLE_POLICY_CLASSES[portfolio]

    # ---------------------------------------------
    # Import dataset
    # ---------------------------------------------
    raw_data_handler = PandasHandler(cfg)
    raw_data_handler = raw_data_handler.cut_data(
        cfg.experiments.experiment_parameters.start_date,
        cfg.experiments.experiment_parameters.end_date,
        cfg.datasets.training.datetime_column,
    )

    # ----------------------------------------------
    # Initialize the electrolyzer (HPP only)
    # ----------------------------------------------
    if is_hpp:
        electrolyzer_efficiency.initiate_HYP_L(cfg)

    # ---------------------------------------------
    # Rolling window backtest
    # ---------------------------------------------
    windows = list(rolling_windows(cfg))
    cfg_snapshot = copy.deepcopy(cfg)

    all_results_hindsight     = []
    all_results_policy        = []
    all_results_bid_forecast  = []
    all_results_bid_max       = []
    all_train_results_dfs_hindsight    = pd.DataFrame()
    all_test_results_dfs_hindsight     = pd.DataFrame()
    all_train_results_dfs_policy       = pd.DataFrame()
    all_test_results_dfs_policy        = pd.DataFrame()
    all_train_results_dfs_bid_forecast = pd.DataFrame()
    all_test_results_dfs_bid_forecast  = pd.DataFrame()
    all_train_results_dfs_bid_max      = pd.DataFrame()
    all_test_results_dfs_bid_max       = pd.DataFrame()

    for window in windows:

        train_start, train_end = window["train"]
        valid_start, valid_end = window["valid"]
        test_start,  test_end  = window["test"]

        logger.info(
            f"Train: {train_start.date()} → {train_end.date()} | "
            f"Valid: {valid_start.date()} → {valid_end.date()} | "
            f"Test: {test_start.date()} → {test_end.date()}"
        )

        cfg = copy.deepcopy(cfg_snapshot)
        data_handler = raw_data_handler.transform_data(cfg)

        nan_counts = data_handler.data.isnull().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"Found NaN values in data:\n{nan_counts[nan_counts > 0]}")
        else:
            logger.info("No NaN values found in data")

        if cfg.experiments.train_parameters.get("feature_selection", False) and window == windows[0]:
            selected_features = feature_selection(cfg, data_handler.data, start=train_start, end=train_end)
            cfg.datasets.training.feature_columns_flex = selected_features
            cfg_snapshot.datasets.training.feature_columns_flex = selected_features

        # ---------------------------------------------
        # Data loading
        # ---------------------------------------------
        data_train = data_handler.cut_data(train_start, train_end, cfg.datasets.training.datetime_column)
        X_train, y_train = split_features_target(cfg, data_train.data, sanitize_names=True)
        logger.info(f"Target counts in training data:\n{y_train.value_counts()}")

        data_valid = data_handler.cut_data(valid_start, valid_end, cfg.datasets.training.datetime_column)
        X_val, y_val = split_features_target(cfg, data_valid.data, sanitize_names=True)

        data_test = data_handler.cut_data(test_start, test_end, cfg.datasets.training.datetime_column)
        X_test, y_test = split_features_target(cfg, data_test.data, sanitize_names=True)

        data_train_full = pd.concat([data_train.data, data_valid.data])
        X_train_full = pd.concat([X_train, X_val])
        datetime_index_train = data_train_full.index
        datetime_index_test  = data_test.data.index
        del data_train_full, data_valid
        data_train = data_handler.data.loc[datetime_index_train]
        data_test  = data_handler.data.loc[datetime_index_test]

        lambda_DA_hat_train = data_train[cfg.datasets.optimization.lambda_DA_hat]
        lambda_B_hat_train  = data_train[cfg.datasets.optimization.lambda_B_hat]
        P_W_hat_train       = data_train[cfg.datasets.optimization.P_W_hat]
        P_W_tilde_train     = data_train[cfg.datasets.optimization.P_W_tilde]
        lambda_DA_hat_test  = data_test[cfg.datasets.optimization.lambda_DA_hat]
        lambda_B_hat_test   = data_test[cfg.datasets.optimization.lambda_B_hat]
        P_W_hat_test        = data_test[cfg.datasets.optimization.P_W_hat]
        P_W_tilde_test      = data_test[cfg.datasets.optimization.P_W_tilde]

        # ------------------------------------- Model 1 - Perfect foresight -------------------------------------
        logger.debug("Running Hindsight Model...")
        hindsight_train = HindsightClass(
            cfg=cfg,
            lambda_DA_hat=lambda_DA_hat_train,
            lambda_B_hat=lambda_B_hat_train,
            P_W_hat=P_W_hat_train,
        )
        hindsight_train.build_model()
        hindsight_train.run_optimization()
        if hindsight_train.results.status != "ok":
            logger.error(f"Hindsight train optimization did not converge. Status: {hindsight_train.results.status}")

        hindsight_test = HindsightClass(
            cfg=cfg,
            lambda_DA_hat=lambda_DA_hat_test,
            lambda_B_hat=lambda_B_hat_test,
            P_W_hat=P_W_hat_test,
        )
        hindsight_test.build_model()
        hindsight_test.run_optimization()
        if hindsight_test.results.status != "ok":
            logger.error(f"Hindsight test optimization did not converge. Status: {hindsight_test.results.status}")

        p_DA_train = pd.Series(hindsight_train.results.p_DA, index=data_train.index)
        p_B_train  = pd.Series(hindsight_train.results.p_B,  index=data_train.index)
        p_H_train  = pd.Series(hindsight_train.results.p_H,  index=data_train.index) if is_hpp else pd.Series(0.0, index=data_train.index)
        h_train    = pd.Series(hindsight_train.results.h,    index=data_train.index) if is_hpp else pd.Series(0.0, index=data_train.index)
        p_DA_test  = pd.Series(hindsight_test.results.p_DA,  index=data_test.index)
        p_B_test   = pd.Series(hindsight_test.results.p_B,   index=data_test.index)
        p_H_test   = pd.Series(hindsight_test.results.p_H,   index=data_test.index) if is_hpp else pd.Series(0.0, index=data_test.index)
        h_test     = pd.Series(hindsight_test.results.h,     index=data_test.index) if is_hpp else pd.Series(0.0, index=data_test.index)

        profit_train = calculate_profit(p_DA_train, h_train, p_B_train, lambda_DA_hat_train, h2_price, lambda_B_hat_train)
        profit_test  = calculate_profit(p_DA_test,  h_test,  p_B_test,  lambda_DA_hat_test,  h2_price, lambda_B_hat_test)

        results_hindsight_train_df = pd.DataFrame({
            "p_DA": p_DA_train, "p_B": p_B_train, "p_H": p_H_train, "h": h_train,
            "profit": profit_train, "P_W_hat": P_W_hat_train, "P_W_tilde": P_W_tilde_train,
            "lambda_DA_hat": lambda_DA_hat_train, "lambda_B_hat": lambda_B_hat_train,
        }, index=data_train.index)
        results_hindsight_test_df = pd.DataFrame({
            "p_DA": p_DA_test, "p_B": p_B_test, "p_H": p_H_test, "h": h_test,
            "profit": profit_test, "P_W_hat": P_W_hat_test, "P_W_tilde": P_W_tilde_test,
            "lambda_DA_hat": lambda_DA_hat_test, "lambda_B_hat": lambda_B_hat_test,
        }, index=data_test.index)

        all_results_hindsight.append({
            "train_profit_total": profit_train.sum(), "train_profit_mean": profit_train.mean(),
            "train_profit_cvar": profit_train[profit_train >= np.percentile(profit_train, 95)].mean(),
            "test_profit_total":  profit_test.sum(),  "test_profit_mean":  profit_test.mean(),
            "test_profit_cvar": profit_test[profit_test >= np.percentile(profit_test, 95)].mean(),
            "train_start": train_start, "train_end": train_end,
            "valid_start": valid_start, "valid_end": valid_end,
            "test_start":  test_start,  "test_end":  test_end,
            "lambda_H": h2_price,
        })
        all_train_results_dfs_hindsight = pd.concat([all_train_results_dfs_hindsight, results_hindsight_train_df])
        all_test_results_dfs_hindsight  = pd.concat([all_test_results_dfs_hindsight,  results_hindsight_test_df])
        del p_DA_train, p_B_train, p_H_train, h_train, profit_train, p_DA_test, p_B_test, p_H_test, h_test, profit_test

        # ------------------------------------- Model 2 - Single Linear Policy -------------------------------------
        logger.debug("Running Single Linear Policy Model...")
        single_policy_train = SinglePolicyClass(
            cfg=cfg,
            lambda_DA_hat=lambda_DA_hat_train,
            lambda_B_hat=lambda_B_hat_train,
            P_W_hat=P_W_hat_train,
            P_W_tilde=P_W_tilde_train,
            X_features=X_train_full,
        )
        single_policy_train.build_model()
        single_policy_train.run_optimization()
        if single_policy_train.results.status != "ok":
            logger.error(f"Single policy optimization did not converge. Status: {single_policy_train.results.status}")

        p_DA_train = pd.Series(single_policy_train.results.p_DA, index=data_train.index)
        p_B_train  = pd.Series(single_policy_train.results.p_B,  index=data_train.index)
        p_H_train  = pd.Series(single_policy_train.results.p_H,  index=data_train.index) if is_hpp else pd.Series(0.0, index=data_train.index)
        h_train    = pd.Series(single_policy_train.results.h,    index=data_train.index) if is_hpp else pd.Series(0.0, index=data_train.index)

        test_bids  = single_policy_train.calculate_bids(cfg, data_test, X_test)
        p_DA_test, p_B_test = test_bids[0], test_bids[1]
        p_H_test   = test_bids[2] if is_hpp else pd.Series(0.0, index=p_DA_test.index)
        h_test     = test_bids[3] if is_hpp else pd.Series(0.0, index=p_DA_test.index)

        profit_train = calculate_profit(p_DA_train, h_train, p_B_train, lambda_DA_hat_train, h2_price, lambda_B_hat_train)
        profit_test  = calculate_profit(p_DA_test,  h_test,  p_B_test,  lambda_DA_hat_test,  h2_price, lambda_B_hat_test)

        results_policy_train_df = pd.DataFrame({
            "p_DA": p_DA_train, "p_B": p_B_train, "p_H": p_H_train, "h": h_train,
            "profit": profit_train, "P_W_hat": P_W_hat_train, "P_W_tilde": P_W_tilde_train,
            "lambda_DA_hat": lambda_DA_hat_train, "lambda_B_hat": lambda_B_hat_train,
        }, index=data_train.index)
        results_policy_test_df = pd.DataFrame({
            "p_DA": p_DA_test, "p_B": p_B_test, "p_H": p_H_test, "h": h_test,
            "profit": profit_test, "P_W_hat": P_W_hat_test, "P_W_tilde": P_W_tilde_test,
            "lambda_DA_hat": lambda_DA_hat_test, "lambda_B_hat": lambda_B_hat_test,
        }, index=data_test.index)

        all_results_policy.append({
            "train_profit_total": profit_train.sum(), "train_profit_mean": profit_train.mean(),
            "train_profit_cvar": profit_train[profit_train >= np.percentile(profit_train, 95)].mean(),
            "test_profit_total":  profit_test.sum(),  "test_profit_mean":  profit_test.mean(),
            "test_profit_cvar": profit_test[profit_test >= np.percentile(profit_test, 95)].mean(),
            "train_start": train_start, "train_end": train_end,
            "valid_start": valid_start, "valid_end": valid_end,
            "test_start":  test_start,  "test_end":  test_end,
            "lambda_H": h2_price,
        })
        all_train_results_dfs_policy = pd.concat([all_train_results_dfs_policy, results_policy_train_df])
        all_test_results_dfs_policy  = pd.concat([all_test_results_dfs_policy,  results_policy_test_df])
        del p_DA_train, p_B_train, p_H_train, h_train, profit_train, p_DA_test, p_B_test, p_H_test, h_test, profit_test

        # ------------------------------------- Model 3 - Bid Forecast -------------------------------------
        logger.debug("Running Bid Forecast Model...")
        if is_hpp:
            p_DA_train = P_W_tilde_train.copy()
            p_DA_train[lambda_DA_hat_train < 0] = -cfg.experiments.optimization_parameters.electrolyzer_capacity
            p_B_train, p_H_train, h_train = calculate_hydrogen_balancing_bids(cfg, p_DA_train, lambda_B_hat_train, P_W_hat_train)
            p_DA_test = P_W_tilde_test.copy()
            p_DA_test[lambda_DA_hat_test < 0] = -cfg.experiments.optimization_parameters.electrolyzer_capacity
            p_B_test, p_H_test, h_test = calculate_hydrogen_balancing_bids(cfg, p_DA_test, lambda_B_hat_test, P_W_hat_test)
        else:
            p_DA_train = P_W_tilde_train.copy()
            p_DA_train[lambda_DA_hat_train < 0] = 0.0
            p_B_train  = P_W_hat_train - p_DA_train
            p_H_train  = pd.Series(0.0, index=p_DA_train.index)
            h_train    = pd.Series(0.0, index=p_DA_train.index)
            p_DA_test  = P_W_tilde_test.copy()
            p_DA_test[lambda_DA_hat_test < 0] = 0.0
            p_B_test   = P_W_hat_test - p_DA_test
            p_H_test   = pd.Series(0.0, index=p_DA_test.index)
            h_test     = pd.Series(0.0, index=p_DA_test.index)

        profit_train = calculate_profit(p_DA_train, h_train, p_B_train, lambda_DA_hat_train, h2_price, lambda_B_hat_train)
        profit_test  = calculate_profit(p_DA_test,  h_test,  p_B_test,  lambda_DA_hat_test,  h2_price, lambda_B_hat_test)

        results_bid_forecast_train_df = pd.DataFrame({
            "p_DA": p_DA_train, "p_B": p_B_train, "p_H": p_H_train, "h": h_train,
            "profit": profit_train, "P_W_hat": P_W_hat_train, "P_W_tilde": P_W_tilde_train,
            "lambda_DA_hat": lambda_DA_hat_train, "lambda_B_hat": lambda_B_hat_train,
        }, index=data_train.index)
        results_bid_forecast_test_df = pd.DataFrame({
            "p_DA": p_DA_test, "p_B": p_B_test, "p_H": p_H_test, "h": h_test,
            "profit": profit_test, "P_W_hat": P_W_hat_test, "P_W_tilde": P_W_tilde_test,
            "lambda_DA_hat": lambda_DA_hat_test, "lambda_B_hat": lambda_B_hat_test,
        }, index=data_test.index)

        all_results_bid_forecast.append({
            "train_profit_total": profit_train.sum(), "train_profit_mean": profit_train.mean(),
            "train_profit_cvar": profit_train[profit_train >= np.percentile(profit_train, 95)].mean(),
            "test_profit_total":  profit_test.sum(),  "test_profit_mean":  profit_test.mean(),
            "test_profit_cvar": profit_test[profit_test >= np.percentile(profit_test, 95)].mean(),
            "train_start": train_start, "train_end": train_end,
            "valid_start": valid_start, "valid_end": valid_end,
            "test_start":  test_start,  "test_end":  test_end,
            "lambda_H": h2_price,
        })
        all_train_results_dfs_bid_forecast = pd.concat([all_train_results_dfs_bid_forecast, results_bid_forecast_train_df])
        all_test_results_dfs_bid_forecast  = pd.concat([all_test_results_dfs_bid_forecast,  results_bid_forecast_test_df])
        del p_DA_train, p_B_train, p_H_train, h_train, profit_train, p_DA_test, p_B_test, p_H_test, h_test, profit_test

        # ---------------------------------------------- Model 4 - Always bid max ----------------------------------------------
        logger.debug("Running Always Bid Max Model...")
        if is_hpp:
            p_DA_train = pd.Series(cfg.experiments.optimization_parameters.wind_capacity, index=data_train.index)
            p_DA_train[lambda_DA_hat_train < 0] = -cfg.experiments.optimization_parameters.electrolyzer_capacity
            p_B_train, p_H_train, h_train = calculate_hydrogen_balancing_bids(cfg, p_DA_train, lambda_B_hat_train, P_W_hat_train)
            p_DA_test = pd.Series(cfg.experiments.optimization_parameters.wind_capacity, index=data_test.index)
            p_DA_test[lambda_DA_hat_test < 0] = -cfg.experiments.optimization_parameters.electrolyzer_capacity
            p_B_test, p_H_test, h_test = calculate_hydrogen_balancing_bids(cfg, p_DA_test, lambda_B_hat_test, P_W_hat_test)
        else:
            p_DA_train = pd.Series(cfg.experiments.optimization_parameters.wind_capacity, index=data_train.index)
            p_DA_train[lambda_DA_hat_train < 0] = 0.0
            p_B_train  = P_W_hat_train - p_DA_train
            p_H_train  = pd.Series(0.0, index=p_DA_train.index)
            h_train    = pd.Series(0.0, index=p_DA_train.index)
            p_DA_test  = pd.Series(cfg.experiments.optimization_parameters.wind_capacity, index=data_test.index)
            p_DA_test[lambda_DA_hat_test < 0] = 0.0
            p_B_test   = P_W_hat_test - p_DA_test
            p_H_test   = pd.Series(0.0, index=p_DA_test.index)
            h_test     = pd.Series(0.0, index=p_DA_test.index)

        profit_train = calculate_profit(p_DA_train, h_train, p_B_train, lambda_DA_hat_train, h2_price, lambda_B_hat_train)
        profit_test  = calculate_profit(p_DA_test,  h_test,  p_B_test,  lambda_DA_hat_test,  h2_price, lambda_B_hat_test)

        results_bid_max_train_df = pd.DataFrame({
            "p_DA": p_DA_train, "p_B": p_B_train, "p_H": p_H_train, "h": h_train,
            "profit": profit_train, "P_W_hat": P_W_hat_train, "P_W_tilde": P_W_tilde_train,
            "lambda_DA_hat": lambda_DA_hat_train, "lambda_B_hat": lambda_B_hat_train,
        }, index=data_train.index)
        results_bid_max_test_df = pd.DataFrame({
            "p_DA": p_DA_test, "p_B": p_B_test, "p_H": p_H_test, "h": h_test,
            "profit": profit_test, "P_W_hat": P_W_hat_test, "P_W_tilde": P_W_tilde_test,
            "lambda_DA_hat": lambda_DA_hat_test, "lambda_B_hat": lambda_B_hat_test,
        }, index=data_test.index)

        all_results_bid_max.append({
            "train_profit_total": profit_train.sum(), "train_profit_mean": profit_train.mean(),
            "train_profit_cvar": profit_train[profit_train >= np.percentile(profit_train, 95)].mean(),
            "test_profit_total":  profit_test.sum(),  "test_profit_mean":  profit_test.mean(),
            "test_profit_cvar": profit_test[profit_test >= np.percentile(profit_test, 95)].mean(),
            "train_start": train_start, "train_end": train_end,
            "valid_start": valid_start, "valid_end": valid_end,
            "test_start":  test_start,  "test_end":  test_end,
            "lambda_H": h2_price,
        })
        all_train_results_dfs_bid_max = pd.concat([all_train_results_dfs_bid_max, results_bid_max_train_df])
        all_test_results_dfs_bid_max  = pd.concat([all_test_results_dfs_bid_max,  results_bid_max_test_df])
        del p_DA_train, p_B_train, p_H_train, h_train, profit_train, p_DA_test, p_B_test, p_H_test, h_test, profit_test

    # ---------------------------------------------
    # Aggregate metrics
    # ---------------------------------------------
    def _agg_split(train_df, test_df):
        def _stats(p):
            return p.sum(), p.mean(), p.std(), p[p <= np.percentile(p, 5)].mean()

        tr_tot, tr_mean, tr_std, tr_cvar = _stats(train_df["profit"])
        te_tot, te_mean, te_std, te_cvar = _stats(test_df["profit"])

        return {
            "train_profit_total": tr_tot,
            "train_profit_mean":  f"{tr_mean:.2f} ± {tr_std:.2f}",
            "train_profit_cvar":  tr_cvar,
            "test_profit_total":  te_tot,
            "test_profit_mean":   f"{te_mean:.2f} ± {te_std:.2f}",
            "test_profit_cvar":   te_cvar,
        }

    avg_metrics_hindsight    = _agg_split(all_train_results_dfs_hindsight,    all_test_results_dfs_hindsight)
    avg_metrics_policy       = _agg_split(all_train_results_dfs_policy,       all_test_results_dfs_policy)
    avg_metrics_bid_forecast = _agg_split(all_train_results_dfs_bid_forecast, all_test_results_dfs_bid_forecast)
    avg_metrics_bid_max      = _agg_split(all_train_results_dfs_bid_max,      all_test_results_dfs_bid_max)

    # ---------------------------------------------
    # Save results
    # ---------------------------------------------
    project_root = Path(cfg.project_root)
    save_path = project_root / "reports" / cfg.experiments.experiment_name

    for name, results_list, avg_metrics, train_df, test_df in [
        ("hindsight",     all_results_hindsight,    avg_metrics_hindsight,    all_train_results_dfs_hindsight,    all_test_results_dfs_hindsight),
        ("single_policy", all_results_policy,       avg_metrics_policy,       all_train_results_dfs_policy,       all_test_results_dfs_policy),
        ("bid_forecast",  all_results_bid_forecast, avg_metrics_bid_forecast, all_train_results_dfs_bid_forecast, all_test_results_dfs_bid_forecast),
        ("bid_max",       all_results_bid_max,      avg_metrics_bid_max,      all_train_results_dfs_bid_max,      all_test_results_dfs_bid_max),
    ]:
        sp = save_path / name / cfg.datasets.dataset_name
        sp.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, sp / "config.yaml")
        pd.DataFrame(results_list).to_csv(sp / "backtest_results.csv", index=False)
        with open(sp / "allwindows_metrics.txt", "w") as f:
            for key, value in avg_metrics.items():
                f.write(f"{key}: {value}\n")
        train_df.to_csv(sp / "all_train_results_hourly.csv", index=True)
        test_df.to_csv(sp / "all_test_results_hourly.csv",  index=True)
        logger.info(f"Results saved to {sp}")


if __name__ == "__main__":
    main()
