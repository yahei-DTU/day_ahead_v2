import linopy
import numpy as np
import pandas as pd
import logging
from omegaconf import DictConfig
from types import SimpleNamespace
import xarray as xr
import hydra
from day_ahead_v2.evaluate import calculate_hydrogen_balancing_bids

logger = logging.getLogger(__name__)

class ModelHindsight:
    """Model with perfect foresight of day-ahead prices, balancing prices, and wind power generation."""
    def __init__(self, cfg: DictConfig, lambda_DA_hat: pd.Series, lambda_B_hat: pd.Series, P_W_hat: pd.Series, **kwargs) -> None:
        """Initializes the HindsightModel with given parameters.

        Args:
            cfg: Configuration object containing experiment parameters.
            lambda_DA_hat (pd.Series): Actual day-ahead electricity prices.
            lambda_B_hat (pd.Series): Actual balancing electricity prices.
            P_W_hat (pd.Series): Actual wind power generation.
        """
        # Containers for variables, constraints, and results
        self.parameters = SimpleNamespace()
        self.constants = SimpleNamespace()
        self.results = SimpleNamespace()
        self.results.objective_value = None # set objective value to None until optimization is performed

        assert lambda_DA_hat.index.equals(lambda_B_hat.index)
        assert lambda_DA_hat.index.equals(P_W_hat.index)

        # Constants
        self.constants.T = lambda_DA_hat.index
        self.constants.D = lambda_DA_hat.index.floor("D").to_numpy()
        self.constants.DAY_INDEX, self.constants.DAY_VALUES = pd.factorize(
            lambda_DA_hat.index.normalize()
        )
        self.constants.DAYS = range(len(self.constants.DAY_VALUES))
        self.constants.P_W_BAR = cfg.experiments.optimization_parameters.wind_capacity
        self.constants.P_H_BAR = cfg.experiments.optimization_parameters.electrolyzer_capacity
        self.constants.H2_PRICE = cfg.experiments.optimization_parameters.hydrogen_price
        self.constants.ELECTROLYZER_LOAD_MIN = cfg.experiments.optimization_parameters.electrolyzer_load_min*cfg.experiments.optimization_parameters.electrolyzer_capacity
        self.constants.A = cfg.experiments.optimization_parameters.a # List of linear segments for hydrogen production function (kg/MWh)
        self.constants.B = cfg.experiments.optimization_parameters.b # List of intercepts for hydrogen production function (kg)
        self.constants.SEGMENTS = range(len(self.constants.A))
        self.constants.H2_MIN = cfg.experiments.optimization_parameters.minimum_daily_hydrogen

        # Parameters
        self.parameters.day_index = xr.DataArray(
            self.constants.DAY_INDEX,
            dims=["datetime"],
            coords={"datetime": self.constants.T}
        )

        self.parameters.lambda_DA_hat = xr.DataArray(
            lambda_DA_hat,
            dims=["datetime"],
            coords={
                "datetime": self.constants.T,
                "day": ("datetime", self.constants.D)
            }
        )
        self.parameters.lambda_B_hat = xr.DataArray(
            lambda_B_hat,
            dims=["datetime"],
            coords={
                "datetime": self.constants.T,
                "day": ("datetime", self.constants.D)
            }
        )
        self.parameters.P_W_hat = xr.DataArray(
            P_W_hat,
            dims=["datetime"],
            coords={
                "datetime": self.constants.T,
                "day": ("datetime", self.constants.D)
            }
        )
        self.parameters.A = xr.DataArray(
            self.constants.A,
            dims=["segment"],
            coords={"segment": self.constants.SEGMENTS}
        )
        self.parameters.B = xr.DataArray(
            self.constants.B,
            dims=["segment"],
            coords={"segment": self.constants.SEGMENTS}
        )

        # Create optimization model
        self.model = linopy.Model()

    def _set_variables(self) -> None:
        """Sets the decision variables for the optimization model."""
        self.model.add_variables(
            lower=-self.constants.P_H_BAR,
            upper=self.constants.P_W_BAR,
            dims=["datetime"],
            coords={"datetime": self.constants.T,
                    "day": ("datetime", self.constants.D)},
            name="p_DA"
        )
        self.model.add_variables(
            dims = ["datetime"],
            coords = {"datetime": self.constants.T,
                     "day": ("datetime", self.constants.D)},
            name = "p_B"
        )
        self.model.add_variables(
            lower=self.constants.ELECTROLYZER_LOAD_MIN,
            upper=self.constants.P_H_BAR,
            dims=["datetime"],
            coords={"datetime": self.constants.T,
                    "day": ("datetime", self.constants.D)},
            name="p_H"
        )
        self.model.add_variables(
            lower=0,
            dims=["datetime"],
            coords={"datetime": self.constants.T,
                    "day": ("datetime", self.constants.D)},
            name="h"
        )

    def _set_objective(self) -> None:
        """Sets the objective function for the optimization model."""
        self.model.add_objective(
            (self.model.variables["p_DA"] * self.parameters.lambda_DA_hat).sum()
            + (self.model.variables["p_B"] * self.parameters.lambda_B_hat).sum()
            + (self.model.variables["h"] * self.constants.H2_PRICE).sum(),
            sense="max"
        )

    def _set_constraints(self) -> None:
        """Sets the constraints for the optimization model."""
        self.model.add_constraints(
            self.model.variables["p_DA"] + self.model.variables["p_B"] + self.model.variables["p_H"] == self.parameters.P_W_hat,
            name="power_balance"
        )
        self.model.add_constraints(
            self.model.variables["h"]
            <= self.parameters.A * self.model.variables["p_H"]
            + self.parameters.B,
            name="hydrogen_production"
        )
        for d in self.constants.DAYS:
            mask = self.constants.DAY_INDEX == d
            self.model.add_constraints(
                self.model.variables["h"][mask].sum() >= self.constants.H2_MIN,
                name=f"daily_hydrogen_{d}"
            )

    def build_model(self) -> None:
        """Builds the optimization model by setting variables, objective, and constraints."""
        self._set_variables()
        self._set_objective()
        self._set_constraints()
        # Log model properties for debugging
        logger.info("Built model with following properties:")
        # Variables
        for var_name in self.model.variables:
            var = self.model.variables[var_name]
            logger.info(f"\tVariable: {var_name}, dims={var.dims}, shape={var.shape}")
        # Constraints
        for con_name, con in self.model.constraints.items():
            # Use .sizes if available
            sizes = getattr(con, "sizes", None)
            if sizes is not None:
                dims_str = ", ".join(f"{k}:{v}" for k, v in sizes.items())
            else:
                dims_str = "unknown"
            logger.info(f"\tConstraint: {con_name}, dims={dims_str}")

    def _solve(self, solver_name="highs") -> None:
        """Solves the optimization model using the specified solver."""
        return self.model.solve(solver_name=solver_name)

    def _save_results(self) -> None:
        """Saves the results of the optimization model."""
        self.results.status = self.model.status
        if self.results.status != "ok":
            logger.warning(f"Optimization did not reach optimality. Status: {self.results.status}")
        logger.info(f"Optimization status: {self.results.status}")
        self.results.objective_value = self.model.objective.value
        self.results.p_DA = self.model.variables["p_DA"].solution.to_pandas()
        self.results.p_B = self.model.variables["p_B"].solution.to_pandas()
        self.results.p_H = self.model.variables["p_H"].solution.to_pandas()
        self.results.h = self.model.variables["h"].solution.to_pandas()

    def run_optimization(self, solver_name="highs") -> None:
        """Runs the optimization process: solves the model and saves results."""
        logger.info("Starting optimization...")
        try:
            nonzeros = np.abs(self.model.matrices.A.data)
            significant = nonzeros[nonzeros > 1e-10]  # filter floating-point near-zeros
            if len(significant) > 0:
                coef_min, coef_max = significant.min(), significant.max()
                ratio = coef_max / coef_min
                logger.info(f"LP coefficient range: [{coef_min:.2e}, {coef_max:.2e}], ratio: {ratio:.2e}")
                if ratio > 1e6:
                    logger.warning(
                        f"Extreme coefficient range detected (ratio={ratio:.2e}). "
                        "LP may be numerically unstable — consider scaling input features."
                    )
                    if hasattr(self, "parameters") and hasattr(self.parameters, "X_features"):
                        X_vals = np.abs(self.parameters.X_features.values)  # shape (T, F)
                        feature_names = list(self.parameters.X_features.coords["feature"].values)
                        rows = []
                        for i, name in enumerate(feature_names):
                            col = X_vals[:, i]
                            nz = col[col > 1e-10]
                            if len(nz) > 0:
                                rows.append((name, nz.min(), col.max()))
                        rows.sort(key=lambda x: x[1])  # sort by min ascending
                        logger.warning("Feature coefficient ranges (sorted by smallest value):")
                        for name, mn, mx in rows:
                            logger.warning(f"  {name}: min={mn:.2e}, max={mx:.2e}")
        except Exception as e:
            logger.debug(f"Could not compute coefficient range: {e}")
        self._solve(solver_name=solver_name)
        self._save_results()
        logger.info("Optimization completed.")

class ModelSinglePolicy(ModelHindsight):
    """Model that uses a single bid adjustment policy (z) for all timestamps, without differentiating by predicted labels."""
    def __init__(
            self,
            cfg: DictConfig,
            lambda_DA_hat: pd.Series,
            lambda_B_hat: pd.Series,
            P_W_hat: pd.Series,
            P_W_tilde: pd.Series,
            X_features: pd.DataFrame, # shape: (T, num_features)
            **kwargs,
        ):
        super().__init__(cfg, lambda_DA_hat, lambda_B_hat, P_W_hat, **kwargs)
        assert X_features.index.equals(lambda_DA_hat.index)
        assert lambda_DA_hat.index.equals(P_W_tilde.index)
        self.parameters.P_W_tilde = xr.DataArray(
            P_W_tilde.values,
            dims=["datetime"],
            coords={
                "datetime": self.constants.T,
                "day": ("datetime", self.constants.D)
            }
        )
        X_features = X_features.copy()
        if "predicted_proba" in X_features.columns:
            X_features["predicted_proba"] = X_features["predicted_proba"].clip(lower=1e-3, upper=1 - 1e-3)
        X_features["intercept"] = 1.0
        X_features["lambda_DA_hat"] = lambda_DA_hat
        # Zero out near-zero values so they are dropped from the sparse LP matrix,
        # preventing extreme coefficient ranges from z-scored features near their mean.
        X_features = X_features.where(X_features.abs() >= 1e-4, other=0.0)
        feature_dim = X_features.columns.tolist()
        self.constants.FEATURE_DIM = feature_dim
        self.parameters.X_features = xr.DataArray(
            X_features.values,
            dims=["datetime", "feature"],
            coords={
                "datetime": self.constants.T,
                "day": ("datetime", self.constants.D),
                "feature": self.constants.FEATURE_DIM,
            },
        )

    def _set_variables(self):
        """Adds additional variables specific to the ModelSinglePolicy."""
        super()._set_variables()
        self.model.add_variables(
            name="z",
            dims=["datetime"],
            coords={"datetime": self.constants.T}
        )
        self.model.add_variables(
            name="q",
            dims=["feature"],
            coords={"feature": self.constants.FEATURE_DIM}
        )

    def _set_constraints(self):
        super()._set_constraints()
        X_T = self.parameters.X_features.transpose("feature", "datetime")
        self.model.add_constraints(
            self.model.variables["q"].dot(X_T) - self.model.variables["z"] == 0,
            name="linear_policy"
        )
        self.model.add_constraints(
            self.model.variables["p_DA"] - self.model.variables["z"] == self.parameters.P_W_tilde,
            name="bid_adjustment"
        )

    def _save_results(self):
        super()._save_results()
        self.results.z = self.model.variables["z"].solution.to_pandas()
        self.results.q = self.model.variables["q"].solution.to_pandas()

    def calculate_bids(self, cfg, data_df: pd.DataFrame, features: pd.DataFrame):
        """Calculate the DA bids and balancing/hydrogen bids using the trained policy."""
        lambda_max = 1000
        lambda_step = 2
        P_W_tilde = data_df[cfg.datasets.optimization.P_W_tilde]
        if P_W_tilde.isna().any():
            logger.error(f"{P_W_tilde.isna().sum()} values in P_W_tilde are NaN. Check data preprocessing.")
        P_W_hat = data_df[cfg.datasets.optimization.P_W_hat]
        lambda_DA_hat = data_df[cfg.datasets.optimization.lambda_DA_hat]
        lambda_B_hat = data_df[cfg.datasets.optimization.lambda_B_hat]
        features = features.copy()
        features["intercept"] = 1.0
        p_DA = pd.Series(index=data_df.index, dtype=float)
        bid_lambda_DA = np.arange(0, lambda_max, lambda_step)
        bid_p_DA = pd.DataFrame(
            np.zeros((len(data_df), len(bid_lambda_DA))),
            index=data_df.index,
            columns=bid_lambda_DA,
        )
        a_DA = self.results.q["lambda_DA_hat"]
        q = self.results.q.drop("lambda_DA_hat")

        for t in data_df.index:
            if lambda_DA_hat.loc[t] < 0:
                logger.warning(f"Negative price detected at time {t}: lambda_DA_hat={lambda_DA_hat.loc[t]}. Setting DA bid to 0.")
                p_DA.loc[t] = -cfg.experiments.optimization_parameters.electrolyzer_capacity
                continue

            b_DA = features.loc[t, :] @ q
            for k in bid_p_DA.columns:
                if bid_lambda_DA[k] > lambda_DA_hat.loc[t]:
                    p_DA.loc[t] = bid_p_DA.loc[t, k - lambda_step]
                    break
                bid_p_DA.loc[t, k] = np.maximum(
                    np.minimum(
                        P_W_tilde.loc[t] + a_DA * bid_lambda_DA[k] + b_DA,
                        cfg.experiments.optimization_parameters.wind_capacity,
                    ),
                    -cfg.experiments.optimization_parameters.electrolyzer_capacity,
                )

        if p_DA.isna().any():
            logger.error(f"{p_DA.isna().sum()} DA bids are NaN. Check calculate_bids method.")
            p_DA.fillna(P_W_tilde, inplace=True)

        p_B, p_H, h = calculate_hydrogen_balancing_bids(cfg, p_DA, lambda_B_hat, P_W_hat)
        return p_DA, p_B, p_H, h


class ModelClassPolicy(ModelHindsight):
    """Model that uses a linear policy for bid adjustments (z = X_features @ q)."""
    def __init__(
            self,
            cfg: DictConfig,
            lambda_DA_hat: pd.Series,
            lambda_B_hat: pd.Series,
            P_W_hat: pd.Series,
            P_W_tilde: pd.Series,
            X_features: pd.DataFrame, # shape: (T, num_features)
            pred_labels: pd.Series,
            **kwargs,
        ):
        super().__init__(cfg, lambda_DA_hat, lambda_B_hat, P_W_hat, **kwargs)
        assert X_features.index.equals(lambda_DA_hat.index)
        assert lambda_DA_hat.index.equals(P_W_tilde.index)
        assert X_features.index.equals(pred_labels.index)
        self.parameters.P_W_tilde = xr.DataArray(
            P_W_tilde.values,
            dims=["datetime"],
            coords={
                "datetime": self.constants.T,
                "day": ("datetime", self.constants.D)
            }
        )
        X_features = X_features.copy()
        if "predicted_proba" in X_features.columns:
            X_features["predicted_proba"] = X_features["predicted_proba"].clip(lower=1e-3, upper=1 - 1e-3)
        X_features["intercept"] = 1.0
        X_features["lambda_DA_hat"] = lambda_DA_hat
        # Zero out near-zero values so they are dropped from the sparse LP matrix,
        # preventing extreme coefficient ranges from z-scored features near their mean.
        X_features = X_features.where(X_features.abs() >= 1e-4, other=0.0)
        feature_dim = X_features.columns.tolist()
        self.constants.FEATURE_DIM = feature_dim
        self.parameters.X_features = xr.DataArray(
            X_features.values,
            dims=["datetime", "feature"],
            coords={
                "datetime": self.constants.T,
                "day": ("datetime", self.constants.D),
                "feature": self.constants.FEATURE_DIM,
            },
        )
        self.parameters.pred_labels = pred_labels

        opt_params = cfg.experiments.optimization_parameters
        self.constants.USE_CVAR = opt_params.get("use_cvar_constraint", False)
        if self.constants.USE_CVAR:
            self.constants.CVAR_ALPHA = opt_params.get("cvar_alpha", 0.95)
            self.constants.CVAR_LIMIT = opt_params.get("cvar_limit", 0.0)

    def _set_variables(self):
        """Adds additional variables specific to the ModelLinearPolicy."""
        super()._set_variables()
        self.model.add_variables(
            name="z",
            dims=["datetime"],
            coords={"datetime": self.constants.T}
        )

        self.model.add_variables(
            name="q_0",
            dims=["feature"],
            coords={"feature": self.constants.FEATURE_DIM}
        )
        self.model.add_variables(
            name="q_1",
            dims=["feature"],
            coords={"feature": self.constants.FEATURE_DIM}
        )
        self.model.add_variables(
            name="q_2",
            dims=["feature"],
            coords={"feature": self.constants.FEATURE_DIM}
        )

        if self.constants.USE_CVAR:
            self.model.add_variables(name="VaR")
            self.model.add_variables(
                lower=0,
                name="xi",
                dims=["datetime"],
                coords={"datetime": self.constants.T}
            )

    def _set_cvar_constraints(self):
        """Adds CVaR constraint on balancing cost using Rockafellar-Uryasev form.

        Enforces: VaR + 1/(T*(1-alpha)) * sum(xi) <= cvar_limit
        where xi[t] >= (lambda_DA_hat[t] - lambda_B_hat[t]) * p_B[t] - VaR, xi[t] >= 0.
        """
        T = len(self.constants.T)
        alpha = self.constants.CVAR_ALPHA
        price_diff = self.parameters.lambda_DA_hat - self.parameters.lambda_B_hat

        # xi[t] >= (lambda_DA_hat[t] - lambda_B_hat[t]) * p_B[t] - VaR
        self.model.add_constraints(
            self.model.variables["xi"] >= price_diff * self.model.variables["p_B"] - self.model.variables["VaR"],
            name="cvar_xi"
        )
        # VaR + 1/(T*(1-alpha)) * sum(xi) <= cvar_limit  (Rockafellar-Uryasev)
        self.model.add_constraints(
            self.model.variables["VaR"]
            + (1.0 / (T * (1.0 - alpha))) * self.model.variables["xi"].sum()
            <= self.constants.CVAR_LIMIT,
            name="cvar_limit"
        )
        logger.info(f"CVaR constraint added: alpha={alpha}, limit={self.constants.CVAR_LIMIT}")

    def _set_constraints(self):
        super()._set_constraints()

        # Masks for each label
        labels = [0, 1, 2]
        pred_labels = self.parameters.pred_labels
        q_vars = {0: self.model.variables["q_0"],
                1: self.model.variables["q_1"],
                2: self.model.variables["q_2"]}

        X_T = self.parameters.X_features.transpose("feature", "datetime")

        for label in labels:
            mask = (pred_labels == label).values  # boolean numpy array
            if mask.sum() == 0:
                continue  # skip if no timestamps with this label

            # Linear policy: z_t = q_label^T X_t
            self.model.add_constraints(
                q_vars[label].dot(X_T)[mask]
                - self.model.variables["z"][mask] == 0,
                name=f"linear_policy_{label}"
            )


        # Conditional z bounds
        self.model.add_constraints(
            self.model.variables["z"][(pred_labels == 0).values] <= 0,
            name="z_le_0"
        )
        self.model.add_constraints(
            self.model.variables["z"][(pred_labels == 1).values] >= 0,
            name="z_ge_0"
        )
        self.model.add_constraints(
            self.model.variables["z"][(pred_labels == 2).values] == 0,
            name="z_eq_0"
        )

        # Bid adjustment constraint
        self.model.add_constraints(
            self.model.variables["p_DA"] - self.model.variables["z"] == self.parameters.P_W_tilde,
            name="bid_adjustment"
        )

        if self.constants.USE_CVAR:
            self._set_cvar_constraints()

        logger.debug(f"X_features shape: {self.parameters.X_features.shape}")
        logger.debug(f"q shape: {self.model.variables['q_1'].shape}")
        logger.debug(f"z shape: {self.model.variables['z'].shape}")
        logger.debug(f"q@X shape: {self.model.variables['q_1'].dot(X_T).shape}")

    def _save_results(self):
        super()._save_results()
        self.results.z = self.model.variables["z"].solution.to_pandas()
        self.results.q_0 = self.model.variables["q_0"].solution.to_pandas()
        self.results.q_1 = self.model.variables["q_1"].solution.to_pandas()
        self.results.q_2 = self.model.variables["q_2"].solution.to_pandas()
        if self.constants.USE_CVAR:
            self.results.VaR = self.model.variables["VaR"].solution.item()
            self.results.xi = self.model.variables["xi"].solution.to_pandas()
            alpha = self.constants.CVAR_ALPHA
            T = len(self.constants.T)
            self.results.CVaR = self.results.VaR + (1.0 / (T * (1.0 - alpha))) * self.results.xi.sum()
            logger.info(f"CVaR solution: VaR={self.results.VaR:.4f}, CVaR={self.results.CVaR:.4f}")

    def calculate_bids(self, cfg, data_df: pd.DataFrame, features: pd.DataFrame, preds_df: pd.DataFrame):
        """Calculate the DA bids and balancing/hydrogen bids using the trained policy."""
        lambda_max = 1000
        lambda_step = 2
        P_W_tilde = data_df[cfg.datasets.optimization.P_W_tilde]
        if P_W_tilde.isna().any():
            logger.error(f"{P_W_tilde.isna().sum()} values in P_W_tilde are NaN. Check data preprocessing.")
        P_W_hat = data_df[cfg.datasets.optimization.P_W_hat]
        lambda_DA_hat = data_df[cfg.datasets.optimization.lambda_DA_hat]
        lambda_B_hat = data_df[cfg.datasets.optimization.lambda_B_hat]
        features = features.copy()
        features["intercept"] = 1.0
        p_DA = pd.Series(index=data_df.index, dtype=float)
        bid_lambda_DA = np.arange(0, lambda_max, lambda_step)
        bid_p_DA = pd.DataFrame(
            np.zeros((len(data_df), len(bid_lambda_DA))),
            index=data_df.index,
            columns=bid_lambda_DA,
        )
        q_map = {
            0: self.results.q_0.copy(),
            1: self.results.q_1.copy(),
            2: self.results.q_2.copy(),
        }

        for t in data_df.index:
            if lambda_DA_hat.loc[t] < 0:
                logger.warning(f"Negative price detected at time {t}: lambda_DA_hat={lambda_DA_hat.loc[t]}. Setting DA bid to 0.")
                p_DA.loc[t] = -cfg.experiments.optimization_parameters.electrolyzer_capacity
                continue

            pred_class = preds_df.loc[t, "thresholded_label"]
            q = q_map.get(pred_class)
            if q is None:
                logger.warning(f"Unknown class {pred_class} at time {t}. Skipping.")
                continue

            for k in bid_p_DA.columns:
                a_DA = q["lambda_DA_hat"]
                b_DA = features.loc[t, :] @ q.drop("lambda_DA_hat")

                if bid_lambda_DA[k] > lambda_DA_hat.loc[t]:
                    p_DA.loc[t] = bid_p_DA.loc[t, k - lambda_step]
                    break

                bid_p_DA.loc[t, k] = np.maximum(
                    np.minimum(
                        P_W_tilde.loc[t] + a_DA * bid_lambda_DA[k] + b_DA,
                        cfg.experiments.optimization_parameters.wind_capacity,
                    ),
                    -cfg.experiments.optimization_parameters.electrolyzer_capacity,
                )

        if p_DA.isna().any():
            logger.error(f"{p_DA.isna().sum()} DA bids are NaN. Check calculate_bids method.")
            p_DA.fillna(P_W_tilde, inplace=True)

        p_B, p_H, h = calculate_hydrogen_balancing_bids(cfg, p_DA, lambda_B_hat, P_W_hat)
        return p_DA, p_B, p_H, h


class ModelAllOrNothing(ModelHindsight):
    """Model that enforces all-or-nothing bidding"""
    def __init__(
            self,
            cfg: DictConfig,
            lambda_DA_hat: pd.Series,
            lambda_B_hat: pd.Series,
            P_W_hat: pd.Series,
            P_W_tilde: pd.Series,
            X_features: pd.DataFrame, # shape: (T, num_features)
            pred_labels: pd.Series,
            **kwargs,
        ):
        super().__init__(cfg, lambda_DA_hat, lambda_B_hat, P_W_hat, **kwargs)
        assert lambda_DA_hat.index.equals(pred_labels.index)
        assert lambda_DA_hat.index.equals(P_W_tilde.index)
        self.parameters.pred_labels = pred_labels
        self.parameters.P_W_tilde = xr.DataArray(
            P_W_tilde.values,
            dims=["datetime"],
            coords={
                "datetime": self.constants.T,
                "day": ("datetime", self.constants.D)
            }
        )

    def _set_constraints(self):
        super()._set_constraints()
        # Masks for each label
        labels = [0, 1, 2]
        pred_labels = self.parameters.pred_labels

        for label in labels:
            mask = (pred_labels == label).values
            if mask.sum() == 0:
                continue

            if label == 0:
                # p_DA = minimum bid
                self.model.add_constraints(
                    self.model.variables["p_DA"][mask] == -self.constants.P_H_BAR,
                    name="bid_min"
                )

            elif label == 1:
                # p_DA = maximum bid
                self.model.add_constraints(
                    self.model.variables["p_DA"][mask] == self.constants.P_W_BAR,
                    name="bid_max"
                )

            elif label == 2:
                # p_DA = wind forecast
                self.model.add_constraints(
                    self.model.variables["p_DA"][mask] == self.parameters.P_W_tilde[mask],
                    name="bid_forecast"
                )

    def calculate_bids(self, cfg, data_df: pd.DataFrame, features: pd.DataFrame, preds_df: pd.DataFrame):
        """Calculate the DA bids and balancing/hydrogen bids using all-or-nothing policy."""
        p_min = -cfg.experiments.optimization_parameters.electrolyzer_capacity
        p_max = cfg.experiments.optimization_parameters.wind_capacity
        lambda_DA_hat = data_df[cfg.datasets.optimization.lambda_DA_hat]
        P_W_hat = data_df[cfg.datasets.optimization.P_W_hat]
        lambda_B_hat = data_df[cfg.datasets.optimization.lambda_B_hat]
        P_W_tilde = data_df[cfg.datasets.optimization.P_W_tilde]

        labels = preds_df["thresholded_label"]
        p_DA = pd.Series(index=data_df.index, dtype=float)

        for t in data_df.index:
            if lambda_DA_hat.loc[t] < 0:
                logger.warning(f"Negative price detected at time {t}: lambda_DA_hat={lambda_DA_hat.loc[t]}. Setting DA bid to minimum.")
                p_DA.loc[t] = p_min
                continue

            label = labels.loc[t]
            if label == 0:
                p_DA.loc[t] = p_min
            elif label == 1:
                p_DA.loc[t] = p_max
            else:
                p_DA.loc[t] = P_W_tilde.loc[t]

        if p_DA.isna().any():
            logger.error(f"{p_DA.isna().sum()} DA bids are NaN. Check calculate_bids method.")
            p_DA.fillna(P_W_tilde, inplace=True)

        p_B, p_H, h = calculate_hydrogen_balancing_bids(cfg, p_DA, lambda_B_hat, P_W_hat)
        return p_DA, p_B, p_H, h


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config_dev")
def main(cfg):
    pass

if __name__ == "__main__":
    main()
