import linopy
import pandas as pd
import logging
from omegaconf import DictConfig
from types import SimpleNamespace
import xarray as xr
import hydra

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
        X_features["intercept"] = 1.0
        X_features["lambda_DA_hat"] = lambda_DA_hat
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
        X_features["intercept"] = 1.0
        X_features["lambda_DA_hat"] = lambda_DA_hat
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

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config_dev")
def main(cfg):
    pass

if __name__ == "__main__":
    main()
