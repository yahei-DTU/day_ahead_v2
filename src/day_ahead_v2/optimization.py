import linopy
from pathlib import Path
import pandas as pd
import logging
from omegaconf import DictConfig
from types import SimpleNamespace
import xarray as xr
import hydra
from day_ahead_v2.data import DataHandler

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
        self.variables = SimpleNamespace()
        self.constraints = SimpleNamespace()
        self.results = SimpleNamespace()
        self.results.objective_value = None # set objective value to None until optimization is performed

        assert lambda_DA_hat.index.equals(lambda_B_hat.index)
        assert lambda_DA_hat.index.equals(P_W_hat.index)

        # Constants
        self.constants.T = lambda_DA_hat.index
        self.constants.P_W_BAR = cfg.experiments.optimization_parameters.wind_capacity

        # Parameters
        self.parameters.lambda_DA_hat = xr.DataArray(
            lambda_DA_hat,
            dims=["datetime"],
            coords=[self.constants.T]
        )
        self.parameters.lambda_B_hat = xr.DataArray(
            lambda_B_hat,
            dims=["datetime"],
            coords=[self.constants.T]
        )
        self.parameters.P_W_hat = xr.DataArray(
            P_W_hat,
            dims=["datetime"],
            coords=[self.constants.T]
        )

        # Create optimization model
        self.model = linopy.Model()

    def _set_variables(self) -> None:
        """Sets the decision variables for the optimization model."""
        self.variables.p_DA = self.model.add_variables(
            lower=0,
            upper=self.constants.P_W_BAR,
            dims=["datetime"],
            coords=[self.constants.T],
            name="p_DA"
        )
        self.variables.delta_p = self.model.add_variables(
            dims = ["datetime"],
            coords = [self.constants.T],
            name = "delta_p"
        )

    def _set_objective(self) -> None:
        """Sets the objective function for the optimization model."""
        self.model.add_objective(
            (self.variables.p_DA * self.parameters.lambda_DA_hat).sum()
            + (self.variables.delta_p * self.parameters.lambda_B_hat).sum(),
            sense="max"
        )

    def _set_constraints(self) -> None:
        """Sets the constraints for the optimization model."""
        self.constraints.power_limit = self.model.add_constraints(
            self.variables.p_DA + self.variables.delta_p == self.parameters.P_W_hat,
            name="power_balance"
        )

    def build_model(self) -> None:
        """Builds the optimization model by setting variables, objective, and constraints."""
        self._set_variables()
        self._set_objective()
        self._set_constraints()

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
        self.results.p_DA = self.variables.p_DA.solution.to_pandas()
        self.results.delta_p = self.variables.delta_p.solution.to_pandas()

    def run_optimization(self, solver_name="highs") -> None:
        """Runs the optimization process: solves the model and saves results."""
        logger.info("Starting optimization...")
        self._solve(solver_name=solver_name)
        self._save_results()
        logger.info("Optimization completed.")

class ModelBidForecast(ModelHindsight):
    """Model that enforces to bid forecast."""
    def __init__(self, cfg: DictConfig, lambda_DA_hat: pd.Series, lambda_B_hat: pd.Series, P_W_hat: pd.Series, P_W_tilde: pd.Series, **kwargs):
        super().__init__(cfg, lambda_DA_hat, lambda_B_hat, P_W_hat, **kwargs)
        assert lambda_DA_hat.index.equals(P_W_tilde.index)
        self.parameters.P_W_tilde = xr.DataArray(
            P_W_tilde.values,
            dims=["datetime"],
            coords={"datetime": P_W_tilde.index}
        )

    def _set_variables(self):
        """Adds additional variables specific to the ModelBidForecast."""
        super()._set_variables()
        self.variables.z = self.model.add_variables(
            name="z",
            dims=["datetime"],
            coords=[self.constants.T]
        )

    def _set_constraints(self):
        """Adds additional constraints specific to the ModelBidForecast."""
        super()._set_constraints()
        # Bid adjustment
        self.constraints.bid_adjustment = self.model.add_constraints(
            self.variables.p_DA - self.variables.z == self.parameters.P_W_tilde,
            name='bid_adjustment'
        )
        self.constraints.z_bounds = self.model.add_constraints(
            self.variables.z == 0,
            name='z_is_0'
        )

    def _save_results(self):
        super()._save_results()
        self.results.z = self.variables.z.solution.to_pandas()

class ModelLinearPolicy(ModelHindsight):
    """Model that uses a linear policy for bid adjustments (z = X_features @ q)."""
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
            coords={"datetime": P_W_tilde.index}
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
                "feature": self.constants.FEATURE_DIM,
            },
        )

    def _set_variables(self):
        """Adds additional variables specific to the ModelLinearPolicy."""
        super()._set_variables()
        self.variables.z = self.model.add_variables(
            name="z",
            dims=["datetime"],
            coords=[self.constants.T]
        )

        self.variables.q = self.model.add_variables(
            name="q",
            dims=["feature"],
            coords=[self.constants.FEATURE_DIM]
        )

    def _set_constraints(self):
        super()._set_constraints()
        self.constraints.linear_policy = self.model.add_constraints(
            self.variables.q.dot(self.parameters.X_features.transpose("feature", "datetime")) - self.variables.z == 0,
            name='linear_policy'
        )
        logger.debug(f"X_features shape: {self.parameters.X_features.shape}")
        logger.debug(f"q shape: {self.variables.q.shape}")
        logger.debug(f"z shape: {self.variables.z.shape}")
        logger.debug(f"q@X shape: {self.variables.q.dot(self.parameters.X_features.transpose("feature", "datetime")).shape}")

        # Bid adjustment
        self.constraints.bid_adjustment = self.model.add_constraints(
            self.variables.p_DA - self.variables.z == self.parameters.P_W_tilde,
            name='bid_adjustment'
        )

    def _save_results(self):
        super()._save_results()
        self.results.z = self.variables.z.solution.to_pandas()
        self.results.q = self.variables.q.solution.to_pandas()

class ModelSurplus(ModelLinearPolicy):
    """Model that allows only surplus bids (z >= 0)."""
    def __init__(self, cfg: DictConfig, lambda_DA_hat: pd.Series, lambda_B_hat: pd.Series, P_W_hat: pd.Series, P_W_tilde: pd.Series, X_features: pd.DataFrame, **kwargs):
        super().__init__(cfg, lambda_DA_hat, lambda_B_hat, P_W_hat, P_W_tilde, X_features, **kwargs)

    def _set_constraints(self):
        """Adds additional constraints specific to the ModelSurplus."""
        super()._set_constraints()
        self.constraints.x_limits = self.model.add_constraints(
            self.variables.z >= 0,
            name="z_low_bound"
        )


class ModelDeficit(ModelLinearPolicy):
    """Model that allows only deficit bids (z <= 0)."""
    def __init__(self, cfg: DictConfig, lambda_DA_hat: pd.Series, lambda_B_hat: pd.Series, P_W_hat: pd.Series, P_W_tilde: pd.Series, X_features: pd.DataFrame, **kwargs):
        super().__init__(cfg, lambda_DA_hat, lambda_B_hat, P_W_hat, P_W_tilde, X_features, **kwargs)

    def _set_constraints(self):
        """Adds additional constraints specific to the ModelDeficit."""
        super()._set_constraints()
        self.constraints.x_limits = self.model.add_constraints(
            self.variables.z <= 0,
            name="z_up_bound"
        )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config_dev")
def main(cfg):
    data_handler = DataHandler(cfg)
    data_handler.set_data(data_handler.data.iloc[100:105])
    logger.info(f"column numbers: {data_handler.data.shape[1]}")
    feature_columns = cfg.datasets.training.feature_columns[:3]
    features = data_handler.data[feature_columns].copy()
    logger.info(f"column numbers: {features.shape[1]}")
    logger.info(f"other columns: {set(data_handler.data.columns) - set(feature_columns)}")
    lambda_DA_hat = data_handler.data[cfg.datasets.optimization.lambda_DA_hat]
    lambda_B_hat = data_handler.data[cfg.datasets.optimization.lambda_B_hat]
    P_W_hat = data_handler.data[cfg.datasets.optimization.P_W_hat]
    P_W_tilde = data_handler.data[cfg.datasets.optimization.P_W_tilde]
    logger.debug(f"lambda_DA_hat: {lambda_DA_hat}")
    logger.debug(f"lambda_B_hat: {lambda_B_hat}")
    logger.debug(f"P_W_hat: {P_W_hat}")
    logger.debug(f"P_W_tilde: {P_W_tilde}")
    logger.debug(f"features: {features}")
    optimizer = ModelSurplus(
        cfg=cfg,
        lambda_DA_hat=lambda_DA_hat,
        lambda_B_hat=lambda_B_hat,
        P_W_hat=P_W_hat,
        P_W_tilde=P_W_tilde,
        X_features=features,
    )
    optimizer.build_model()
    root = Path(__file__).resolve().parent.parent.parent
    save_path = root / "models" / "lp_files" / "model.lp"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer.model.to_file(save_path)
    optimizer.run_optimization()
    logger.info(f"Optimizer objective value: {optimizer.results.objective_value}")


if __name__ == "__main__":
    main()
