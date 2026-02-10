import linopy
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class expando(object):
    '''
        A small class which can have attributes set
    '''
    pass

class ModelHindsight:
    """Model with perfect foresight of day-ahead prices, balancing prices, and wind power generation."""
    def __init__(self, cfg, lambda_DA_hat: pd.Series, lambda_B_hat: pd.Series, P_W_hat: pd.Series, P_W_tilde: pd.Series):
        """Initializes the HindsightModel with given parameters.

        Args:
            cfg: Configuration object containing experiment parameters.
            lambda_DA_hat (pd.Series): Actual day-ahead electricity prices.
            lambda_B_hat (pd.Series): Actual balancing electricity prices.
            P_W_hat (pd.Series): Actual wind power generation.
            P_W_tilde (pd.Series): Forecasted wind power generation.
        """
        # Containers for variables, constraints, and results
        self.parameters = expando()
        self.constants = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self.results.objective_value = None # set objective value to None until optimization is performed

        # Parameters
        assert lambda_DA_hat.index.equals(lambda_B_hat.index)
        assert lambda_DA_hat.index.equals(P_W_hat.index)
        assert lambda_DA_hat.index.equals(P_W_tilde.index)
        self.parameters.T = lambda_DA_hat.index
        self.parameters.lambda_DA_hat = lambda_DA_hat
        self.parameters.lambda_B_hat = lambda_B_hat
        self.parameters.P_W_hat = P_W_hat
        self.parameters.P_W_tilde = P_W_tilde

        # Constants
        self.constants.P_W_BAR = cfg.experiments.optimization_parameters.wind_capacity

        # Create Linopy model
        self.model = linopy.Model()
        self._build_model()

    def _set_variables(self):
        """Sets the decision variables for the optimization model."""
        self.variables.p_DA = self.model.add_variables(
            lower=0,
            upper=self.constants.P_W_BAR,
            coords=[self.parameters.T],
            name="p_DA"
        )
        self.variables.delta_p = self.model.add_variables(
            lower=-self.constants.P_W_BAR,
            upper=self.constants.P_W_BAR,
            coords=[self.parameters.T],
            name="delta_p"
        )
        self.variables.x = self.model.add_variables(
            lower=-self.constants.P_W_BAR,
            upper=self.constants.P_W_BAR,
            name="x"
        )

    def _set_objective(self):
        """Sets the objective function for the optimization model."""
        self.model.add_objective(
            (self.variables.p_DA * self.parameters.lambda_DA_hat).sum()
            + (self.variables.delta_p * self.parameters.lambda_B_hat).sum(),
            sense="max"
        )

    def _set_constraints(self):
        """Sets the constraints for the optimization model."""
        self.constraints.power_limit = self.model.add_constraints(
            self.parameters.P_W_hat >= self.variables.p_DA + self.variables.delta_p,
            name="power_limit"
        )

    def _build_model(self):
        """Builds the optimization model by setting variables, objective, and constraints."""
        self._set_variables()
        self._set_objective()
        self._set_constraints()

    def _solve(self, solver_name="highs"):
        """Solves the optimization model using the specified solver."""
        return self.model.solve(solver_name=solver_name)

    def _save_results(self):
        """Saves the results of the optimization model."""
        self.results.status = self.model.status
        if self.results.status != "ok":
            logger.warning(f"Optimization did not reach optimality. Status: {self.results.status}")
        logger.info(f"Optimization status: {self.results.status}")
        self.results.objective_value = self.model.objective.value
        self.results.p_DA = self.variables.p_DA.solution.to_pandas()
        self.results.delta_p = self.variables.delta_p.solution.to_pandas()
        self.results.x = float(self.variables.x.solution)

    def run_optimization(self, solver_name="highs"):
        """Runs the optimization process: solves the model and saves results."""
        logger.info("Starting optimization...")
        self._solve(solver_name=solver_name)
        self._save_results()
        logger.info("Optimization completed.")


class ModelBalance(ModelHindsight):
    def __init__(self, cfg, lambda_DA_hat: pd.Series, lambda_B_hat: pd.Series, P_W_hat: pd.Series, P_W_tilde: pd.Series):
        """Model with adjustable bid (x) that can be positive or negative."""
        super().__init__(cfg, lambda_DA_hat, lambda_B_hat, P_W_hat, P_W_tilde)

    def _set_constraints(self):
        """Adds additional constraints specific to the ModelBalance."""
        self.constraints.bid_constraint = self.model.add_constraints(
            self.variables.p_DA == self.parameters.P_W_tilde + self.variables.x,
            name="bid_constraint"
        )

class ModelBidForecast(ModelBalance):
    """Model that enforces no bid adjustments (x == 0)."""
    def __init__(self, cfg, lambda_DA_hat, lambda_B_hat, P_W_hat, P_W_tilde):
        super().__init__(cfg, lambda_DA_hat, lambda_B_hat, P_W_hat, P_W_tilde)

    def _set_constraints(self):
        """Adds additional constraints specific to the ModelBidForecast."""
        super()._set_constraints()
        self.constraints.x_limits = self.model.add_constraints(
            self.variables.x == 0,
            name="x_limits"
        )

class ModelSurplus(ModelBalance):
    """Model that allows only surplus bids (x >= 0)."""
    def __init__(self, cfg, lambda_DA_hat, lambda_B_hat, P_W_hat, P_W_tilde):
        super().__init__(cfg, lambda_DA_hat, lambda_B_hat, P_W_hat, P_W_tilde)

    def _set_constraints(self):
        """Adds additional constraints specific to the ModelSurplus."""
        super()._set_constraints()
        self.constraints.x_limits = self.model.add_constraints(
            self.variables.x >= 0,
            name="x_limits"
        )

class ModelDeficit(ModelBalance):
    """Model that allows only deficit bids (x <= 0)."""
    def __init__(self, cfg, lambda_DA_hat, lambda_B_hat, P_W_hat, P_W_tilde):
        super().__init__(cfg, lambda_DA_hat, lambda_B_hat, P_W_hat, P_W_tilde)

    def _set_constraints(self):
        """Adds additional constraints specific to the ModelDeficit."""
        super()._set_constraints()
        self.constraints.x_limits = self.model.add_constraints(
            self.variables.x <= 0,
            name="x_limits"
        )
