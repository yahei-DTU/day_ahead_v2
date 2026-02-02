import numpy as np
import pandas as pd
import sklearn
import logging
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import lightgbm as lgb
import xgboost as xgb

logger = logging.getLogger(__name__)

class LogisticRegression(sklearn.linear_model.LogisticRegression):
    """
    Extends sklearn's LogisticRegression to include methods for probability prediction
    and converting probabilities to class labels based on a threshold.

    Attributes:
        kwargs: Keyword arguments for the scikit-learn LogisticRegression constructor.
    """
    def __init__(self, C: float = 1.0, solver: str = 'lbfgs', max_iter: int = 100, l1_ratio: float = 0.0, **kwargs):
        super().__init__(C=C, solver=solver, max_iter=max_iter, l1_ratio=l1_ratio, **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Fit the Logistic Regression model.

        Args:
            X (pd.DataFrame): Feature data for training.
            y (pd.Series): Target labels for training.
            **kwargs: Additional keyword arguments for fit.
        """
        super().fit(X, y, **kwargs)

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Predict class probabilities for the input features.

        Args:
            X (pd.DataFrame): Feature data for prediction.
            **kwargs: Additional keyword arguments for predict_proba.

        Returns:
            np.ndarray: Predicted class probabilities with shape (n_samples, n_classes).
        """
        return super().predict_proba(X, **kwargs)

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Predict class labels for the input features.

        Args:
            X (pd.DataFrame): Feature data for prediction.
            **kwargs: Additional keyword arguments for predict.

        Returns:
            np.ndarray: Predicted class labels.
        """
        return super().predict(X, **kwargs)

class GaussianProcessClassifier(sklearn.gaussian_process.GaussianProcessClassifier):
    """
    Wrapper around sklearn's GaussianProcessClassifier
    with a consistent interface for your pipeline.
    """

    def __init__(
        self,
        kernel=None,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer: int = 0,
        max_iter_predict: int = 100,
        warm_start: bool = False,
        random_state: int | None = None,
        **kwargs,
    ):
        # Sensible default kernel if none is provided
        if kernel is None:
            kernel = C(1.0) * RBF(1.0)

        super().__init__(
            kernel=kernel,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            max_iter_predict=max_iter_predict,
            warm_start=warm_start,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Fit the Gaussian Process Classifier.
        """
        super().fit(X, y, **kwargs)

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Predict class probabilities.

        Returns:
            np.ndarray of shape (n_samples, n_classes)
        """
        return super().predict_proba(X, **kwargs)

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Predict class labels for the input features.

        Args:
            X (pd.DataFrame): Feature data for prediction.
            **kwargs: Additional keyword arguments for predict.

        Returns:
            np.ndarray: Predicted class labels.
        """
        return super().predict(X, **kwargs)

class LightGBMClassifier(lgb.LGBMClassifier):
    """
    Wrapper around LightGBM's LGBMClassifier to ensure compatibility
    with the training and evaluation pipeline.
    """

    def __init__(
        self,
        objective: str = "multiclass",
        num_class: int = 3,
        learning_rate: float = 0.05,
        n_estimators: int = 200,
        max_depth: int = -1,
        num_leaves: int = 31,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            objective=objective,
            num_class=num_class,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Fit LightGBM model.
        """
        super().fit(X, y, **kwargs)

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X (pd.DataFrame): Feature data for prediction.
            **kwargs: Additional keyword arguments for predict_proba.

        Returns:
            np.ndarray of shape (n_samples, n_classes)
        """
        return super().predict_proba(X, **kwargs)

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Predict class labels for the input features.

        Args:
            X (pd.DataFrame): Feature data for prediction.
            **kwargs: Additional keyword arguments for predict.

        Returns:
            np.ndarray: Predicted class labels.
        """
        return super().predict(X, **kwargs)



class XGBoostClassifier(xgb.XGBClassifier):
    """
    Wrapper around XGBoost's XGBClassifier to ensure compatibility
    with the training and evaluation pipeline.
    """

    def __init__(
        self,
        objective: str = "multi:softprob",
        learning_rate: float = 0.05,
        n_estimators: int = 200,
        max_depth: int = 6,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            objective=objective,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            eval_metric="mlogloss",   # default multi-class metric
            **kwargs,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Fit the XGBoost model.
        """
        super().fit(X, y, **kwargs)

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X (pd.DataFrame): Feature data for prediction.
            **kwargs: Additional keyword arguments for predict_proba.

        Returns:
            np.ndarray of shape (n_samples, n_classes)
        """
        return super().predict_proba(X, **kwargs)

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Predict class labels for the input features.

        Args:
            X (pd.DataFrame): Feature data for prediction.
            **kwargs: Additional keyword arguments for predict.

        Returns:
            np.ndarray: Predicted class labels.
        """
        return super().predict(X, **kwargs)
