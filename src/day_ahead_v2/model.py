import numpy as np
import pandas as pd
import sklearn
import logging
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import lightgbm as lgb
import xgboost as xgb
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

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
        logger.info("LogisticRegression initialized with parameters")

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
        logger.info("GaussianProcessClassifier initialized with parameters")

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
            objective="binary",
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            **kwargs,
        )
        logger.info("LightGBMClassifier initialized with parameters")

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
        learning_rate: float = 0.03,
        n_estimators: int = 500,
        max_depth: int = 3,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 5,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            objective="binary:logistic",
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            min_child_weight=min_child_weight,
            eval_metric="mlogloss", # default multi-class metric
            **kwargs,
            )
        logger.info("XGBoostClassifier initialized with parameters")

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Fit the XGBoost model.

        Args:
            X (pd.DataFrame): Feature data for training.
            y (pd.Series): Target labels for training.
            **kwargs: Additional keyword arguments for fit.
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


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activation: str = "relu",
        dropout: float = 0.2,
        device: str = "cpu",
        random_state: int = 42,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 50,
        weight_decay: float = 1e-4,
        patience: int = 5,
        verbose: int = 1,
        optimizer: str = "Adam",
    ):
        super().__init__()
        torch.manual_seed(random_state)
        self.device = device

        # Build hidden layers
        layers = []
        prev_dim = input_dim
        act_fn = nn.ReLU if activation == "relu" else nn.Tanh

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers).to(device)

        logger.info(f"MLPClassifier initialized with architecture: {self.model}")

        # Set training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.patience = patience
        self.verbose = verbose
        self.optimizer = optimizer

        # Initialize classes_ attribute
        self.classes_ = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def fit(self, X: torch.Tensor | np.ndarray | pd.DataFrame, y: torch.Tensor | np.ndarray | pd.Series):
        """
        Train the MLP classifier.
        Args:
            X: Input features (Tensor, ndarray, or DataFrame).
            y: Target labels (Tensor, ndarray, or Series).
        """
        # Convert X to tensor
        if isinstance(X, pd.DataFrame):
            X = torch.tensor(X.values, dtype=torch.float32)
        elif isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        # Convert y to tensor
        if isinstance(y, pd.Series):
            y = torch.tensor(y.values, dtype=torch.float32)
        elif isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)

        # Set classes_ attribute
        self.classes_ = torch.unique(y).sort()[0].cpu().numpy()
        logger.info(f"Classes found in training data: {self.classes_}")

        # Check if features match model input dimension
        if X.shape[1] != self.model[0].in_features:
            raise ValueError(f"Input feature dimension {X.shape[1]} does not match model expected dimension {self.model[0].in_features}.")

        # Ensure labels are 0/1
        if not torch.all((y == 0) | (y == 1)):
            raise ValueError("Binary labels must be 0 and 1.")

        dataset = TensorDataset(X, y.unsqueeze(1))  # Ensure y is of shape (n_samples, 1)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = (optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
                     if self.optimizer == "Adam"
                     else optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay))

        best_loss = float("inf")
        patience_counter = 0

        # Choose iterator based on verbose
        iterator = trange(self.max_epochs, desc="Training") if self.verbose == 1 else range(self.max_epochs)

        for epoch in iterator:
            epoch_loss = 0.0
            self.train()
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)

            epoch_loss /= len(loader.dataset)

            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose > 0:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            # Verbose printing
            if self.verbose == 2:
                logger.info(f"Epoch {epoch+1}/{self.max_epochs}, Loss: {epoch_loss:.4f}")

    def predict_proba(self, X: torch.Tensor | np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for the input samples.

        Args:
            X: Input features (Tensor, ndarray, or DataFrame).

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        self.eval()
        if isinstance(X, pd.DataFrame):
            X = torch.tensor(X.values, dtype=torch.float32)
        elif isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self(X.to(self.device))
            probs_pos = torch.sigmoid(logits)
            probs_neg = 1 - probs_pos
            probs = torch.cat([probs_neg, probs_pos], dim=1)
            return probs.cpu().numpy()


    def predict(self, X: torch.Tensor | np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for the input samples.

        Args:
            X: Input features (Tensor, ndarray, or DataFrame).

        Returns:
            np.ndarray: Predicted class labels.
        """
        self.eval()  # still set the model to eval mode
        if isinstance(X, pd.DataFrame):
            X = torch.tensor(X.values, dtype=torch.float32)
        elif isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self(X.to(self.device))
            probs = torch.sigmoid(logits)
            return (probs >= 0.5).long().cpu().numpy().flatten()
