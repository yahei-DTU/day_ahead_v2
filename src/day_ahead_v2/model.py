import numpy as np
import pandas as pd
from torch import nn
import torch
import sklearn


class LogisticRegression(sklearn.linear_model.LogisticRegression):
    """Just a dummy Logistic Regression model to show how to structure your code"""
    def __init__(self, **model_params):
        super().__init__(**model_params)

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        """
        Run predictions using the loaded model.
        """
        return pd.Series(
            super().predict_proba(features)[:, 1],
            index=features.index,
        )

    def predict_SI(self, predict_proba: pd.Series, alpha: float) -> pd.Series:
        """
        Convert predicted probabilities to class labels.
        """
        return predict_proba.apply(
            lambda p: 1 if p >= alpha else (-1 if p <= 1 - alpha else 0)
        )

class NN(nn.Module):
    """Just a dummy BART model to show how to structure your code"""
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

class XGBoost:
    """Just a dummy XGBoost model to show how to structure your code"""
    def __init__(self):
        pass

    def predict(self, x):
        return x.sum(axis=1)
    

    


if __name__ == "__main__":
    model = LogisticRegression(C=1.0, l1_ratio=1, solver='saga', max_iter=10000, class_weight='balanced')
    model.fit(pd.DataFrame(np.array([[1, 2], [2, 3], [3, 4]])), np.array([0, 1, 0]))
    preds = model.predict_proba(pd.DataFrame(np.array([[1, 2], [2, 3]])))
    print(preds)
