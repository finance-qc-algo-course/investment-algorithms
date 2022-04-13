import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from typing import Type


class Model:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def fit(self):
        raise NotImplementedError

    def predict(self, X_test):
        raise NotImplementedError


def cross_validate_time_series(X: np.ndarray, y: np.ndarray, metric, model_class: Type[Model]):
    tscv = TimeSeriesSplit()

    metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]

        model = model_class(X_train, y_train)
        model.fit()
        predictions = model.predict(X_test).flatten()
        assert y_test.shape == predictions.shape, f"{y_test.shape}, {predictions.shape}"

        fold_metric = metric(y_test, predictions)
        metrics.append(fold_metric)
        print(f"Fold {fold_idx + 1}: {fold_metric}")

    return np.mean(metrics)
