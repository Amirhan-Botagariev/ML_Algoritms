import pandas as pd
import numpy as np

class MyLineReg:
    metrics = ['mae', 'mse', 'rmse', 'mape', 'r2']
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.metric_result = 0

    def fit(self, x: pd.DataFrame, y: pd.Series, verbose=False):
        X = np.array(pd.concat([pd.Series([1] * len(x), name='ones'), x], axis=1))
        Y = np.array(y).ravel()

        n_features = X.shape[1]
        n = Y.size
        self.weights = np.ones(n_features)

        for iteration in range(1, self.n_iter + 2):
            pred = np.dot(X, self.weights)
            gradient = (2 / n) * np.dot(X.T, (pred - Y))
            self.weights -= self.learning_rate * gradient
            mse = np.mean((Y - pred) ** 2)
            metric = self.calculate_metric(Y, pred)
            self.metric_result = metric

            if verbose and iteration % verbose == 0 and self.metric is not None:
                print(f"{iteration} | loss: {mse:.5f} | {self.metric}: {metric}")
            elif verbose and iteration % verbose == 0:
                print(f"{iteration} | loss: {mse:.5f}")

    def predict(self, x: pd.DataFrame):
        X = np.array(
            pd.concat([pd.Series([1] * len(x), name='ones'), x], axis=1)
        )

        pred = np.dot(X, self.weights)
        return sum(pred)

    def get_coef(self):
        return self.weights[1:]

    def calculate_metric(self, Y, pred):
        if self.metric == 'mae':
            return np.mean(np.abs(Y - pred))
        elif self.metric == 'mse':
            return np.mean((Y - pred) ** 2)
        elif self.metric == 'rmse':
            return np.sqrt(np.mean((Y - pred) ** 2))
        elif self.metric == 'mape':
            return 100 * np.mean(np.abs((Y - pred) / Y))
        elif self.metric == 'r2':
            ss_res = np.sum((Y - pred) ** 2)
            ss_tot = np.sum((Y - np.mean(Y)) ** 2)
            return 1 - (ss_res / ss_tot)

    def get_best_score(self):
        return self.metric_result

    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"