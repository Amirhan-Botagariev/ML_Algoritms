import pandas as pd
import numpy as np

class MyLineReg:
    metrics = ['mae', 'mse', 'rmse', 'mape', 'r2']
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None, reg=None, l1_coef=0, l2_coef=0):
        self.n_iter = n_iter

        self.learning_rate = learning_rate
        self.is_dynamic_lr = callable(self.learning_rate)
        self.weights = weights

        self.metric = metric
        self.metric_result = 0

        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

    def fit(self, x: pd.DataFrame, y: pd.Series, verbose=False):
        X = np.array(pd.concat([pd.Series([1] * len(x), name='ones'), x], axis=1))
        Y = np.array(y).ravel()

        learning_rate = self.learning_rate
        n_features = X.shape[1]
        n = Y.size
        self.weights = np.ones(n_features)

        for iteration in range(1, self.n_iter + 1):
            if self.is_dynamic_lr:
                learning_rate = self.learning_rate(iteration)

            pred = np.dot(X, self.weights)
            gradient = self.calculate_gradient(X, Y, pred, self.weights)
            self.weights -= learning_rate * gradient

            loss = self.calculate_loss(Y, pred)
            self.metric_result = self.calculate_metric(Y, pred)

            if verbose and iteration % verbose == 0 and self.metric is not None:
                print(f"{iteration} | loss: {loss:.5f} | {self.metric}: {self.metric_result}")
            elif verbose and iteration % verbose == 0:
                print(f"{iteration} | loss: {loss:.5f}")

    def predict(self, x: pd.DataFrame):
        X = np.array(
            pd.concat([pd.Series([1] * len(x), name='ones'), x], axis=1)
        )

        pred = np.dot(X, self.weights)
        return sum(pred)

    def get_coef(self):
        return self.weights[1:]

    def calculate_loss(self, Y, pred):
        if self.reg == 'l1' and self.l1_coef > 0:
            return np.mean((Y - pred) ** 2) + self.l1_coef * np.sum(abs(self.weights))
        elif self.reg == 'l2' and self.l2_coef > 0:
            return np.mean((Y - pred) ** 2) + self.l2_coef * np.sum(self.weights**2)
        elif self.reg == 'elasticnet' and self.l1_coef > 0 and self.l2_coef > 0:
            return ((np.mean((Y - pred) ** 2)
                    + self.l1_coef * np.sum(abs(self.weights)))
                    + self.l2_coef * np.sum(self.weights**2))
        else:
            return np.mean((Y - pred) ** 2)

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

    def calculate_gradient(self, X, Y, pred, W):
        if self.reg == 'l1' and self.l1_coef > 0:
            return (2 / len(Y)) * np.dot(X.T, (pred - Y)) + self.l1_coef * np.sign(W)
        elif self.reg == 'l2' and self.l2_coef > 0:
            return (2 / len(Y)) * np.dot(X.T, (pred - Y)) + 2 * self.l2_coef * W
        elif self.reg == 'elasticnet' and self.l1_coef > 0 and self.l2_coef > 0:
            return ((2 / len(Y)) * np.dot(X.T, (pred - Y))
                    + self.l1_coef * np.sign(W)
                    + 2 * self.l2_coef * W)
        else:
            return (2 / len(Y)) * np.dot(X.T, (pred - Y))

    def get_best_score(self):
        return self.metric_result

    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"