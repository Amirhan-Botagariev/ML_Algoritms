import pandas as pd
import numpy as np

class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def fit(self, x: pd.DataFrame, y: pd.Series, verbose=False):
        X = np.array(pd.concat([pd.Series([1] * len(x), name='ones'), x], axis=1))
        Y = np.array(y).ravel()

        n_features = X.shape[1]
        n = Y.size
        self.weights = np.ones(n_features)
        pred = np.dot(X, self.weights)
        mse_start = np.mean((Y - pred) ** 2)

        if verbose:
            print(f"start | loss: {mse_start:.5f}")

        for iteration in range(1, self.n_iter + 1):
            gradient = (2 / n) * np.dot(X.T, (pred - Y))
            self.weights -= self.learning_rate * gradient
            pred = np.dot(X, self.weights)
            mse = np.mean((Y - pred) ** 2)

            if verbose and iteration % verbose == 0:
                print(f"{iteration * 10} | loss: {mse:.5f}")

    def predict(self, x: pd.DataFrame):
        X = np.array(
            pd.concat([pd.Series([1] * len(x), name='ones'), x], axis=1)
        )

        pred = np.dot(X, self.weights)
        return sum(pred)

    def get_coef(self):
        return self.weights[1:]



    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"