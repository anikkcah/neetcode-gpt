import numpy as np
from numpy.typing import NDArray
import math

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places

        y_hat = []

        for i in range(len(X)):
            y_hat.append(round(np.sum(np.multiply(X[i],weights)),5))

        return y_hat
        

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places

        n = len(model_prediction)

        sum_ = 0

        for i in range(n):
            sum_ = np.sum((model_prediction - ground_truth)**2)

        sum_ = sum_/n

        return round(sum_,5)
