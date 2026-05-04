import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)

        n = len(y_true)

        lst = []

        for i in range(n):

            lst.append(y_true[i]*math.log(y_pred[i]+math.pow(10,-7)) + (1-y_true[i])*math.log(1-y_pred[i]+math.pow(10,-7)))

        
        sum_ = sum(lst) 

        output = - sum_ / n

        return round(output,4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)

        n = len(y_true)     

        lst = []

        for i in range(n):
            for j in range(len(y_true[0])):
                lst.append(y_true[i][j]*math.log(y_pred[i][j]+math.pow(10,-7)))

        sum_ = - sum(lst) / n

        return round(sum_,4)

        
