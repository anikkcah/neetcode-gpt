import numpy as np
from numpy.typing import NDArray
import math

class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        output = []
        for i in z:
            sg = 1 / (1 + math.exp(-i))
            sg = round(sg,5)
            output.append(sg)

        return output

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        output = []
        for i in z:
            relu = max(0,i)
            relu = float(relu)
            output.append(relu)

        return output
