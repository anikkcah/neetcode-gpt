import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_val = max(z)
        
        sum_ = sum([math.exp(i-max_val) for i in z])

        output = []

        for i in z:
            output.append(np.round(math.exp(i-max_val)/sum_,4))

        return output
