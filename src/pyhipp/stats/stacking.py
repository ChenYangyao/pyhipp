import numpy as np

class Stack:
    @staticmethod
    def mean_and_sd(x: np.ndarray):
        x = np.asarray(x)
        return x.mean(0), x.std(0)