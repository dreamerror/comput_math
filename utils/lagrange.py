import numpy as np


class Lagrange:
    def __init__(self, n: int, x_values: list | np.ndarray, y_values: list | np.ndarray):
        self.n = n
        self.x_values = x_values
        self.y_values = y_values

    def __multiple(self, index: int, x_p: float):
        result = 1
        for j in range(self.n):
            if j != index:
                result *= (x_p - self.x_values[j]) / (self.x_values[index] - self.x_values[j])
        return result

    def __single_sum(self, index: int, x_p: float):
        return self.y_values[index] * self.__multiple(index, x_p)

    def polynomial(self, x_p: float):
        result = 0
        for i in range(self.n):
            result += self.__single_sum(i, x_p)
        return result

    def polynomial_results(self, x_vals: list | np.ndarray):
        return list(
            map(
                self.polynomial, x_vals
            )
        )