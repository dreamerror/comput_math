import numpy as np
from sympy import Expr, symbols


class Integrate:
    def __init__(self, func: Expr, x_0: float, x_n: float):
        self.func = func
        self.x_0 = x_0
        self.x_n = x_n

    def __func(self, x: float):
        return self.func.subs(symbols("x"), x)

    def left_rectangles(self, n: int, eps: float):
        i_n, i_2n = 0, 0
        x_vals = list(np.linspace(self.x_0, self.x_n, n))
        x_vals_2n = list(np.linspace(self.x_0, self.x_n, 2*n))
        for i in range(2*n):
            if i < n:
                i_n += self.__func(x_vals[i]) * (x_vals[i + 1] - x_vals[i])
            i_2n += self.__func(x_vals_2n[i]) * (x_vals_2n[i + 1] - x_vals_2n[i])
        if abs(i_2n - i_n) < eps:
            return i_2n, 2*n
        else:
            return self.left_rectangles(2*n)