from random import uniform

import numpy as np
from sympy import Expr, symbols


class Integrate:
    def __init__(self, func: Expr, x_0: float, x_n: float):
        self.func = func
        self.x_0 = x_0
        self.x_n = x_n

    def __func(self, x: float):
        return self.func.subs(symbols("x"), x)

    def left_rectangles(self, n: int):
        res = 0
        x_vals = list(np.linspace(self.x_0, self.x_n, n+1))
        for i in range(n):
            res += self.__func(x_vals[i]) * (x_vals[i + 1] - x_vals[i])
        return res

    def right_rectangles(self, n: int):
        res = 0
        x_vals = list(np.linspace(self.x_0, self.x_n, n+1))
        for i in range(1, n+1):
            res += self.__func(x_vals[i]) * (x_vals[i] - x_vals[i-1])
        return res

    def central_rectangles(self, n: int):
        res = 0
        x_vals = list(np.linspace(self.x_0, self.x_n, n+1))
        for i in range(n):
            res += self.__func((x_vals[i] + x_vals[i+1])/2) * (x_vals[i+1] - x_vals[i])
        return res

    def cotes_method(self, n: int):
        x_vals = list(np.linspace(self.x_0, self.x_n, n+1))
        step = x_vals[1] - x_vals[0]
        res = (self.__func(x_vals[0]) + self.__func(x_vals[n])) / 2
        for i in range(1, n):
            res += self.__func(x_vals[i])
        return res * step

    def simpson(self):
        return ((self.x_n - self.x_0) / 6) * (self.__func(self.x_0) + 4 * self.__func((self.x_0 + self.x_n) / 2) +
                                              self.__func(self.x_n))

    def monte_carlo(self, n: int):
        res = 0
        for i in range(n):
            a = uniform(self.x_0, self.x_n)
            res += self.__func(a)
        return res * ((self.x_n - self.x_0) / n)
