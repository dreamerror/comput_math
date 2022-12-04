from random import uniform

import numpy as np
from sympy import Expr, symbols, integrate

from .tools import derivative


class Integrate:
    def __init__(self, func: Expr, x_0: float, x_n: float):
        self.func = func
        self.x_0 = x_0
        self.x_n = x_n

    def __func(self, x: float):
        return self.func.subs(symbols("x"), x)

    @property
    def definite_integral(self):
        return integrate(self.func, (symbols("x"), self.x_0, self.x_n))

    def abs_mistake(self, i_n: float) -> float:
        return abs(self.definite_integral.evalf() - i_n)

    def rel_mistake(self, i_n: float) -> float:
        return self.abs_mistake(i_n)/abs(self.definite_integral)

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
        """
        Also known as Trapezoidal method
        Cotes' formula is just a pretty version that I found on Russian Wiki page
        """
        x_vals = list(np.linspace(self.x_0, self.x_n, n+1))
        step = x_vals[1] - x_vals[0]
        res = (self.__func(x_vals[0]) + self.__func(x_vals[n])) / 2
        for i in range(1, n):
            res += self.__func(x_vals[i])
        return res * step

    def simpson(self, n: int):
        res = self.__func(self.x_0) + self.__func(self.x_n)
        N = 2*n
        x_vals = list(np.linspace(self.x_0, self.x_n, N+1))
        step = (self.x_n - self.x_0) / N
        for k in range(1, N):
            if k % 2:
                res += 4 * self.__func(x_vals[k])
            else:
                res += 2 * self.__func(x_vals[k])
        return res * (step / 3)

    def monte_carlo(self, n: int):
        res = 0
        for i in range(n):
            a = uniform(self.x_0, self.x_n)
            res += self.__func(a)
        return res * ((self.x_n - self.x_0) / n)

    def r_n_left(self, n: int) -> float:
        x_vals = list(np.linspace(self.x_0, self.x_n, n + 1))
        y_vals = list((derivative(self.func, 1).subs(symbols("x"), x).evalf() for x in x_vals))
        y = max(y_vals, key=lambda item: abs(item))
        return y/(2*(n+1)) * (self.x_n - self.x_0)**2

    def r_n_right(self, n: int) -> float:
        return -1 * self.r_n_left(n)

    def r_n_central(self, n: int) -> float:
        x_vals = list(np.linspace(self.x_0, self.x_n, n + 1))
        y_vals = list((derivative(self.func, 2).subs(symbols("x"), x).evalf() for x in x_vals))
        y = max(y_vals, key=lambda item: abs(item))
        return y / 24 * (self.x_n - self.x_0) ** 3

    def r_n_cotes(self, n: int) -> float:
        x_vals = np.linspace(self.x_0, self.x_n, n + 1)
        y_vals = list((derivative(self.func, 2).subs(symbols("x"), x).evalf() for x in list(x_vals)))
        y = max(y_vals, key=lambda item: abs(item)) * -1
        return y/12 * (self.x_n - self.x_0) * ((self.x_n - self.x_0)/(n+1))**2

    def r_n_simpson(self, n: int) -> float:
        x_vals = list(np.linspace(self.x_0, self.x_n, n + 1))
        y_vals = list((derivative(self.func, 4).subs(symbols("x"), x).evalf() for x in x_vals))
        y = max(y_vals, key=lambda item: abs(item))
        return -1 * (((self.x_n - self.x_0)/180) * ((self.x_n - self.x_0)/(n+1))**4 * y)

    def r_n_monte(self, n: int) -> float:
        """
        Dispersion should be similar for great enough n,
        so estimation quite ok
        """
        x_vals = list(uniform(self.x_0, self.x_n) for i in range(n+1))
        dispersion = float(np.var(x_vals))
        return np.math.sqrt(dispersion/n)
