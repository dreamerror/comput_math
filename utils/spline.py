import numpy as np
from sympy import Expr, symbols, simplify

from .tools import derivative


class Spline:
    def __init__(self, x_i: float, x_i_1: float, h_i: float, func: Expr):
        self.x_i = x_i
        self.x_i_1 = x_i_1
        self.h_i = h_i
        self.func = func

    def __func(self, p: float) -> float:
        return self.func.subs(symbols("x"), p).evalf()

    def derivative(self, order: int, p: float) -> float:
        return derivative(self.func, order).subs(symbols("x"), p).evalf()

    @property
    def norm(self) -> float:
        return max(list(self.__func(p) for p in (self.x_i, self.x_i_1)))

    @property
    def a_i(self) -> float:
        return self.__func(self.x_i)

    @property
    def b_i(self) -> float:
        return self.derivative(1, self.x_i)

    @property
    def c_i(self) -> float:
        der_sum = (self.derivative(1, self.x_i) + self.derivative(1, self.x_i_1)) / 2
        val_diff = (self.__func(self.x_i_1) - self.__func(self.x_i)) / self.h_i
        return (12 / (self.h_i ** 2)) * (der_sum - val_diff)

    @property
    def d_i(self) -> float:
        der = (-2 * self.derivative(1, self.x_i) + self.derivative(1, self.x_i_1)) / 3
        val = (self.__func(self.x_i_1) - self.__func(self.x_i)) / self.h_i
        return (6 / self.h_i) * (der + val)

    @property
    def equation(self) -> Expr:
        x = symbols("x")
        res = simplify(self.a_i)
        res += self.b_i * (x - self.x_i)
        res += self.c_i * (x - self.x_i) ** 2
        res += self.d_i * (x - self.x_i) ** 3
        return res


class SplineGroup:
    def __init__(self, x_0: float, x_n: float, n: int, func: Expr):
        self.func = func
        self.x_0 = x_0
        self.x_n = x_n
        self.n = n

    @property
    def x_vals(self) -> list[float]:
        return list(np.linspace(self.x_0, self.x_n, self.n + 1))

    @property
    def h(self) -> float:
        return self.x_vals[1] - self.x_vals[0]

    @property
    def splines(self) -> list[Spline]:
        return list((Spline(self.x_vals[i],
                            self.x_vals[i + 1],
                            self.h,
                            self.func)
                     for i in range(self.n)))
