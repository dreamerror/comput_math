import numpy as np
from sympy import Expr, symbols, simplify

from .tools import derivative


def tdma(a, b, c, f):
    a, b, c, f = tuple(map(lambda k_list: list(map(float, k_list)), (a, b, c, f)))

    alpha = [-b[0] / c[0]]
    beta = [f[0] / c[0]]
    n = len(f)
    x = [0] * n

    for i in range(1, n):
        alpha.append(-b[i] / (a[i] * alpha[i - 1] + c[i]))
        beta.append((f[i] - a[i] * beta[i - 1]) / (a[i] * alpha[i - 1] + c[i]))

    x[n - 1] = beta[n - 1]

    for i in range(n - 1, 0, -1):
        x[i - 1] = alpha[i - 1] * x[i] + beta[i - 1]

    return x


class Spline:
    def __init__(self, x_i: float, x_i_1: float, h_i: float, m_i: float, m_i_1: float, func: Expr):
        self.x_i = x_i
        self.x_i_1 = x_i_1
        self.h_i = h_i
        self.m_i = m_i
        self.m_i_1 = m_i_1
        self.func = func

    def __func(self, p: float) -> float:
        return self.func.subs(symbols("x"), p).evalf()

    def derivative(self, order: int, p: float) -> float:
        return derivative(self.func, order).subs(symbols("x"), p).evalf()

    @property
    def a_i(self) -> float:
        return 6 / self.h_i * ((self.__func(self.x_i) - self.__func(self.x_i_1)) / self.h_i
                               - (self.m_i + self.m_i_1 / 3))

    @property
    def b_i(self) -> float:
        return 12 / (self.h_i ** 2) * ((self.m_i + self.m_i_1 / 2)
                                       - (self.__func(self.x_i) - self.__func(self.x_i_1) / self.h_i))

    @property
    def s_i_1(self):
        func_val = self.__func(self.x_i)
        return func_val + self.m_i * self.h_i + self.a_i * (self.h_i**2/2) + self.b_i * (self.h_i**3/6)

    @property
    def abs_mistake(self):
        return abs(self.s_i_1 - self.__func(self.x_i_1))

    @property
    def rel_mistake(self):
        return self.abs_mistake/abs(self.__func(self.x_i_1)) * 100


class SplineGroup:
    def __init__(self, x_0: float, x_n: float, n: int, func: Expr):
        self.func = func
        self.x_0 = x_0
        self.x_n = x_n
        self.n = n

    def __func(self, p: float) -> float:
        return self.func.subs(symbols("x"), p).evalf()

    @property
    def x_vals(self) -> list[float]:
        return list(np.linspace(self.x_0, self.x_n, self.n + 1))

    @property
    def h(self) -> float:
        return self.x_vals[1] - self.x_vals[0]

    @property
    def results_matrix(self):
        matrix = list()
        first = (3 / self.h * (self.__func(self.x_vals[1] - self.x_0))) - \
                (self.h / 2 * derivative(self.func, 2).subs(symbols("x"), self.x_0).evalf())
        matrix.append(first)

        for i in range(1, self.n):
            res = 3 * ((self.__func(self.x_vals[i + 1]) - self.x_vals[i]) / 2 * self.h +
                       (self.__func(self.x_vals[i]) - self.x_vals[i - 1]) / 2 * self.h)
            matrix.append(res)

        last = (self.h * derivative(self.func, 2).subs(symbols("x"), self.x_n).evalf()) / 2 + \
               (3 * (self.__func(self.x_n) - self.__func(self.x_vals[-2]) / self.h))

        matrix.append(last)

        return matrix

    @property
    def main_diagonal(self):
        return [2] * (self.n+1)

    @property
    def diagonal_under(self):
        result = [0]
        result += list([0.5 for i in range(1, self.n)])
        result += [1]
        return result

    @property
    def diagonal_above(self):
        result = [1]
        result += list([0.5 for i in range(1, self.n)])
        result += [0]
        return result

    @property
    def m_values(self):
        return tdma(self.diagonal_under, self.main_diagonal, self.diagonal_above, self.results_matrix)

    @property
    def splines(self) -> list[Spline]:
        return list((Spline(self.x_vals[i],
                            self.x_vals[i + 1],
                            self.h,
                            self.m_values[i],
                            self.m_values[i+1],
                            self.func)
                     for i in range(self.n)))
