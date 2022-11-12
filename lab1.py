from typing import Iterable

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sympy import diff, symbols, log

from utils.lagrange import Lagrange

matplotlib.use('TkAgg')

n_list = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def func(x: float):
    return np.power(x, 2) + np.log(x)


class Solution:
    def __init__(self, n: int):
        self.n = n
        self.x_0 = 0.4
        self.x_n = 0.9

    @property
    def x_values(self, x_0: float = 0.4, x_n: float = 0.9) -> np.ndarray:
        return np.linspace(x_0, x_n, self.n)

    @property
    def y_values(self) -> list:
        return list(
            map(
                func, self.x_values
            )
        )

    @property
    def lagrange(self) -> Lagrange:
        return Lagrange(self.n, self.x_values, self.y_values)

    @property
    def r_n(self):
        ders = list()
        for x in self.x_values:
            t = symbols("t")
            deriv = diff(t**2 + log(t), t, self.n+1)
            ders.append(deriv.as_expr().subs(t, x))

        norm = max(list(map(
            abs, ders
        )))
        quotient = norm/np.math.factorial(self.n+1)
        return quotient * np.power((self.x_n-self.x_0), self.n+1)

    def abs_mistake(self):
        x_vals = np.linspace(self.x_0, self.x_n, self.n * 5)
        func_res = list(map(func, x_vals))
        return list([abs(x1 - x2) for x1, x2 in zip(self.lagrange.polynomial_results(x_vals), func_res)])

    def rel_mistake(self):
        x_vals = np.linspace(self.x_0, self.x_n, self.n * 5)
        lagrange_res = self.lagrange.polynomial_results(x_vals)
        abs_mistakes = self.abs_mistake()
        maxim = max(list(map(abs, lagrange_res)))
        return list(map(lambda x: x/maxim, abs_mistakes))


class SolutionFull:
    def __init__(self, n_values: Iterable):
        self.n_values = n_values
        self.solutions = list()
        for n in n_values:
            self.solutions.append(Solution(n))

    def rel_mistake_n_depend(self):
        result = list()
        for item in self.solutions:
            result.append((item.n, max(item.rel_mistake())))
        return result

    def abs_mistake_n_depend(self):
        result = list()
        for item in self.solutions:
            result.append((item.n, max(item.abs_mistake())))
        return result

    def rn_n_depend(self):
        result = list()
        for item in self.solutions:
            result.append((item.n, item.r_n))
        return result

    def show_rel_mistake_plot(self):
        rel = list()
        for item in self.rel_mistake_n_depend():
            rel.append(item[1])
        plt.plot(self.n_values, rel)
        plt.show()

    def show_abs_rn(self):
        abs_m, rn = list(), list()
        for item in self.abs_mistake_n_depend():
            abs_m.append(item[1])
        for item in self.rn_n_depend():
            rn.append(item[1])
        plt.subplot(121)
        plt.plot(self.n_values, abs_m)
        plt.title("Abs mistake")
        plt.subplot(122)
        plt.plot(self.n_values, rn)
        plt.title("R_n")
        plt.show()


s = SolutionFull(n_list)
s.show_rel_mistake_plot()

