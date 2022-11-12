from typing import Callable

from sympy import symbols, diff

X = symbols("x")


class Lagrange:
    def __init__(self, n: int, x_vals: list, f: Callable):
        self.n = n
        self.x_vals = x_vals
        self.f = f

    def __basis_poly(self, index: int):
        res = 1
        for i in range(self.n):
            if i != index:
                expr = (X - self.x_vals[index])/(self.x_vals[index] - self.x_vals[i])
                res *= expr
        return res * self.f(self.x_vals[index])

    @property
    def polynomial(self):
        res = 1
        for i in range(self.n):
            res += self.__basis_poly(i)
        return res

    def get_polynomial_diff_func(self, order: int):
        return diff(self.polynomial, X, order)

    def get_diff_in_point(self, order: int, point: float):
        return self.get_polynomial_diff_func(order).as_expr().subs(X, point)

