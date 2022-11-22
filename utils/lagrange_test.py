import numpy as np
from sympy import symbols, diff, simplify, Expr

from .tools import ValueInfo, MinMax, func_diff_vals, get_rn_vals, derivative


class Lagrange:
    def __init__(self, n: int, x_vals: list, f: Expr):
        self.n = n
        self.x_vals = x_vals
        self.f = f
        self.X = symbols("x")
        self.H = symbols("h")

    def get_h(self, left: int, right: int):
        return self.H * (left - right)

    def call_func(self, x: float):
        return self.f.subs(self.X, x)

    def __basis_poly(self, index: int):
        res = simplify("1")
        for i in range(self.n):
            if i != index:
                expr = (self.X - self.x_vals[index])/(self.get_h(index, i))
                res *= expr
        return res * self.call_func(self.x_vals[index])

    @property
    def polynomial(self):
        res = simplify("1")
        for i in range(self.n):
            res += self.__basis_poly(i)
        return res

    def get_polynomial_diff_func(self, order: int):
        return diff(self.polynomial, self.X, order)

    def get_diff_in_point(self, order: int, point: float):
        return self.get_polynomial_diff_func(order).as_expr().subs(self.X, point)

    @property
    def get_func_diff_vals(self) -> MinMax:
        return func_diff_vals(derivative(self.f, 1), self.n, self.X, self.x_vals)

    @property
    def get_rn(self):
        return get_rn_vals(self.x_vals, self.X, self.get_func_diff_vals, self.n)

    @property
    def get_rnk_formula(self) -> Expr:
        prod = simplify("1")
        for x in self.x_vals:
            prod *= self.X - x

        max_der = self.get_func_diff_vals.max.val

        return prod * (max_der / np.math.factorial(self.n + 1))

    def get_rnk_values(self) -> MinMax:
        rnk = self.get_rnk_formula
        values = list((ValueInfo((rnk / (self.X - x)).subs(self.X, x), x) for x in self.x_vals))

        max_val = max(values, key=lambda x: x.val)
        min_val = min(values, key=lambda x: x.val)

        return MinMax(min_val, max_val)


