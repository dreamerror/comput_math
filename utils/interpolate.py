import numpy as np
from sympy import diff, symbols, simplify, Expr


def derivative(expr: Expr, order: int) -> Expr:
    return diff(expr, order)


class FiniteDifference:
    def __init__(self, x_vals: list[float], func: Expr):
        """

        :param func: Expr with x using as a symbol
        """
        self.x_vals = x_vals
        self.n = len(x_vals)
        self.step = x_vals[1] - x_vals[0]
        self.func = func
        self.X = symbols("x")

    def call_func(self, x: float):
        return self.func.subs(self.X, x)

    def dif_forward(self, index: int, order: int = 1):
        while index >= self.n:
            self.x_vals.append(self.x_vals[-1] + self.step)
        if order > 1:
            return self.dif_forward(index + 1, order - 1) - self.dif_forward(index, order - 1)
        return self.call_func(self.x_vals[index+1]) - self.call_func(self.x_vals[index])

    def dif_backwards(self, index: int, order: int = 1):
        while index >= self.n:
            self.x_vals.append(self.x_vals[-1] + self.step)
        if order > 1:
            return self.dif_forward(index, order - 1) - self.dif_forward(index - 1, order - 1)
        return (self.call_func(self.x_vals[index]) - self.call_func(self.x_vals[index - 1])) / self.step


class NewtonInterpolate:
    def __init__(self, start: float, end: float, n: int, func: Expr):
        self.x_0 = start
        self.x_n = end
        self.n = n
        self.x_vals = list(np.linspace(start, end, n))
        self.func = func
        self.X = symbols("x")

    def call_func(self, x: float):
        return self.func.subs(self.X, x)

    def divided_difference(self, x_vals: list[float]):
        if len(x_vals) == 1:
            return self.call_func(x_vals[0])
        result = 0
        for i in range(len(x_vals)):
            divider = 1
            for j in range(len(x_vals)):
                if j != i:
                    divider *= x_vals[j] - x_vals[i]
            result += self.call_func(x_vals[i]) / divider
        return result

    def poly(self, n: int, forward: bool = True):
        if forward:
            x_vals = list(self.x_vals)
        else:
            x_vals = list(self.x_vals[::-1])
        expr = simplify("0")
        for i in range(n):
            prod = 1
            for j in range(i):
                prod *= self.X - x_vals[j]
            expr += prod * self.divided_difference(x_vals[0:i+1])
        return expr[0]
