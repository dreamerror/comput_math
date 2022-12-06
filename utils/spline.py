import numpy as np
from sympy import Expr, symbols

from .tools import derivative


class Spline:
    def __init__(self, x_i: float, x_i_1: float, h_i: float, func: Expr):
        self.x_i = x_i
        self.x_i_1 = x_i_1
        self.h_i = h_i
        self.func = func

    def __func(self, p: float):
        return self.func.subs(symbols("x"), p).evalf()

    def derivative(self, order: int, p: float):
        return derivative(self.func, order).subs(symbols("x"), p).evalf()

    @property
    def a_i(self):
        return self.__func(self.x_i)

    @property
    def b_i(self):
        return self.derivative(1, self.x_i)

    @property
    def c_i(self):
        der_sum = (self.derivative(1, self.x_i) + self.derivative(1, self.x_i_1))/2
        val_diff = (self.__func(self.x_i_1) - self.__func(self.x_i))/self.h_i
        return (12/(self.h_i**2)) * (der_sum - val_diff)

    @property
    def d_i(self):
        der = (-2 * self.derivative(1, self.x_i) + self.derivative(1, self.x_i_1))/3
        val = (self.__func(self.x_i_1) - self.__func(self.x_i))/self.h_i
        return (6/self.h_i) * (der + val)

    @property
    def equation(self):
        x = symbols("x")
        res = self.a_i
        res += self.b_i * (x - self.x_i)
        res += self.c_i * (x - self.x_i)**2
        res += self.d_i * (x - self.x_i)**3
        return res
