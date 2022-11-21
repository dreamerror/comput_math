from dataclasses import dataclass

import numpy as np
from sympy import symbols, simplify, diff, Expr, Symbol


def derivative(expr: Expr, order: int) -> Expr:
    return diff(expr, symbols("x"), order)


@dataclass
class ValueInfo:
    val: float
    x: float


@dataclass
class MinMax:
    min: ValueInfo
    max: ValueInfo


def func_diff_vals(func: Expr, n: int, x_symbol: Symbol, x_vals: list) -> MinMax:
    deriv = derivative(func, n + 1)
    check_list = list((ValueInfo(val=deriv.subs(x_symbol, x), x=x)) for x in x_vals)

    max_val = max(check_list, key=lambda item: item.val)
    min_val = min(check_list, key=lambda item: item.val)

    return MinMax(min_val, max_val)


def get_rn_vals(x_vals: list, x_symbol: Symbol, derivatives: MinMax, n: int) -> MinMax:
    prod = simplify("1")
    for x in x_vals:
        prod *= x_symbol - x

    max_val = ValueInfo(val=prod.subs(x_symbol, derivatives.max.x) * derivatives.max.val / np.math.factorial(n + 1),
                        x=derivatives.max.x)
    min_val = ValueInfo(val=prod.subs(x_symbol, derivatives.min.x) * derivatives.min.val / np.math.factorial(n + 1),
                        x=derivatives.min.x)

    return MinMax(min_val, max_val)
