from sympy import sqrt as sympy_sqrt
from sympy import ln as sympy_ln
from sympy import Piecewise
import math as m

SYMBOLIC_FUNCTION_MAP = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "smoothed_max": lambda x, y: ((x + y) + sympy_sqrt((x - y) ** 2 + 0.05)) * 0.5,
    "smoothed_max_gpu": lambda x, y: ((x + y) + sympy_sqrt((x - y) ** 2 + 0.05)) * 0.5,
    "smoothed_min": lambda x, y: -((-x - y) + sympy_sqrt((-x + y) ** 2 + 0.05) * 0.05),
    "smoothed_min_gpu": lambda x, y: -(
        (-x - y) + sympy_sqrt((-x + y) ** 2 + 0.05) * 0.05
    ),
    "protected_div": lambda x, y: x / y,
    "protected_div_gpu": lambda x, y: x / y,
    "sqrt": sympy_sqrt,
    "sqrt_gpu": sympy_sqrt,
    "inverse": lambda x: 1.0 / x,
    "inverse_gpu": lambda x: 1.0 / x,
    "avg": lambda x, y: 0.5 * (x + y),
    "avg_gpu": lambda x, y: 0.5 * (x + y),
    "ln": sympy_ln,
    "ln_gpu": sympy_ln,
    "sin": lambda x: m.sin(x),
    "sin_gpu": lambda x: m.sin(x),
    "cos": lambda x: m.cos(x),
    "cos_gpu": lambda x: m.cos(x),
    "gt": lambda x, y: Piecewise((1, x > y), (0, x <= y)),
    "gt_gpu": lambda x, y: Piecewise((1, x > y), (0, x <= y)),
    "lt": lambda x, y: Piecewise((1, x < y), (0, x >= y)),
    "lt_gpu": lambda x, y: Piecewise((1, x < y), (0, x >= y)),
    "goe": lambda x, y: Piecewise((1, x >= y), (0, x < y)),
    "goe_gpu": lambda x, y: Piecewise((1, x >= y), (0, x < y)),
    "loe": lambda x, y: Piecewise((1, x <= y), (0, x > y)),
    "loe_gpu": lambda x, y: Piecewise((1, x <= y), (0, x > y)),
    "et": lambda x, y: Piecewise((1, x == y), (0, x != y)),
    "et_gpu": lambda x, y: Piecewise((1, x == y), (0, x != y)),
    "net": lambda x, y: Piecewise((1, x != y), (0, x == y)),
    "net_gpu": lambda x, y: Piecewise((1, x != y), (0, x == y)),
    "sum_linker_template": lambda *args: sum(args),
    "avg_linker_template": lambda *args: (1.0 / len(args)) * sum(args),
    "sum_linker_templalte_numba": lambda *args: sum(args),
    "avg_linker_template_numba": lambda *args: (1.0 / len(args)) * sum(args),
}
