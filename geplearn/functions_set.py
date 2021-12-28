import math as m
from numbers import Number
from numba import njit, cuda


@njit(nogil=True)
def protected_div(x1: Number, x2: Number) -> Number:
    """
    Protected division. If the denominator is less than 1e-6, it will be
    replaced by 1 to avoid errors.

    Parameters
    ----------
    x1: Number
        Numerator.
    x2: Number
        Denominator.

    Returns
    -------
    Number
        Result of the division between x1 and x2
    """
    if abs(x2) < 1e-6:
        x2 = 1

    return x1 / x2


@cuda.jit("float64(float64, float64)", device=True, inline=True)
def protected_div_gpu(x1: float, x2: float) -> float:
    """
    GPU version of the protected_div function.

    Parameters
    ----------
    x1: float
        Numerator.
    x2: float
        Denominator.

    Returns
    -------
    Number
        Result of the division between x1 and x2
    """
    if abs(x2) < 1e-6:
        x2 = 1

    return x1 / x2


protected_div_gpu.__name__ = "protected_div_gpu"


@njit(nogil=True)
def inverse(x1: Number) -> Number:
    """
    Computes the inverse of a number. If the number is less than 1e-6, it
    returns 1 to avoid errors.

    Parameters
    ----------
    x1: Number
        Number to invert.

    Returns
    -------
    float
        Inverse of x1.
    """
    if abs(x1) < 1e-6:
        return 1.0

    return 1.0 / x1


@cuda.jit("float64(float64)", device=True, inline=True)
def inverse_gpu(x1: float) -> float:
    """
    GPU version of the inverse function.

    Parameters
    ----------
    x1: float
        Number to invert.

    Returns
    -------
    float
        Inverse of x1.
    """
    if abs(x1) < 1e-6:
        return 1.0

    return 1.0 / x1


inverse_gpu.__name__ = "inverse_gpu"


@njit(nogil=True)
def avg(x1: Number, x2: Number) -> float:
    """
    Computes the average between two numbers.

    Parameters
    ----------
    x1: Number
        A number.
    x2: Number
        Another number.

    Returns
    -------
    float
        Average between x1 and x2.
    """
    return 0.5 * (x1 + x2)


@cuda.jit("float64(float64, float64)", device=True, inline=True)
def avg_gpu(x1: float, x2: float) -> float:
    """
    GPU version of the avg function.

    Parameters
    ----------
    x1: float
        A number.
    x2: float
        Another number.

    Returns
    -------
    float
        Average between x1 and x2.
    """
    return 0.5 * (x1 + x2)


avg_gpu.__name__ = "avg_gpu"


@njit(nogil=True)
def neg(x1: Number) -> Number:
    """
    Computes the negative of a number.

    Parameters
    ----------
    x1: Number
        A number.

    Returns
    -------
    Number
        Negative of x1
    """
    return -x1


@cuda.jit("float64(float64)", device=True, inline=True)
def neg_gpu(x1: float) -> float:
    """
    GPU version of the negative function.

    Parameters
    ----------
    x1: float
        A number.

    Returns
    -------
    Number
        Negative of x1
    """
    return -x1


neg_gpu.__name__ = "neg_gpu"


@njit(nogil=True)
def sqrt(x1: Number) -> float:
    """
    Computes the square root of a number. If the number is less than 0, it
    returns the same number (identity function).

    Parameters
    ----------
    x1: Number
        A number.

    Returns
    -------
    float
        Square root of x1.
    """
    if x1 < 0.0:
        return x1

    return m.sqrt(x1)


@cuda.jit("float64(float64)", device=True, inline=True)
def sqrt_gpu(x1: Number) -> float:
    """
    GPU version of the square root function.

    Parameters
    ----------
    x1: float
        A number.

    Returns
    -------
    float
        Square root of x1.
    """
    if x1 < 0.0:
        return x1

    return m.sqrt(x1)


sqrt_gpu.__name__ = "sqrt_gpu"


@njit(nogil=True)
def ln(x1: Number) -> float:
    """
    Computes the natural logarithm of a number. If it is less than or equal
    to 0, it returns the same number (identity function).

    Parameters
    ----------
    x1: Number
        A number.

    Returns
    -------
    float
        Natural logarithm of x1.
    """
    if x1 <= 0.0:
        return x1

    return m.log(x1)


@cuda.jit("float64(float64)", device=True, inline=True)
def ln_gpu(x1: Number) -> float:
    """
    GPU version of the logarithm function.

    Parameters
    ----------
    x1: float
        A number.

    Returns
    -------
    float
        Natural logarithm of x1.
    """
    if x1 <= 0.0:
        return x1

    return m.log(x1)


ln_gpu.__name__ = "ln_gpu"


@njit(nogil=True)
def sin(x1: Number) -> Number:
    """
    Computes the sine of a number.

    Parameters
    ----------
    x1: Number
        A number.

    Returns
    -------
    float
        Sine of x1.
    """
    return m.sin(x1)


@cuda.jit("float64(float64)", device=True, inline=True)
def sin_gpu(x1: float) -> float:
    """
    GPU version of the sine function.

    Parameters
    ----------
    x1: float
        A number.

    Returns
    -------
    float
        Sine of x1.
    """
    return m.sin(x1)


sin_gpu.__name__ = "sin_gpu"


@njit(nogil=True)
def cos(x1: Number) -> Number:
    """
    Computes the cosine of a number.

    Parameters
    ----------
    x1: Number
        A number.

    Returns
    -------
    float
        Cosine of x1.
    """
    return m.cos(x1)


@cuda.jit("float64(float64)", device=True, inline=True)
def cos_gpu(x1: float) -> float:
    """
    GPU version of the cosine function.

    Parameters
    ----------
    x1: float
        A number.

    Returns
    -------
    float
        Cosine of x1.
    """
    return m.cos(x1)


cos_gpu.__name__ = "cos_gpu"


@njit(nogil=True)
def lt(x1: Number, x2: Number) -> float:
    """
    "Less than" function. Returns 1. if x1 is less than x2, zero otherwise.

    Parameters
    ----------
    x1: Number
        A number.
    x2: Number
        Another number.

    Returns
    -------
    float
        1. if x1<x2, else 0.
    """
    if x1 < x2:
        return 1.0

    return 0.0


@cuda.jit("float64(float64, float64)", device=True, inline=True)
def lt_gpu(x1: float, x2: float) -> float:
    """
    GPU version of the less than function.

    Parameters
    ----------
    x1: float
        A number.
    x2: float
        Another number.

    Returns
    -------
    float
        1. if x1<x2, else 0.
    """
    if x1 < x2:
        return 1.0

    return 0.0


lt_gpu.__name__ = "lt_gpu"


@njit(nogil=True)
def gt(x1: Number, x2: Number) -> float:
    """
    "Greater than" function. Returns 1. if x1 is greater than x2, zero
    otherwise.

    Parameters
    ----------
    x1: Number
        A number.
    x2: Number
        Another number.

    Returns
    -------
    float
        1. if x1>x2, else 0.
    """
    if x1 > x2:
        return 1.0

    return 0.0


@cuda.jit("float64(float64, float64)", device=True, inline=True)
def gt_gpu(x1: float, x2: float) -> float:
    """
    GPU version of the greater than function.

    Parameters
    ----------
    x1: float
        A number.
    x2: float
        Another number.

    Returns
    -------
    float
        1. if x1>x2, else 0.
    """
    if x1 > x2:
        return 1.0

    return 0.0


gt_gpu.__name__ = "gt_gpu"


@njit(nogil=True)
def loe(x1: Number, x2: Number) -> float:
    """
    "Less than or equal" function. Returns 1. if x1 is less than or equal x2,
    zero otherwise.

    Parameters
    ----------
    x1: Number
        A number.
    x2: Number
        Another number.

    Returns
    -------
    float
        1. if x1<=x2, else 0.
    """
    if x1 <= x2:
        return 1.0

    return 0.0


@cuda.jit("float64(float64, float64)", device=True, inline=True)
def loe_gpu(x1: float, x2: float) -> float:
    """
    GPU version of the less than or equal function.

    Parameters
    ----------
    x1: float
        A number.
    x2: float
        Another number.

    Returns
    -------
    float
        1. if x1<=x2, else 0.
    """
    if x1 <= x2:
        return 1.0

    return 0.0


loe_gpu.__name__ = "loe_gpu"


@njit(nogil=True)
def goe(x1: Number, x2: Number) -> float:
    """
    "Greater than or equal" function. Returns 1. if x1 is greater than or equal
    x2, zero otherwise.

    Parameters
    ----------
    x1: Number
        A number.
    x2: Number
        Another number.

    Returns
    -------
    float
        1. if x1>=x2, else 0.
    """
    if x1 >= x2:
        return 1.0
    return 0.0


@cuda.jit("float64(float64, float64)", device=True, inline=True)
def goe_gpu(x1: float, x2: float) -> float:
    """
    GPU version of the greater than or equal function.

    Parameters
    ----------
    x1: float
        A number.
    x2: float
        Another number.

    Returns
    -------
    float
        1. if x1>=x2, else 0.
    """
    if x1 >= x2:
        return 1.0
    return 0.0


goe_gpu.__name__ = "goe_gpu"


@njit(nogil=True)
def et(x1: Number, x2: Number) -> float:
    """
    "Equals to" function. Returns 1. if x1 is equal to x2, zero otherwise.

    Parameters
    ----------
    x1: Number
        A number.
    x2: Number
        Another number.

    Returns
    -------
    float
        1. if x1==x2, else 0.
    """
    if x1 == x2:
        return 1.0
    return 0.0


@cuda.jit("float64(float64, float64)", device=True, inline=True)
def et_gpu(x1: float, x2: float) -> float:
    """
    GPU version of the equals to function.

    Parameters
    ----------
    x1: float
        A number.
    x2: float
        Another number.

    Returns
    -------
    float
        1. if x1==x2, else 0.
    """
    if x1 == x2:
        return 1.0
    return 0.0


@njit(nogil=True)
def net(x1: Number, x2: Number) -> float:
    """
    "Not Equal to" function. Returns 1. if x1 not equal to x2, zero otherwise.

    Parameters
    ----------
    x1: Number
        A number.
    x2: Number
        Another number.

    Returns
    -------
    float
        1. if x1!=x2, else 0.
    """
    if x1 != x2:
        return 1.0
    return 0.0


@cuda.jit("float64(float64, float64)", device=True, inline=True)
def net_gpu(x1: float, x2: float) -> float:
    """
    GPU version of the not equal to function.

    Parameters
    ----------
    x1: float
        A number.
    x2: float
        Another number.

    Returns
    -------
    float
        1. if x1!=x2, else 0.
    """
    if x1 != x2:
        return 1.0
    return 0.0


net_gpu.__name__ = "net_gpu"


@njit(nogil=True)
def smoothed_max(x1: Number, x2: Number) -> float:
    """
    Smoothed (differentiable) version of the max function.

    Parameters
    ----------
    x1: Number
        A number.
    x2: Number
        Another number.

    Returns
    -------
    float
        Smoothed max of x1 and x2
    """
    return ((x1 + x2) + sqrt((x1 - x2) ** 2 + 0.05)) * 0.5


@cuda.jit("float64(float64, float64)", device=True, inline=True)
def smoothed_max_gpu(x1: float, x2: float) -> float:
    """
    GPU version of the smoothed max function.

    Parameters
    ----------
    x1: float
        A number.
    x2: float
        Another number.

    Returns
    -------
    float
        Smoothed max of x1 and x2
    """
    return ((x1 + x2) + sqrt((x1 - x2) ** 2 + 0.05)) * 0.5


smoothed_max_gpu.__name__ = "smoothed_max_gpu"


@njit(nogil=True)
def smoothed_min(x1: Number, x2: Number) -> float:
    """
    Smoothed (differentiable) version of the min function.

    Parameters
    ----------
    x1: Number
        A number.
    x2: Number
        Another number.

    Returns
    -------
    float
        Smoothed min of x1 and x2
    """
    return -((-x1 - x2) + sqrt(-x1 + x2) ** 2 + 0.05) * 0.5


@cuda.jit("float64(float64, float64)", device=True, inline=True)
def smoothed_min_gpu(x1: float, x2: float) -> float:
    """
    GPU version of the smoothed min function.

    Parameters
    ----------
    x1: float
        A number.
    x2: float
        Another number.

    Returns
    -------
    float
        Smoothed min of x1 and x2
    """
    return -((-x1 - x2) + sqrt(-x1 + x2) ** 2 + 0.05) * 0.5


smoothed_min_gpu.__name__ = "smoothed_min_gpu"
