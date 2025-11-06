"""
Chebyquad Testproblem

Course material for the course FMNN25
Version for Python 3.4
Claus Führer (2016)

"""

import numpy as np
from numpy import linspace
dot = np.dot
import scipy.optimize as so
from numpy import array


def T(x, n):
    """
    Recursive evaluation of the Chebychev Polynomials of the first kind
    x evaluation point (scalar)
    n degree
    """
    if n == 0:
        return 1.0
    if n == 1:
        return x
    return 2.0 * x * T(x, n - 1) - T(x, n - 2)


def U(x, n):
    """
    Recursive evaluation of the Chebychev Polynomials of the second kind
    x evaluation point (scalar)
    n degree
    Note d/dx T(x,n)= n*U(x,n-1)
    """
    if n == 0:
        return 1.0
    if n == 1:
        return 2.0 * x
    return 2.0 * x * U(x, n - 1) - U(x, n - 2)


def chebyquad_fcn(x):
    """
    Nonlinear function: R^n -> R^n
    """
    n = len(x)

    def exact_integral(n):
        """
        Generator object to compute the exact integral of
        the transformed Chebychev function T(2x-1,i), i=0...n
        """
        for i in range(n):
            if i % 2 == 0:
                yield -1.0 / (i**2 - 1.0)
            else:
                yield 0.0

    exint = exact_integral(n)

    def approx_integral(i):
        """
        Approximates the integral by taking the mean value
        of n sample points
        """
        return sum(T(2.0 * xj - 1.0, i) for xj in x) / n

    return array([approx_integral(i) - e for i, e in enumerate(exint)])


def chebyquad(x):
    """
    norm(chebyquad_fcn)**2
    """
    chq = chebyquad_fcn(x)
    return dot(chq, chq) # type: ignore


def gradchebyquad(x):
    """
    Evaluation of the gradient function of chebyquad
    """
    chq = chebyquad_fcn(x)
    UM = (
        4.0
        / len(x)
        * array(
            [[(i + 1) * U(2.0 * xj - 1.0, i) for xj in x] for i in range(len(x) - 1)]
        )
    )
    return dot(chq[1:].reshape((1, -1)), UM).reshape((-1,)) # type: ignore


if __name__ == "__main__":
    x = linspace(0, 1, 8) # type: ignore
    xmin = so.fmin_bfgs(
        chebyquad, x, gradchebyquad
    )  # should converge after 18 iterations
