"""
Test the example functions.

"""

import sys

import numpy as np
from pytest import mark

sys.path.append(".")
from Final import example_funs


funs = [
    example_funs.Rosenbrock2d(),
    example_funs.RosenbrockNd(2),
    example_funs.RosenbrockNd(3),
    example_funs.Sphere(1),
    example_funs.Sphere(2),
    example_funs.Sphere(18),
    example_funs.ThreeHumpCamel(),
    example_funs.Booth(),
]


@mark.parametrize("x", [(0, 0), (0, 1), (1, -1), (0.02, -0.333), (-7, 24)])
def test_rosenbrock2d(x):
    f1 = example_funs.Rosenbrock2d()
    f2 = example_funs.RosenbrockNd(2)

    xa = np.array(x)
    assert f1.f(xa) == f2.f(xa)


@mark.parametrize("ex", funs, ids=[str(ex) for ex in funs])
def test_zero_at_min(ex: example_funs.ExampleFun):
    """We exepect all to be zero at min..."""
    assert ex(ex.global_min) == 0


@mark.parametrize("ex", funs, ids=[str(ex) for ex in funs])
def test_approx_min(ex: example_funs.ExampleFun):
    """We shouldnt find something smaller than minimum by chance."""
    limits = (-3, 3)

    fmin = ex(ex.global_min)  # should be zero...

    rng = np.random.default_rng(1337)
    # just a few random trials
    for _ in range(1000):
        x_test = rng.uniform(limits[0], limits[1], ex.ndim)
        assert ex(x_test) >= fmin
