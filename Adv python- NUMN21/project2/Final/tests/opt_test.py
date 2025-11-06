import itertools
import sys
from typing import Type

import numpy as np
from pytest import approx, mark


sys.path.append(".")

from Final.newton import ClassicalNewton
from Final import quasi_newton as qn
from Final import example_funs
from Final.optimization import OptimizationMethod, LS_METHODS, OptimizationProblem

# For sampling starting points
rng = np.random.default_rng(1337)
X0_LIMITS = (-3, 3)
NTRIALS = 5
# NOTE: allow some extra numeric errors
TOL_TEST = 1e-4


OPT_CLS: list[Type[OptimizationMethod]] = [
    ClassicalNewton,
    qn.BadBroyden,
    qn.GoodBroyden,
    qn.SymmetricBroydenSR1,
    qn.DFP,
    qn.BFGS,
]


@mark.parametrize(
    "cls, ls",
    itertools.product(OPT_CLS, LS_METHODS),
)
def test_sphere2d(cls, ls):
    """Should be easy enough even without LS."""
    # no gradients
    example = example_funs.Sphere(2)
    problem = OptimizationProblem(example.f, None)

    method = cls(problem, ls_method=ls)

    # just a few random trials
    for _ in range(NTRIALS):
        x_0 = rng.uniform(X0_LIMITS[0], X0_LIMITS[1], example.ndim)
        steps = method.solve(x_0)[-1]

        assert steps.shape[-1] == example.ndim
        assert steps[-1] == approx(example.global_min, rel=TOL_TEST, abs=TOL_TEST)


@mark.parametrize(
    "cls, ls",
    itertools.product(
        [ClassicalNewton, qn.DFP, qn.BFGS, qn.GoodBroyden],
        ["golden", "wolfe-strong"],
    ),
)
def test_rosenbrock2d_w_grad(cls, ls):
    """This should work, at least for the "better" methods"""
    example = example_funs.Rosenbrock2d()
    problem = OptimizationProblem(example.f, example.g)
    method = cls(problem, ls_method=ls)

    # just a few random trials
    for _ in range(NTRIALS):
        x_0 = rng.uniform(X0_LIMITS[0], X0_LIMITS[1], example.ndim)
        steps = method.solve(x_0)
        assert steps.ndim == 2, "expects N_iter x d_problem"
        assert steps.shape[1] == 2, "expects N_iter x d_problem"
        assert steps[-1] == approx(example.global_min, rel=TOL_TEST, abs=TOL_TEST)
