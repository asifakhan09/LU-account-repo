"""
Idea, try all methods with different parameters on different test functions

- measure: compute time, final error
- try different: LS,
"""

from pathlib import Path
import sys
import time
from typing import Type
import polars as pl
import numpy as np


sys.path.append(".")

from Final.newton import ClassicalNewton
from Final import quasi_newton as qn
from Final import example_funs
from Final.optimization import OptimizationMethod, LS_METHODS, OptimizationProblem
from Final.dummy import OptDummy, ScipyWrapper


MAX_ITER = 1000
n_start = 20
DATA_PATH = Path(f"data/bench_{n_start}.parquet")
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)


EXAMPLES = [
    # example_funs.Sphere(1),
    example_funs.Sphere(2),
    example_funs.Sphere(15),
    example_funs.Booth(),
    example_funs.RosenbrockNd(2),
    example_funs.ThreeHumpCamel(),  # NOTE: tricky!
]

OPT_METHODS = [
    OptDummy,
    ClassicalNewton,
    qn.GoodBroyden,
    qn.SymmetricBroydenSR1,
    qn.DFP,
    qn.BFGS,
    ScipyWrapper,
]


def run_bench(
    ex: example_funs.ExampleFun,
    opt_methods: list[Type[OptimizationMethod]],
    ls: str | None = None,
    sigma_start=5.0,
):
    """Compare..."""

    print("\n\n======================\n", ex)
    rng = np.random.default_rng(1337)

    # collect results
    all_results = []

    for cls in opt_methods:
        # TODO: with and without grad?
        problem = OptimizationProblem(ex.f)

        assert ls is None or ls in LS_METHODS

        m = cls(problem, ls_method=ls)

        times = np.empty(n_start)
        errs = np.empty(n_start)
        for i in range(n_start):
            # random start point
            x0 = rng.normal(ex.global_min, sigma_start, ex.ndim)

            ts = time.time()
            steps = m.solve(x0, max_iter=MAX_ITER)
            t_compute = time.time() - ts

            assert steps.shape[-1] == ex.ndim
            solution = steps[-1]

            err = np.linalg.norm(ex.global_min - solution)
            times[i] = t_compute
            errs[i] = err
        print(
            f"\n{cls.__name__:<24}| Error: min={errs.min():.4f}, avg={errs.mean():.4f}, max={errs.max():.4f} | t={times.mean() * 1000:.1f} ms"
        )

        all_results.append(
            {
                "example": str(ex),
                "method": cls.__name__,
                "ls_method": ls,
                "e_min": errs.min(),
                "e_avg": errs.mean(),
                "e_max": errs.max(),
                "time_ms": times.mean() * 1000,
                "errors": errs.tolist(),
            }
        )
    return all_results


if __name__ == "__main__":
    print("hello benchmark")
    all_results = []
    for ex in EXAMPLES:
        for ls in ["exact", "golden", "wolfe-weak", "wolfe-strong"]:
            res_ex = run_bench(ex, OPT_METHODS, ls)
            all_results.extend(res_ex)

    df = pl.DataFrame(all_results)

    print(df)

    df.write_parquet(DATA_PATH)
