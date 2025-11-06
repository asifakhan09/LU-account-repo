import numpy as np

from Final.optimization import OptimizationMethod
from scipy import optimize as sp_optimize


class OptDummy(OptimizationMethod):
    def solve(self, x0: np.ndarray, max_iter: int = 10_000):
        fmin = float("inf")
        sol = x0
        rng = np.random.default_rng()
        xx = np.empty((max_iter, x0.size))
        for i in range(max_iter):
            x_try = rng.normal(0, 10, x0.shape)
            fv = self.obj_fct(x_try)
            xx[i] = x_try
            if fv < fmin:
                fmin = fv
                sol = x_try

        # just insert best
        xx[-1] = sol
        return sol


class ScipyWrapper(OptimizationMethod):
    def solve(self, x0: np.ndarray, max_iter: int = 10_000):
        steps = []

        def cb(xk):
            steps.append(xk)

        res = sp_optimize.minimize(
            self.obj_fct,
            x0,
            jac=self.obj_grad if self.obj_grad is not None else "2-point",
            method="BFGS",
            callback=cb,
        )
        assert (res.x == steps[-1]).all()

        return np.stack(steps, 0)
