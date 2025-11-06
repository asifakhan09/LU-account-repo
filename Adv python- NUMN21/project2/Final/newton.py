import sys

from numpy.typing import NDArray

sys.path.append(".")
import numpy as np
from Final.optimization import OptimizationMethod


class ClassicalNewton(OptimizationMethod):
    def hessian(self, x: NDArray[np.floating]):
        """Compute the hessian using the finite difference method"""
        fct = self.obj_fct
        dim = len(x)
        H = np.zeros((dim, dim))
        dx = 1e-5  # smaller dx gets close to machine precision
        f0 = fct(x)  # used multiple times
        for i in range(dim):
            for j in range(i + 1):
                if i == j:
                    xplus = x.copy()
                    xplus[i] += dx
                    xminus = x.copy()
                    xminus[i] -= dx
                    H[i, j] = (fct(xplus) - 2 * f0 + fct(xminus)) / (dx**2)
                else:
                    # Finite differences computation
                    xp = x.copy()
                    xp[i] += dx
                    xp[j] += dx
                    xm = x.copy()
                    xm[i] -= dx
                    xm[j] -= dx
                    xipjm = x.copy()
                    xipjm[i] += dx
                    xipjm[j] -= dx
                    ximjp = x.copy()
                    ximjp[i] -= dx
                    ximjp[j] += dx
                    H[i, j] = (fct(xp) - fct(xipjm) - fct(ximjp) + fct(xm)) / (
                        4 * dx**2
                    )

        # Fill in symmetric matrix
        for i in range(dim):  # loop over rows
            for j in range(i + 1, dim):
                H[i, j] = H[j, i]
        return H

    def _init_invhess(self, dim) -> None:
        return None  # not relevant here

    def _update_invhess(self, Hk, s, y) -> None:
        return None  # not relevant here

    def _direction(self, Hk, gk, xk):
        # doesnt need Hk, should be None anyway
        pk = np.linalg.solve(self.hessian(xk), (-1) * gk)
        return pk
