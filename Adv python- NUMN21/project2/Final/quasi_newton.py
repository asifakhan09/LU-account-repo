from abc import abstractmethod
from typing import Any
import numpy as np
from numpy.typing import NDArray

from Final.optimization import OptimizationMethod, OptimizationProblem


class QuasiNewtonBase(OptimizationMethod):
    """
    Base class for Quasi-Newton methods that carry an inverse-Hessian
    approximation H_k and use your golden-section line search.
    Subclasses implement _update(Hk, s, y, xk, gk, xk1, gk1) -> H_{k+1}.
    """

    def __init__(
        self,
        opt_problem: OptimizationProblem,
        tol_step: float = 0.000001,
        tol_grad: float = 0.000001,
        tol_ls: float = 0.000001,
        ls_method: Any | None = None,
        H0: np.ndarray | None = None,
    ):
        super().__init__(opt_problem, tol_step, tol_grad, tol_ls, ls_method)
        self.H0 = H0  # if None -> identity

    # same central-difference fallback as your ClassicalNewton.compute_grad
    def _compute_grad(self, x: np.ndarray):
        if self.obj_grad is not None:
            return self.obj_grad(x), 0
        fct = self.obj_fct
        dim = len(x)
        g = np.zeros(dim)
        dx = 1e-8
        for i in range(dim):
            xplus = x.copy()
            xminus = x.copy()
            xplus[i] = x[i] + dx
            xminus[i] = x[i] - dx
            g[i] = (fct(xplus) - fct(xminus)) / (2 * dx)
        return g, 2 * dim

    def _init_invhess(self, dim: int) -> NDArray[np.floating]:
        return self.H0.copy() if self.H0 is not None else np.eye(dim)

    @abstractmethod
    def _update_invhess(self, Hk, s, y) -> NDArray[np.floating]:
        """Update the approximate inverse hessian."""
        pass  # subclass does this!

    def _direction(self, Hk, gk, xk) -> NDArray[np.floating]:
        # ignore xk
        assert Hk is not None, "needs aprrox hessian inverse"
        return -Hk @ gk  # p_k = -H_k g_k


# --------- Good Broyden: rank-1 on G (= H^{-1}), Sherman–Morrison ----------
class GoodBroyden(QuasiNewtonBase):
    r"""
    Update B_{k+1} = B_k + ((y - B_k s) s^T)/(s^T s)
    Use Sherman–Morrison to update H_{k+1} = B_{k+1}^{-1} without inverting:
      H_{k+1} = H_k - (H_k u v^T H_k)/(1 + v^T H_k u),
      u = y - B_k s,  v = s/(s^T s).
    Implemented by solving H_k w = s (so w = B_k s) → u = y - w.
    """

    def _update_invhess(self, Hk, s, y):
        ss = float(s @ s)
        if ss < 1e-18:
            return Hk
        # w = B_k s via H_k w = s
        w = np.linalg.solve(Hk, s)
        u = y - w
        v = s / ss
        denom = 1.0 + v @ (Hk @ u)
        if abs(denom) < 1e-18:
            return Hk
        Hu = Hk @ u
        vTH = v @ Hk
        return Hk - np.outer(Hu, vTH) / denom


# --------- Bad Broyden: rank-1 directly on H ----------
class BadBroyden(QuasiNewtonBase):
    r"""
    H_{k+1} = H_k + ((s - H_k y) y^T)/(y^T y)
    (satisfies secant H_{k+1} y = s; generally non-symmetric).
    """

    def _update_invhess(self, Hk, s, y):
        yy = float(y @ y)
        if yy < 1e-18:
            return Hk
        Hy = Hk @ y
        return Hk + np.outer(s - Hy, y) / yy


# --------- Symmetric Broyden (SR1) ----------
class SymmetricBroydenSR1(QuasiNewtonBase):
    r"""
    SR1 (symmetric rank-1):
      r = s - H_k y
      H_{k+1} = H_k + (r r^T)/(r^T y)
    Use a skip rule if |r^T y| is too small.
    """

    def _update_invhess(self, Hk, s, y):
        r = s - Hk @ y
        denom = float(r @ y)
        if abs(denom) < 1e-12:
            return Hk  # skip for stability
        return Hk + np.outer(r, r) / denom


# --------- DFP (Davidon–Fletcher–Powell, rank-2, symmetric) ----------
class DFP(QuasiNewtonBase):
    r"""
    H_{k+1} = H_k + (s s^T)/(y^T s) - (H_k y y^T H_k)/(y^T H_k y)
    """

    def _update_invhess(self, Hk, s, y):
        ys = float(y @ s)
        if abs(ys) < 1e-18:
            return Hk  # skip on bad curvature
        Hy = Hk @ y
        yHy = float(y @ Hy)
        if abs(yHy) < 1e-18:
            return Hk
        return Hk + np.outer(s, s) / ys - np.outer(Hy, Hy) / yHy


# --------- BFGS (rank-2, symmetric, positive-definite if y^T s > 0) ----------
class BFGS(QuasiNewtonBase):
    r"""
    ρ = 1/(y^T s)
    H_{k+1} = (I - ρ s y^T) H_k (I - ρ y s^T) + ρ s s^T
    """

    def _update_invhess(self, Hk, s, y):
        ys = float(y @ s)
        if ys <= 1e-18:
            return Hk  # maintain stability
        rho = 1.0 / ys
        ident = np.eye(len(s))
        V = ident - rho * np.outer(s, y)
        return V @ Hk @ V.T + rho * np.outer(s, s)


if __name__ == "__main__":
    # problem
    def rosen(x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    def rosen_grad(x):  # analytic grad for plotting ||∇f||
        dfdx = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
        dfdy = 200 * (x[1] - x[0] ** 2)
        return np.array([dfdx, dfdy], dtype=float)

    x0 = np.array([-0.5, 3.0], dtype=float)
    op = OptimizationProblem(rosen)

    methods: list[tuple[str, QuasiNewtonBase]] = [
        ("Good Broyden", GoodBroyden(op, ls_method="golden")),
        ("Bad Broyden", BadBroyden(op, ls_method="golden")),
        ("SR1", SymmetricBroydenSR1(op, ls_method="golden")),
        ("DFP", DFP(op, ls_method="golden")),
        ("BFGS", BFGS(op, ls_method="golden")),
    ]

    results = {}
    summary = []
    for name, M in methods:
        steps = M.solve(x0, max_iter=20_000)  # big cap for bad Broyden
        fx = np.array([rosen(s) for s in steps])
        gnorm = np.array([np.linalg.norm(rosen_grad(s)) for s in steps])
        results[name] = {"steps": steps, "fx": fx, "gnorm": gnorm}
        summary.append((name, len(steps) - 1, steps[-1], fx[-1], gnorm[-1]))

    # pretty print summary
    print(
        "Method              iters   final x                     f(x_final)         ||grad||"
    )
    for name, iters, xfin, ffin, gfin in summary:
        print(
            f"{name:16s} {iters:6d}   [{xfin[0]: .8f}, {xfin[1]: .8f}]   {ffin: .3e}   {gfin: .3e}"
        )
