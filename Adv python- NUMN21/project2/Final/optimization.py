from abc import abstractmethod
from typing import Callable, Literal
import numpy as np
from numpy.typing import NDArray


# Possible methods
LS_METHODS = ("exact", "golden", "wolfe-strong", "wolfe-weak")


class OptimizationProblem:
    def __init__(self, obj_fct: Callable, obj_grad: Callable | None = None):
        self.obj_fct = obj_fct
        self.obj_grad = obj_grad


class OptimizationMethod:
    def __init__(
        self,
        opt_problem: OptimizationProblem,
        tol_step=1e-6,
        tol_grad=1e-6,
        tol_ls=1e-6,
        ls_method: Literal["exact", "golden", "wolfe-strong", "wolfe-weak"]
        | None = None,
    ):
        self.opt_problem = opt_problem
        self.obj_fct = opt_problem.obj_fct
        self.obj_grad = opt_problem.obj_grad
        self.tol_grad = tol_grad  # Global convergence tolerance
        self.tol_step = tol_step  # Step tolerance used in ClassicalNewton.solve.
        self.ls_tol = tol_ls  # Line search (step lenght) tolerance
        self.ls_method = ls_method.lower() if ls_method else None

    def __str__(self) -> str:
        if self.obj_grad is None:
            return f"{type(self).__name__}: f={self.obj_fct.__name__} No grad (ls={self.ls_method})"
        else:
            return f"{type(self).__name__}: f={self.obj_fct.__name__} g={self.obj_grad.__name__} (ls={self.ls_method})"

    @abstractmethod
    def _init_invhess(self, dim: int) -> NDArray[np.floating] | None:
        """Init the approximate inverse hessian, if applicable"""
        pass

    @abstractmethod
    def _update_invhess(self, Hk, s, y) -> NDArray[np.floating] | None:
        """Update the approximate inverse hessian, if applicable"""
        pass

    @abstractmethod
    def _direction(
        self,
        Hk: NDArray[np.floating] | None,
        gk: NDArray[np.floating],
        xk: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        pass

    def solve(
        self,
        x0: NDArray[np.floating],
        max_iter: int = 10_000,
        cb_hk: Callable | None = None,
    ) -> NDArray[np.floating]:
        xk = x0.astype(float).copy()
        Hk = self._init_invhess(len(xk))

        if cb_hk:
            cb_hk(Hk)

        # track steps
        steps: list[NDArray[np.floating]] = [xk.copy()]
        gk = self.compute_grad(xk)

        for _ in range(max_iter):
            pk = self._direction(Hk, gk, xk)

            # Guard: want finite and descent direction
            if not np.all(np.isfinite(pk)) or gk @ pk >= 0.0:
                pk = -gk  # fallback to steepest descent

            # Line search
            alpha = self.line_search(xk, pk)

            x_next = xk + alpha * pk
            g_next = self.compute_grad(x_next)

            s = x_next - xk
            y = g_next - gk

            # stopping:
            grad_norm = np.linalg.norm(g_next)
            step_norm = np.linalg.norm(s)

            # small gradient or step?
            if grad_norm <= self.tol_grad or step_norm <= self.tol_step:
                xk = x_next
                steps.append(xk.copy())
                break

            # quasi newton:
            if Hk is not None:
                if np.allclose(s, 0) or np.allclose(y, 0):
                    # Reset if update information is degenerate
                    Hk = self._init_invhess(len(xk))
                else:
                    Hk = self._update_invhess(Hk, s, y)

            if cb_hk:
                cb_hk(Hk)

            # next iteration
            steps.append(x_next.copy())
            xk, gk = x_next, g_next

        return np.vstack(steps)

    def compute_grad(self, x: np.ndarray, dx: float = 1e-8):
        """
        Evaluate provided gradient if given, otherwise compute using finite difference.

        Parameters:
        ----------
        x: np.ndarray
            Current point
        dx: float
            Step size for finite difference

        Returns:
        ----------
        g: np.ndarray
            Gradient.
        """

        if self.obj_grad is not None:
            return np.asarray(
                self.obj_grad(x)
            )  # if analytic available, else finite diff

        fct = self.obj_fct
        dim = len(x)
        g = np.zeros(dim)

        for i in range(dim):
            xplus = x.copy()
            xminus = x.copy()
            xplus[i] = x[i] + dx
            xminus[i] = x[i] - dx
            g[i] = (fct(xplus) - fct(xminus)) / (2 * dx)

        return g

    def ls_wolfe(
        self,
        xk,
        direction,
        alpha0=1.0,
        c1=0.1,  # from Fletcher
        c2=0.01,  # from Fletcher
        max_iter: int = 30,
        expand_factor=9,  # From Fletcher
        strong=True,
    ):
        """
        Perform a line search along search direction starting from xk,
        using the Wolfe conditions (weak/strong).

        Theory:
        ----------
        - We want to find a step length alpha > 0 such that the new point
        x_{k+1} = xk + alpha* direction satisfies:
            (1) Armijo (sufficient decrease) condition:
                f(xk + alpha* direction) ≤ f(xk) + c1 *alpha* ∇f(xk)^T *direction
            (2) Wolfe curvature condition:
                ∇f(xk + alpha* direction)^T *direction ≥ c2 ∇f(xk)^T *direction (weak Wolfe)
                or |∇f(xk + alpha* direction)^T *direction| ≤ -c2 ∇f(xk)^T *direction (strong Wolfe)

        - These conditions guarantee sufficient decrease (progress)
        and that the slope has been reduced enough to approximate a local minimizer.

        Algorithm:
        ----------
        1. Check descent condition at alpha = 0 (g0 < 0). If not, return alpha = 0 (dont move).
        2. Bracketing phase:
        Expand alpha from alpha0 (by `expand_factor`) until either
        Armijo fails or slope changes sign. Then we know the minimizer lies
        between the last two alpha values → call `_zoom`.
        3. Curvature condition:
        If Armijo is satisfied AND Wolfe condition is satisfied, accept alpha.
        4. If max_iter reached without success, return the last alpha as fallback.

        Parameters:
        ----------
        xk : ndarray
            Current point.
        direction : ndarray
            Search direction (should be descent).
        alpha0 : float
            Initial trial step length.
        c1 : float
            Armijo parameter (sufficient decrease, 0 < c1 < 1). Sigma in Fletcher.
        c2 : float
            Wolfe constant (curvature condition, 0 < c2 < 1). Rho in fletcher.
        max_iter : int
            Maximum iterations for bracketing phase.
        expand_factor : float
            Factor to expand trial alpha in bracketing phase. Safeguard. Tau in Fletcher.
        strong : bool, default=True
            If True, enforce strong Wolfe condition; else weak Wolfe.

        Returns:
        ----------
        alpha : float
            Step length satisfying Wolfe conditions (or fallback if not found).

        """

        # φ(α) := f(x_k + α * direction). phi0 = φ(0) = f(x_k).
        # g0 = φ'(0) = ∇f(x_k)^T * direction. For a descent direction we need g0 < 0.
        # If g0 >= 0 then direction is not a descent direction, no positive step will decrease f. Returning 0.0 means don’t move. Maybe handle this differently?
        f = self.obj_fct
        phi0 = f(xk)
        g0 = np.dot(self.compute_grad(xk), direction)

        # Norm of the gradient
        grad_norm = np.linalg.norm(g0)

        # If pk is not descent but gradient is nonzero, fallback
        if g0 >= 0:
            if grad_norm > 1e-12:  # or self.tol or similar
                direction = -g0  # fallback to steepest descent
                g0 = -(grad_norm**2)
            else:
                # gradient is zero or extremely small -> likely at minimum
                return 0.0

        # shorthands
        def phi(alpha):
            return f(xk + alpha * direction)

        def phip(alpha):
            return np.dot(self.compute_grad(xk + alpha * direction), direction)

        # bracketing phase: expand α until we find an interval [a, b] known to contain a point that satisfies Armijo & Wolfe
        alpha_prev, phi_prev = 0.0, phi0
        alpha = alpha0
        for i in range(max_iter):
            phi_alpha = phi(alpha)

            # First check armijo NEGATED, meaning if true Armijo failed so minimizer is between current alpha and previous one.
            # Second check detects that the sample is not improving relative to the previous, so passed minimizer.
            # In either case, stop expanding and enter zoom on [α_prev, α].
            if (phi_alpha > phi0 + c1 * alpha * g0) or (
                i > 0 and phi_alpha >= phi_prev
            ):
                return self._zoom(
                    alpha_prev, alpha, phi0, g0, c1, c2, strong, phi, phip, max_iter
                )

            # If armijo didn't fail, now check wolfe
            grad_alpha = phip(alpha)
            if strong:
                if abs(grad_alpha) <= -c2 * g0:
                    return alpha
            else:
                if grad_alpha >= c2 * g0:
                    return alpha

            # If grad_alpha >= 0 the derivative changed sign, means the minimizer lies between α_prev and α.
            # So then call _zoom to search the bracket where the slope changed sign
            if grad_alpha >= 0:
                return self._zoom(
                    alpha, alpha_prev, phi0, g0, c1, c2, strong, phi, phip, max_iter
                )

            # If none of the stopping conditions hold, we expand the trial step α.
            alpha_prev, phi_prev = alpha, phi_alpha
            alpha *= expand_factor

        return alpha  # fallback

    def _zoom(self, alo, ahi, phi0, g0, c1, c2, strong, phi, phip, max_iter):
        """
        Zoom (sectioning) phase of Wolfe line search.

        Theory:
        ----------
        - Given a bracket [alo, ahi] that is known to contain an acceptable alpha,
        iteratively shrink the bracket until we find a step satisfying
        the Wolfe conditions.

        - At each iteration:
            1. Choose a trial alpha_j inside (alo, ahi).
            * If alo == 0, use quadratic interpolation with (0, f(0), f'(0))
                and (ahi, f(ahi)).
            * Otherwise, use the midpoint of the bracket.
            * A safeguard keeps alpha_j away from endpoints. (If alpha_j ≈ a_hi or a_lo then bracket will barely shrink,
              risk getting stuck with a_hi - a_lo ≈ constant => loop would never terminate.) Note: We don't miss out on
              optimal steps bc of this. We aren't stopping near endpoint points from being solution, just trial step. If
              true minimizing step is near endpoint, we will still converge (shrink bracket) to that solution.
            2. Check Armijo condition:
                If f(xk+alpha_j*direction) > f(0) + c1 *alpha_j* f'(0) OR
                    f(xk+alpha_j *direction) ≥ f(xk+alo *direction),
                then shrink right bound: ahi = alpha_j.
            3. Otherwise check Wolfe condition:
                If satisfied, return alpha_j.
                Else update bracket endpoints based on slope sign so that
                minimizer remains inside.
        - Stop if bracket length < ls_tol or after max_iter.

        Parameters:
        ----------
        alo, ahi : float
            Bracket endpoints.
        phi0 : float
            φ(0) = f(xk).
        g0 : float
            φ'(0) = ∇f(xk)^T*direction.
        c1 : float
            Armijo parameter (sufficient decrease, 0 < c1 < 1). Sigma in Fletcher.
        c2 : float
            Wolfe constant (curvature condition, 0 < c2 < 1). Rho in fletcher.
        strong : bool
            If True, enforce strong Wolfe condition.
        phi : callable
            Function φ(alpha) = f(xk + alpha* direction).
        phip : callable
            Derivative φ'(alpha) = ∇f(xk+alpha*direction)^T *direction.
        max_iter : int
            Maximum iterations of zoom. Safeguard.

        Returns:
        ----------
        alpha : float
            Step length satisfying Wolfe conditions, or midpoint fallback.

        """

        a_lo, a_hi = min(alo, ahi), max(alo, ahi)

        for i in range(max_iter):
            # quadratic interpolation if a_lo == 0, else midpoint. Quadratic is a better guess for
            # candidate alpha we are looking for than just midpoint of interval. We essentially pick
            # the minimizer of the quadratic as the candidate. Quadratic interpolation is q = phi0 + g0*alpha + c*alpha**2
            # where we determine c so that passes through (a_hi, phi_hi). The minimizer is then given by
            # q´= 0 => alpha = (g0*a_hi**2) / (2(phi_hi - phi0 - g0*a_hi)).
            if a_lo == 0.0:
                phi_hi = phi(a_hi)
                denom = phi_hi - phi0 - g0 * a_hi
                if denom != 0:
                    alpha_j = -(g0) * a_hi * a_hi / (2.0 * denom)
                else:
                    alpha_j = 0.5 * (a_lo + a_hi)
            # Can't do reliable quadratic interpolation if a_lo != 0 since then only a_lo, a_hi known. If a_lo == 0 then
            # Three points known: phi0, phip0, a_hi (from bracketing).
            else:
                alpha_j = 0.5 * (a_lo + a_hi)

            # safeguard, avoid being too close to endpoints. Keeps candidate strictly in bracket.
            safeg = 0.1 * (a_hi - a_lo)
            alpha_j = np.clip(alpha_j, a_lo + safeg, a_hi - safeg)

            # Mirrors the bracketing checks in ls_wolfe but now inside the bracket.
            phi_j = phi(alpha_j)
            if (phi_j > phi0 + c1 * alpha_j * g0) or (phi_j >= phi(a_lo)):
                a_hi = alpha_j
            else:
                grad_j = phip(alpha_j)
                if strong:
                    if abs(grad_j) <= -c2 * g0:
                        return alpha_j
                else:
                    if grad_j >= c2 * g0:
                        return alpha_j

                # ensure the bracket endpoints end up on opposite sides of the minimizer
                if grad_j * (a_hi - a_lo) >= 0:
                    a_hi = a_lo  # flip endpoints so the next assignment a_lo = alpha_j produces the proper ordering
                a_lo = alpha_j  # shifts left endpoint to α_j for the next iteration

            # If bracket smaller than ls_tol, return midpoint.
            if abs(a_hi - a_lo) < self.ls_tol:
                return 0.5 * (a_lo + a_hi)

        # If max_iter exhausted also return midpoint.
        return 0.5 * (a_lo + a_hi)

    def _bracketing(self, b0, xk, direction):
        # Algorithm 2 in Diehl
        alpha = 2
        fct = self.obj_fct
        b = b0
        f0 = fct(xk)
        while fct(xk + b * direction) > f0:
            b = b / alpha
        k = 0
        while fct(xk + b * direction) < f0 and k < 1000:
            k += 1
            b = alpha * b
        return b

    def ls_golden_section(
        self,
        xk: np.ndarray,
        direction: np.ndarray,
        b0: float = 1.0,
    ):
        # p. 47 in Diehl
        tol = 1e-4
        tau = (np.sqrt(5) - 1) / 2
        fct = self.obj_fct
        b = self._bracketing(b0, xk, direction)
        a = 0
        L = b
        ml = b - tau * L
        mr = tau * L
        while L >= tol:
            if fct(xk + ml * direction) < fct(xk + mr * direction):
                b = mr
                L = b - a
                mr = ml
                ml = b - tau * L
            else:
                a = ml
                L = b - a
                ml = mr
                mr = a + tau * L
        return np.mean([a, b]).item()

    def ls_exact(self, x, direction: np.ndarray, max_step_size=1.0, n_trials=50):
        candidates = np.linspace(0.0, max_step_size, n_trials, dtype=float)
        best = float("inf")
        best_alpha = 1.0
        for alpha in candidates:
            step = alpha * direction
            fval = self.obj_fct(x + step)
            if fval < best:
                best = fval
                best_alpha = alpha

        return float(best_alpha)

    def line_search(
        self,
        xk: np.ndarray,
        direction: np.ndarray,
    ) -> float:
        """Use the chosen line search method"""
        if self.ls_method is None:
            alpha = 1.0
        elif self.ls_method == "exact":
            alpha = self.ls_exact(xk, direction)
        elif self.ls_method == "golden":
            alpha = self.ls_golden_section(xk, direction)
        elif self.ls_method == "wolfe-weak":
            alpha = self.ls_wolfe(xk, direction, strong=False)
        elif self.ls_method == "wolfe-strong":
            alpha = self.ls_wolfe(xk, direction, strong=True)
        else:
            raise ValueError(f"what is {self.ls_method}?")

        return alpha
