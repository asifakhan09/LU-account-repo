import abc
from collections.abc import Callable
import enum
import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt


"""
- line search från Fletcher
- how fast is Hessian? Implement #fct evals
- function class?
- Line search meethods not tested independently!
- Jmfr olika BFGS
"""


class StopCrit(enum.Enum):
    """Not used (for now)"""

    RESIDUAL = 1
    CAUCHY = 2


class OptimizationProblem:
    def __init__(self, obj_fct: Callable[[np.ndarray], float], obj_grad=None):
        self.obj_fct = obj_fct
        self.obj_grad = obj_grad


class OptimizationMethod:
    def __init__(self, opt_problem: OptimizationProblem, tol: float = 1e-4):
        self.opt_problem = opt_problem
        self.obj_fct = opt_problem.obj_fct
        self.obj_grad = opt_problem.obj_grad
        self.tol = tol

    # @abc.abstractmethod
    # def hessian(self):
    #     pass

    def compute_grad(self, x: np.ndarray):
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

        fct_evals = 2 * dim  # number of function evaluations done here
        return g, fct_evals
    
    # Didn't understand how exact line search can be implemented so I did a basic inexact algorithm

    def bracketing(self, b0, xk, dk):
        # Algorithm 2 in Diehl
        alpha = 2
        fct = self.obj_fct
        b = b0
        f0 = fct(xk)
        while fct(xk + b * dk) > f0:
            b = b / alpha
        k = 0
        while fct(xk + b * dk) < f0 and k < 1000:
            k += 1
            b = alpha * b
        return b

    def golden_section(self, b0, xk, dk):
        # p. 47 in Diehl
        tol = 1e-4
        tau = (np.sqrt(5) - 1) / 2
        fct = self.obj_fct
        b = self.bracketing(b0, xk, dk)
        a = 0
        L = b
        ml = b - tau * L
        mr = tau * L
        while L >= tol:
            if fct(xk + ml * dk) < fct(xk + mr * dk):
                b = mr
                L = b - a
                mr = ml
                ml = b - tau * L
            else:
                a = ml
                L = b - a
                ml = mr
                mr = a + tau * L
        return np.mean([a, b])

    # From function
    def hessian(self, x: np.ndarray):
        # Object function is used, fct as parameter instead?
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

    
    # Step and method as general functions?
    # ls as parameter for ClassicalNewton so it can be removed from step and method?
    # Or just remove ls altogether because we will always test with line search?



class ClassicalNewton(OptimizationMethod):

    def _newton_step(self, xk: np.ndarray, ls: bool):
        """take a single step"""

        # Starting at x_k, this method calculates x_k+1
        if self.obj_grad is not None:
            gk = self.obj_grad(xk)
        else:
            gk, _ = self.compute_grad(xk)
        Hk = self.hessian(xk)
        dk = la.solve(Hk, (-1) * gk)

        if not ls:  # no line search
            return xk + dk
        else:  # line search
            b0 = 1  # start value for bracketing
            k = self.golden_section(b0, xk, dk)
            return xk + k * dk

    def newton_method(self, x0: np.ndarray, ls: bool = False):
        steps = [x0.copy()]
        xk = x0.copy()
        xold = np.inf * np.ones(len(x0))
        while la.norm(xk - xold) >= self.tol:
            xold = xk.copy()
            xk = self._newton_step(xk, ls)
            steps = np.vstack([steps, xk])
        return steps

class BFGS(OptimizationMethod):

    def _bfgs_step(self, xk, gk, Dk):
        # Notation from Diehl p. 78

        xold = xk.copy()
        gold = gk.copy()
        dk = - Dk @ gk # search direction

        # Update xk with line search
        b0 = 1
        k = self.golden_section(b0, xk, dk)
        xk += k*dk

        # Update Dk
        if self.obj_grad is not None:
            gk = self.obj_grad(xk)
        else:
            gk, _ = self.compute_grad(xk)
        pk = xk - xold
        qk = gk - gold
        # alpha, beta introduced for readability
        alpha = np.dot(pk.T,qk)
        assert alpha > 0, f"pk^T qk = {alpha}" # Should always be true when using Wolfe conditions
        # kolla hur litet alpha blir med Wolfe!
        beta = (1 + (1/alpha)*np.dot(qk.T,np.dot(Dk, qk)))
        Dk += (1/alpha) * ( beta*np.outer(pk,pk.T) - Dk @ np.outer(qk,pk.T) - np.outer(pk,qk.T) @ Dk )
       
        return xk, gk, Dk

    def bfgs_method(self, x0):

        dim = len(x0)
        steps = [x0.copy()]
        #Dk = np.identity(dim) 
        Dk = la.inv(self.hessian(x0))
        hess_app = [Dk.copy()]

        # First iteration:
        xk = x0.copy()
        if self.obj_grad is not None:
            gk = self.obj_grad(xk)
        else:
            gk, _ = self.compute_grad(xk)

        #hessians = [self.hessian(xk)] # for task 12
        xold = np.inf * np.ones(dim)
        i = 0
        while la.norm(xk - xold) >= self.tol:
        # while la.norm(gk) >= self.tol:
            xold = xk.copy()
            print(i)
            xk, gk, Dk = self._bfgs_step(xk, gk, Dk)
            steps.append(xk.copy())
            hess_app.append(Dk.copy())
            i += 1
        return np.vstack(steps), np.stack(hess_app,2)
    

# Testing ----------------------------------------------------------

# grad is [x_1, x_2]
# Solution is [0, 0]
def fct(x):
    return 0.5 * (x[0] ** 2 + x[1] ** 2)



# Rosenbrock
# Solution is [1,1]
def Rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


x0 = np.array([1.5, 3.])
tol = 1e-6
op = OptimizationProblem(Rosenbrock)
# om = ClassicalNewton(op, tol)
om = BFGS(op, tol)

x_steps, Dk_steps = om.bfgs_method(x0)
assert isinstance(x_steps, np.ndarray), "Should be array!"

# print("solution=", x_steps[-1, :])


# Plot for task 5:

xgrid = np.arange(-1.0, 2.0, 0.01)
ygrid = np.arange(-2.0, 4.0, 0.01)
X, Y = np.meshgrid(xgrid, ygrid)
Z = Rosenbrock([X, Y])
levels = [1, 3, 10, 30, 100, 300, 1000, 3000]  # The way I got the correct appearance

# Following assumes 2D domain
fig = plt.figure()
plt.contour(X, Y, Z, levels, colors="black", linewidths=0.5)

for i in range(np.size(x_steps, 0) - 1):
    plt.plot(x_steps[i, 0], x_steps[i, 1], color="blue", markersize=3, marker="o")
    plt.plot(
        [x_steps[i, 0], x_steps[i + 1, 0]],
        [x_steps[i, 1], x_steps[i + 1, 1]],
        color="blue",
        linewidth=0.5,
        linestyle="dashed",
    )
plt.plot(x_steps[-1, 0], x_steps[-1, 1], color="red", markersize=8, marker="*")
plt.text(x_steps[0, 0], x_steps[0, 1] + 0.1, "x0")
plt.text(x_steps[-1, 0] + 0.05, x_steps[-1, 1] - 0.15, "solution")

#plt.show()



