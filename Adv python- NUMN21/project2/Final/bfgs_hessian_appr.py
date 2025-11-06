import sys 
sys.path.append(".")

from Final import quasi_newton as qn
from Final.optimization import OptimizationMethod, LS_METHODS, OptimizationProblem
import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt

class Function:        

    def function(self, x):
        pass

class Rosenbrock(Function):

    def __init__(self):
        self.name = "Rosenbrock"

    def function(self, x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

class Booth(Function):

    def __init__(self):
        self.name = "Booth"

    def function(self, x):
        return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2
    
class ThreeHumpCamel(Function):

    def __init__(self):
        self.name = "ThreeHumpCamel"

    def function(self, x):
        return 2 * x[0] ** 2 - 1.05 * x[0] ** 4 + (x[0] ** 6) / 6 + x[0] * x[1] + x[1] ** 2

    
def hessian(x, fct):
    # Object function is used, fct as parameter instead?
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

def run(fct,name):
    rng = np.random.default_rng(1337)
    X0_LIMITS = (-3, 3)
    TOL_TEST = 1e-6
    dk_steps = []
    def cb_hk(hk):
        dk_steps.append(hk)
    x_0 = rng.uniform(X0_LIMITS[0], X0_LIMITS[1], 2)
    print('x_0 = ',x_0)
    problem = OptimizationProblem(fct, None)
    method = qn.BFGS(problem, ls_method = "wolfe-strong")
    xk_steps = method.solve(x_0, cb_hk = cb_hk)
    print(f"Solution for {name}: ",xk_steps[-1,:])
    np.stack(dk_steps,2)
    return xk_steps, dk_steps

def plot_hess_diff(fcts, names):
    for fct, name in zip(fcts, names, strict=True):
        xk_steps, dk_steps = run(fct,name)
        hk_steps = []
        for i in range(len(xk_steps)-1):
            hk_steps.append(hessian(xk_steps[i,:], fct))
        np.stack(hk_steps,2)
        hk_steps = np.array(hk_steps)
        dk_steps = np.array(dk_steps)

        norm_diff = []
        for i in range(len(hk_steps)):
            Hk = hk_steps[i,:,:]
            Dk = dk_steps[i,:,:]
            rel_diff = la.norm(la.inv(Hk) - Dk, 2) / la.norm(la.inv(Hk), 2)
            norm_diff.append(rel_diff)
        norm_diff = np.array(norm_diff)
        plt.plot(norm_diff, label = name)
    plt.legend()
    plt.title('||Hk^(-1) - Dk|| / ||Hk^(-1)||')
    plt.ylim([0,2])
    plt.show()

rosenbrock = Rosenbrock()
booth = Booth()
threehumpcamel = ThreeHumpCamel()
fcts = [rosenbrock.function, booth.function, threehumpcamel.function]
names = [rosenbrock.name, booth.name, threehumpcamel.name]

plot_hess_diff(fcts,names)













