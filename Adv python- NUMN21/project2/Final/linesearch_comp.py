import sys 
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from optimization import OptimizationMethod, OptimizationProblem
from newton import ClassicalNewton



#################### Testing Line Search separately from Newton #################

#Rosenbrock, solution is [1,1]
def rosenbrock(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

#Analytical grad for steepest descent, for now testing line search method separately
def rosenbrock_grad(x):
    return np.array([
        -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0]),
        200*(x[1]-x[0]**2)
    ])

#Setup problem and method
op = OptimizationProblem(rosenbrock, rosenbrock_grad)
x0 = np.array([-0.5, 3.0])
tol = 1e-6
method = OptimizationMethod(op, x0, tol)

#Steepest descent with Wolfe line search 
steps = [x0.copy()]
xk = x0.copy()
xold = np.inf*np.ones_like(x0)

#With len(steps) <50 got "Final point: [-0.39704165 0.15243965] f(x)= 1.954431911317191", but with 500 or 5000 it is very
#close to converging to [1,1] ([0.9996827  0.99936444]  f(x)= 1.007927564178627e-07), so seems to work correctly. .
while np.linalg.norm(xk - xold) >= tol and len(steps) < 5000:
    xold = xk.copy()
    pk = -method.compute_grad(xk)  #steepest descent direction
    alpha = method.ls_wolfe(xk, pk,
                                     alpha0=1.0,
                                     c1=0.1,  #Fletchers Armijo param
                                     c2=0.01, #Fletchers curvature param
                                     max_iter= 30,
                                     expand_factor=9,
                                     strong=True)
    xk = xk + alpha * pk
    steps.append(xk.copy())

steps = np.vstack(steps)

#Solution is [1,1] so expect output something like "Final point: [0.999...  0.999...]  f(x)= very close to 0"
print("Final point:", steps[-1], " f(x)=", rosenbrock(steps[-1]))

#Plot path on Rosenbrock contours
xgrid = np.linspace(-1.5, 2.0, 400)
ygrid = np.linspace(-1.0, 4.0, 400)
X, Y = np.meshgrid(xgrid, ygrid)
Z = rosenbrock([X, Y])

plt.figure(figsize=(6, 5))
plt.contour(X, Y, Z, levels=np.logspace(-1, 3.5, 20), cmap="jet")

plt.plot(steps[:,0], steps[:,1], marker='o', color='red', linewidth=1)
plt.scatter(steps[0,0], steps[0,1], marker = 'x', color='blue', s=60, label="Start")
plt.scatter(steps[-1,0], steps[-1,1], marker='x', color='green', s=60, label="End")
plt.legend()
plt.title("Strong Wolfe Line Search on Rosenbrock")
plt.show() 


########### Test Wolfe Line search vs. exact line search with Newton #########

op = OptimizationProblem(rosenbrock, rosenbrock_grad)
x0 = np.array([-0.5, 3.0])


#Newton without line search
newton_none = ClassicalNewton(op, ls_method=None)
steps_none = newton_none.solve(x0.copy())

#Newton with exact line search
newton_exact = ClassicalNewton(op, ls_method='exact')
steps_exact = newton_exact.solve(x0.copy()) 

#Newton with Weak Wolfe line search
newton_wolfeW = ClassicalNewton(op, ls_method='wolfe-weak')
steps_wolfeW = newton_wolfeW.solve(x0.copy())

#Newton with Strong Wolfe line search
newton_wolfeS = ClassicalNewton(op, ls_method='wolfe-strong')
steps_wolfeS = newton_wolfeS.solve(x0.copy())



#Solutions
print("Newton without line search solution:", steps_none[-1], " f=", rosenbrock(steps_none[-1]))
print("Newton with exact line search solution:", steps_exact[-1], " f=", rosenbrock(steps_exact[-1]))
print("Newton with Weak Wolfe line search solution:", steps_wolfeW[-1], " f=", rosenbrock(steps_wolfeW[-1]))
print("Newton with Strong Wolfe line search solution:", steps_wolfeS[-1], " f=", rosenbrock(steps_wolfeS[-1]))


#Plot contour + both trajectories
xgrid = np.linspace(-1.5, 2.0, 400)
ygrid = np.linspace(-1.0, 4.0, 400)
X, Y = np.meshgrid(xgrid, ygrid)
Z = rosenbrock([X, Y])

plt.figure(figsize=(7, 6))
plt.contour(X, Y, Z, levels=np.logspace(-1, 3.5, 20), cmap="jet")


#Newton None path
plt.plot(steps_none[:,0], steps_none[:,1], marker='o', color='green', linewidth=1, label="Newton None")
plt.scatter(steps_none[0,0], steps_none[0,1], color='green', s=60, marker='x')
plt.scatter(steps_none[-1,0], steps_none[-1,1], color='green', s=80, marker='*')

#Exact Newton path
plt.plot(steps_exact[:,0], steps_exact[:,1], marker='o', color='blue', linewidth=1, label="Exact Newton")
plt.scatter(steps_exact[0,0], steps_exact[0,1], color='blue', s=60, marker='x')
plt.scatter(steps_exact[-1,0], steps_exact[-1,1], color='blue', s=80, marker='*')

#Weak Wolfe Newton path
plt.plot(steps_wolfeW[:,0], steps_wolfeW[:,1], marker='o', color='orange', linewidth=1, label="Newton + Weak Wolfe")
plt.scatter(steps_wolfeW[0,0], steps_wolfeW[0,1], color='orange', s=60, marker='x')
plt.scatter(steps_wolfeW[-1,0], steps_wolfeW[-1,1], color='orange', s=80, marker='*')

#Strong Wolfe Newton path
plt.plot(steps_wolfeS[:,0], steps_wolfeS[:,1], marker='o', color='red', linewidth=1, label="Newton +Strong Wolfe")
plt.scatter(steps_wolfeS[0,0], steps_wolfeS[0,1], color='red', s=60, marker='x')
plt.scatter(steps_wolfeS[-1,0], steps_wolfeS[-1,1], color='red', s=80, marker='*')

plt.legend()
plt.title("Newtons Method with different line searches on Rosenbrock")
plt.show()



#Function values vs iteration plot, shows convergence difference. 

#The plot confirms the theoretical convergence behavior of Newtons method. 
#Newton without line search has fixed step size 1 so takes large steps, reaching minimum quickly (quadratic converge).
#Weak wolfe has less strict conditions so can take larger steps, can be more unstable but works well for this case making it a bit faster.
#Strong wolfe has more restrictions on step size, meaning more cautious convergence, so a bit slower but more stable (didnt matter for this case though).
#Exact line search is slowest. With exact line search, the optimizer often chooses a shorter step that minimizes along the Newton direction but “pulls back” from the full Newton step. This destroys the quadratic convergence property.
 
fvals_none = [rosenbrock(x) for x in steps_none]
fvals_exact = [rosenbrock(x) for x in steps_exact]
fvals_wolfeW = [rosenbrock(x) for x  in steps_wolfeW]
fvals_wolfeS = [rosenbrock(x) for x in steps_wolfeS]

plt.figure(figsize=(6, 5))
plt.semilogy(fvals_none, marker='o', color='green', label="Newton None")
plt.semilogy(fvals_exact, marker='o', color='blue', label="Exact Newton")
plt.semilogy(fvals_wolfeW, marker='o', color='orange', label="Newton + Weak Wolfe")
plt.semilogy(fvals_wolfeS, marker='o', color='red', label="Newton + Strong Wolfe")

plt.xlabel("Iteration")
plt.ylabel("f(x)")
plt.title("Function Value vs Iteration (log scale)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()