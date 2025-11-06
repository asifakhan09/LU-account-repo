import numpy as np
import matplotlib.pyplot as plt


class OptimizationProblem:

    def __init__(self, obj_fct, obj_grad = None):
        self.obj_fct = obj_fct
        self.obj_grad = obj_grad
        

class OptimizationMethod:

    def __init__(self, opt_problem: OptimizationProblem, x0: np.ndarray, tol, ls_tol = 1e-6):
        self.opt_problem = opt_problem
        self.obj_fct = opt_problem.obj_fct 
        self.obj_grad = opt_problem.obj_grad
        self.x0 = x0
        self.tol = tol #global convergence tolerance
        self.ls_tol = ls_tol #line search (step lenght) tolerance


    def compute_grad(self, x: np.ndarray, dx = 1e-8):

        if self.obj_grad is not None:
            return np.asarray(self.obj_grad(x))  #if analytic available, else finite diff
        
        #finite diff
        assert len(x) == len(self.x0)
        fct = self.obj_fct
        dim = len(x)
        g = np.zeros(dim)

        for i in range(dim):
            xplus = x.copy()
            xminus = x.copy()
            xplus[i] = x[i] + dx
            xminus[i] = x[i] - dx
            g[i] = (fct(xplus) - fct(xminus))/(2*dx)

        return g
    
    def hessian(self, x: np.ndarray, dx = 1e-5):

        # Object function is used, fct as parameter instead?
        assert len(x) == len(self.x0)
        fct = self.obj_fct
        dim = len(x)
        H = np.zeros((dim,dim))
        f0 = fct(x) #used multiple times

        for i in range(dim):
            for j in range(i+1):
                if i == j:
                    xplus = x.copy(); xplus[i] += dx
                    xminus = x.copy(); xminus[i] -= dx
                    H[i,j] = ( fct(xplus) - 2*f0 + fct(xminus) ) / (dx**2)
                else:
                    xp = x.copy(); xp[i] += dx; xp[j] += dx
                    xm = x.copy(); xm[i] -= dx; xm[j] -= dx
                    xipjm = x.copy(); xipjm[i] += dx; xipjm[j] -= dx
                    ximjp = x.copy(); ximjp[i] -= dx; ximjp[j] += dx
                    H[i,j] = ( fct(xp) - fct(xipjm) - fct(ximjp) + fct(xm) ) / (4*dx**2)
        
        return H + H.T - np.diag(H.diagonal())
    

    def line_search_wolfe(self, xk, pk, alpha0, c1, c2,
                          max_iter, expand_factor, strong=True):
        
        """ 
        Perform a line search along search direction pk starting from xk,
        using the Wolfe conditions (weak/strong).

        Theory:
        ----------
        - We want to find a step length alpha > 0 such that the new point
        x_{k+1} = xk + alpha* pk satisfies:
            (1) Armijo (sufficient decrease) condition:
                f(xk + alpha* pk) ≤ f(xk) + c1 *alpha* ∇f(xk)^T pk
            (2) Wolfe curvature condition:
                ∇f(xk + alpha* pk)^T pk ≥ c2 ∇f(xk)^T pk    (weak Wolfe)
                or |∇f(xk + alpha* pk)^T pk| ≤ -c2 ∇f(xk)^T pk   (strong Wolfe)

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
        pk : ndarray
            Search direction (should be descent).
        alpha0 : float
            Initial trial step length.
        c1 : float
            Armijo parameter (sufficient decrease, 0 < c1 < 1). Sigma in Fletcher.
        c2 : float
            Wolfe constant (curvature condition, 0 < c2 < 1). Rho in fletcher.
        max_iter : int
            Maximum iterations for bracketing phase. Safeguard.
        expand_factor : float
            Factor to expand trial alpha in bracketing phase. Tau in Fletcher. 
        strong : bool, default=True
            If True, enforce strong Wolfe condition; else weak Wolfe.

        Returns:
        ----------
        alpha : float
            Step length satisfying Wolfe conditions (or fallback if not found).
        
        """
        

        #φ(α) := f(x_k + α p_k). phi0 = φ(0) = f(x_k).
        #g0 = φ'(0) = ∇f(x_k)^T p_k. For a descent direction we need g0 < 0.
        #If g0 >= 0 then p_k is not a descent direction, no positive step will decrease f. Returning 0.0 means don’t move. Maybe handle this differently?
        f = self.obj_fct
        phi0 = f(xk)
        g0 = np.dot(self.compute_grad(xk), pk)
        if g0 >= 0:
            return 0.0 
        
        #shorthands
        def phi(alpha): return f(xk + alpha * pk)
        def phip(alpha): return np.dot(self.compute_grad(xk + alpha * pk), pk)


        #bracketing phase: expand α until we find an interval [a, b] known to contain a point that satisfies Armijo & Wolfe
        alpha_prev, phi_prev = 0.0, phi0
        alpha = alpha0
        for i in range(max_iter):
            phi_alpha = phi(alpha)

            #First check armijo NEGATED, meaning if true Armijo failed so minimizer is between current alpha and previous one. 
            #second check detects that the sampled φ(α) is not improving relative to the previous φ, so passed minimizer.
            #in either case, stop expanding and enter zoom (sectioning) on [alpha_prev, alpha]
            if (phi_alpha > phi0 + c1 * alpha * g0) or (i > 0 and phi_alpha >= phi_prev):
                return self._zoom(alpha_prev, alpha, phi0, g0,
                                  c1, c2, strong, phi, phip, max_iter)
            
            #If armijo didn't fail, now check wolfe
            grad_alpha = phip(alpha)
            if strong:
                if abs(grad_alpha) <= -c2 * g0:
                    return alpha 
            else:
                if grad_alpha >= c2 * g0:
                    return alpha
            
            #If grad_alpha >= 0 the derivative changed sign, means the minimizer lies between α_prev and α. 
            #So then call _zoom to search the bracket where the slope changed sign.
            if grad_alpha >= 0:
                return self._zoom(alpha, alpha_prev, phi0, g0,
                                  c1, c2, strong, phi, phip, max_iter)


            #If none of the stopping conditions hold, we expand the trial step α.
            alpha_prev, phi_prev = alpha, phi_alpha
            alpha *= expand_factor

        return alpha #fallback
    

    def _zoom(self, alo, ahi, phi0, g0, c1, c2,
              strong, phi, phip, max_iter):
        
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
            * A safeguard keeps alpha_j away from endpoints.
            2. Check Armijo condition:
                If f(xk+alpha_j pk) > f(0) + c1 *alpha_j* f'(0) OR
                    f(xk+alpha_j *pk) ≥ f(xk+alo pk),
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
            φ'(0) = ∇f(xk)^T pk.
        c1 : float
            Armijo parameter (sufficient decrease, 0 < c1 < 1). Sigma in Fletcher.
        c2 : float
            Wolfe constant (curvature condition, 0 < c2 < 1). Rho in Fletcher.
        strong : bool
            If True, enforce strong Wolfe condition.
        phi : callable
            Function φ(alpha) = f(xk + alpha* pk).
        phip : callable
            Derivative φ'(alpha) = ∇f(xk+alpha*pk)^T pk.
        max_iter : int
            Maximum iterations of zoom. Safeguard.

        Returns:
        ----------
        alpha : float
            Step length satisfying Wolfe conditions, or midpoint fallback.
        
        """
        

        a_lo, a_hi = min(alo, ahi), max(alo, ahi)

        for i in range(max_iter):
            #quadratic interpolation if a_lo == 0, else midpoint. Quadratic is a better guess for 
            #candidate alpha we are looking for than just midpoint of interval. We essentially pick
            #the minimizer of the quadratic as the candidate. Quadratic interpolation is q = phi0 + g0*alpha + c*alpha**2
            #where we determine c so that passes through (a_hi, phi_hi). The minimizer is then given by
            #q´= 0 => alpha = (g0*a_hi**2) / (2(phi_hi - phi0 - g0*a_hi)). 
            if a_lo == 0.0:
                phi_hi = phi(a_hi)
                denom = (phi_hi - phi0 - g0 * a_hi)
                if denom != 0:
                    alpha_j = - (g0) * a_hi * a_hi / (2.0 * denom)
                else:
                    alpha_j = 0.5 * (a_lo + a_hi)
            else:
                alpha_j = 0.5 * (a_lo + a_hi)

            #Safeguard, avoid being too close to endpoints. Keeps candidate strictly in bracket.
            safeg = 0.1 * (a_hi - a_lo)
            alpha_j = np.clip(alpha_j, a_lo + safeg, a_hi - safeg)

            #Mirrors the bracketing checks but now inside the bracket. 
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
                    
    
                if grad_j * (a_hi - a_lo) >= 0:  #ensure the bracket endpoints end up on opposite sides of the minimizer
                    a_hi = a_lo   #flip endpoints so the next assignment a_lo = alpha_j produces the proper ordering
                a_lo = alpha_j   #shifts left endpoint to α_j for the next iteration

            #If bracket smaller than ls_tol, return midpoint. 
            if abs(a_hi - a_lo) < self.ls_tol:
                return 0.5 * (a_lo + a_hi)
            
        #If max_iter exhausted also return midpoint. (should theoretically not happen but good safeguard incase for 
        #example bad start guess in line search gives infinte loops or something. Makes sure we always return something.)  
        return 0.5 * (a_lo + a_hi)




class ClassicalNewton(OptimizationMethod):

    
    def newton_step(self,xk: np.ndarray,ls):
        """
        Newton step with inexact wolfe line search if ls true
        """

        #Starting at x_k, this method calculates x_k+1
        gk = self.compute_grad(xk)
        Hk = self.hessian(xk)
        dk = np.linalg.solve(Hk,-gk)

        if ls: #line search (inexact, wolfe)
            alpha = self.line_search_wolfe(xk, dk, alpha0=1.0, c1=0.1, c2=0.01, max_iter=30, expand_factor=9.0)
            return xk + alpha * dk
        
        return xk + dk


    def newton_method(self,ls):

        steps = [self.x0.copy()]
        xk = self.x0.copy()
        xold = np.inf*np.ones(len(self.x0))

        while np.linalg.norm(xk - xold) >= self.tol:
            xold = xk.copy()
            xk = self.newton_step(xk,ls) 
            steps.append(xk)

        return np.vstack(steps)



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

#With len(steps) <50 i got "Final point: [-0.39704165 0.15243965] f(x)= 1.954431911317191", but with 500 or 5000 it is very
#close to converging to [1,1] ([0.9996827  0.99936444]  f(x)= 1.007927564178627e-07), so seems to work correctly. 
#Slower convergence is expected compared to Newton so not a problem i think.
while np.linalg.norm(xk - xold) >= tol and len(steps) < 5000:
    xold = xk.copy()
    pk = -method.compute_grad(xk)  #steepest descent direction
    alpha = method.line_search_wolfe(xk, pk,
                                     alpha0=1.0,
                                     c1=0.1,  #Fletchers Armijo param
                                     c2=0.01, #Fletchers curvature param
                                     max_iter= 30,
                                     expand_factor=9,
                                     strong=True)
    xk = xk + alpha * pk
    steps.append(xk.copy())

steps = np.vstack(steps)

#Solution is [1,1] so expect output something like "Final point: [0.999...  0.999...]  f(x)= very small"
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
plt.title("Steepest Descent with Wolfe Line Search (Rosenbrock)")
plt.show()


########### Test Wolfe vs exact Newton #########

#Newton without line search (exact step)
op = OptimizationProblem(rosenbrock, rosenbrock_grad)
x0 = np.array([-0.5, 3.0])
tol = 1e-6
newton_exact = ClassicalNewton(op, x0.copy(), tol)
steps_exact = newton_exact.newton_method(ls=False)

#Newton with Wolfe line search
newton_wolfe = ClassicalNewton(op, x0.copy(), tol)
steps_wolfe = newton_wolfe.newton_method(ls=True)

print("Exact Newton solution:", steps_exact[-1], " f=", rosenbrock(steps_exact[-1]))
print("Newton+Wolfe solution:", steps_wolfe[-1], " f=", rosenbrock(steps_wolfe[-1]))

#Plot contour + both trajectories
xgrid = np.linspace(-1.5, 2.0, 400)
ygrid = np.linspace(-1.0, 4.0, 400)
X, Y = np.meshgrid(xgrid, ygrid)
Z = rosenbrock([X, Y])

plt.figure(figsize=(7, 6))
plt.contour(X, Y, Z, levels=np.logspace(-1, 3.5, 20), cmap="jet")

#Exact Newton path
plt.plot(steps_exact[:,0], steps_exact[:,1], marker='o', color='blue', linewidth=1, label="Exact Newton")
plt.scatter(steps_exact[0,0], steps_exact[0,1], color='blue', s=60, marker='x')
plt.scatter(steps_exact[-1,0], steps_exact[-1,1], color='blue', s=80, marker='*')

#Wolfe Newton path
plt.plot(steps_wolfe[:,0], steps_wolfe[:,1], marker='o', color='red', linewidth=1, label="Newton + Wolfe")
plt.scatter(steps_wolfe[0,0], steps_wolfe[0,1], color='red', s=60, marker='x')
plt.scatter(steps_wolfe[-1,0], steps_wolfe[-1,1], color='red', s=80, marker='*')

plt.legend()
plt.title("Newtons Method (Exact vs Wolfe Line Search) on Rosenbrock")
plt.show()



#Function values vs iteration plot, shows convergence difference. 

#The plot confirms the theoretical convergence behavior of Newtons method. 
#Exact Newton shows quadratic convergence, once close to the minimum, the function values drop extremely fast. 
#Newton with Wolfe line search converges more cautiously. early iterations are slower because 
#the step size is reduced to guarantee global convergence. But once near the solution, both methods approach 
#the true minimum at (1,1).

#Quadratic convergence => Error at next step is proportional to the square of error at current step.
#Wolfe seems to have linear convergence at first but i quickly speeds up. 

fvals_exact = [rosenbrock(x) for x in steps_exact]
fvals_wolfe = [rosenbrock(x) for x in steps_wolfe]

plt.figure(figsize=(6, 5))
plt.semilogy(fvals_exact, marker='o', color='blue', label="Exact Newton")
plt.semilogy(fvals_wolfe, marker='o', color='red', label="Newton + Wolfe")

plt.xlabel("Iteration")
plt.ylabel("f(x)")
plt.title("Function Value vs Iteration (log scale)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()