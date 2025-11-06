import numpy as np

# Hessian at point x for function with known gradient grad
def Hessian_from_grad(x, grad):
    dim = len(x)
    H = np.zeros((dim,dim))
    dx = 1e-8 # -8 works because no dx**2

    # Lower triangular part
    for i in range(dim): # loop over rows
        hi = np.zeros(i+1) # h:th row in lower triangular part of Hessian
        xplus = x.copy()
        xminus = x.copy()
        xplus[i] = xplus[i] + dx
        xminus[i] = xminus[i] - dx
        for j in range(i+1):
            hi[j] = (grad(xplus)[j] - grad(xminus)[j]) / (2*dx) # is all of grad calculated each time?
        H[i,0:i+1] = hi

    # Fill out the rest
    for i in range(dim): # loop over rows
        for j in range(i+1,dim):
            H[i,j] = H[j,i]
    return H

# Hessian from function
def Hessian_from_fct(x, fct):
    dim = len(x)
    H = np.zeros((dim,dim))
    dx = 1e-5 # smaller dx gets close to machine precision
    f0 = fct(x) # used multiple times
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
    
    for i in range(dim): # loop over rows
        for j in range(i+1,dim):
            H[i,j] = H[j,i]
    return H

# Testing

# f = 0.5x_1^2 + 0.5x_2^2, gradient is (x_1, x_2)
def fct(x):
    return 0.5*(x[0]**2 + x[1]**2)
def grad(x):
    return [x[0], x[1]]

# f = x_1^2 * x_2^2, gradient is 2*(x_1*x_2^2,x_1^2*x_2)
def fct2(x):
    return x[0]**2 * x[1]**2
def grad2(x):
    return [2*x[0]*x[1]**2, 2*x[0]**2*x[1]]

x = [1, 1]
H = Hessian_from_fct(x, fct)
#print('H=',H)


# Matlab gradient function:
'''
    function g = grad(f,x)
    % g = grad(f,x)
    %
    % Calculates the gradient (column) vector of the function f at x.

    lx = length(x);
    g = zeros(lx,1);
    for i = 1:lx
        xplus = x;
        xminus = x;
        xplus(i) = x(i) + 1.e-8;
        xminus(i) = x(i) - 1.e-8;
        g(i,1) = ( f(xplus) - f(xminus) )/2.e-8;
    end
'''

"""
alpha = method.line_search_wolfe(xk, pk,
                                     alpha0=1.0,
                                     c1=0.1,  #Fletchers Armijo param
                                     c2=0.01, #Fletchers curvature param
                                     max_iter= 30,
                                     expand_factor=9,
                                     strong=True)
"""


"""
Anteckningar:
- positive definiteness!!!
- Exact line search
- Convergence
"""