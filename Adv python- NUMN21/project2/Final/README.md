
## Project 2 - Optimization
The aim of the project was to implement different optimization methods and line search algorithms, 
test these on given optimization problems (functions), and compare our results with for example SciPys.


## Contents 

# benchmark.py
Trying all methods with different parameters on different test functions. 

# dummy.py
Contains baseline optimization methods for comparisons (random and SciPy).

# example_funs.py
Contains more example functions for testing method (other than Rosenbrock).

# newton.py
Contains classical newton method class with methods for hessian, stepping and solving (Task 3)

# optimization.py
Contains general optimization method and optimization problem classes (Task 1,2). Methods for computing gradients and for different line search methods (wolfe, golden section, exact). (Task 4,6). 

# quasi_newton.py
Contains base class for quasi newton methods: Good Broyden, Bad Broyden, Symmetric Broyden, DFP (Davidon–Fletcher–Powell), 
and BFGS.

# linesearch_comp.py
Testing inexact line search separatley from newtons method (Task 7) and applying different (and no) line searches on Newton
for tests and comparisons (Task 8).

# chebyquad_notebook.ipynb
Comparing minimization of Chebyquad function with our methods and SciPy (Task 11). 

# bfgs_hessian_appr.py
Studies the approximation of the hessian inverse of the BFGS method.  


## CONTRIBUTIONS 
The first tasks (1-5) were done at least to some degree by all, but other than that below is a rough divide of the work.

Asifa - Task 9,11 , Chebyquad_notebook, quasi_newton
Marcus - Task 9, benchmark, test, quasi_newton, optimization
Joel - Task 9,12 , quasi_newton, newton, optimization, bfgs_hessian_appr
Jonathan - Task 6-8. linesearch_comp.py, optimization.py (ls_wolfe, _zoom)

