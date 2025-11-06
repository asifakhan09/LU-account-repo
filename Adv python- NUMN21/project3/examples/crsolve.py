from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve,cg
import numpy as np
# row indices, column indices and values
rowptr = np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3])
colind = np.array([0, 1, 0, 1, 2, 1, 2, 3, 2, 3])
values = np.array([-2.,1.,1.,-2.,1.,1.,-2.,1.,1.,-2.])
# create sparse from COO format
A = csr_matrix( (values, (rowptr, colind)) )

# setup right hand side
b = np.array([1.,2.,2.,1.])

# solve Ax = b
x = spsolve( A, b )

# solve Ax = b, returns tuple with solution and iteration count
x = cg( A, b )[0]

# lets check the result and print || Ax - b ||
print("Solution is correct? ", np.allclose( A@x, b))
