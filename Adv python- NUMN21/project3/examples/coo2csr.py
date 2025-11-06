from scipy.sparse import csr_matrix
import numpy as np

# row indices
rowind = np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3])

# column indices
colind = np.array([0, 1, 0, 1, 2, 1, 2, 3, 2, 3])

# matrix values
values = np.array([-2.,1.,1.,-2.,1.,1.,-2.,1.,1.,-2.])

# create sparse from COO format
# all arrays have the same length
A = csr_matrix( (values, (rowind, colind) ) )

# let's see what we got
print (A)
