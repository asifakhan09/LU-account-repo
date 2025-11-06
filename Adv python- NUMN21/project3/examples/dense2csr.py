from scipy.sparse import csr_matrix
import numpy as np

# create a dense matrix
D = np.array([[ -2.,  1.,  0.,  0.],
              [  1., -2.,  1.,  0.],
              [  0.,  1., -2.,  1.],
              [  0.,  0.,  1., -2.]])

# create sparse matrix from dense matrix
A = csr_matrix( D )

# let's see what we got
print ( A )
