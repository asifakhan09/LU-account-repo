# Run with:
# mpirun -np 3 python3 Joel/trying_mpi.py

import numpy as np
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE



"""
Test problem:

- Each of the three processors is assigned a 2x2 matrix Ai and a 2x1 array bi
- First eq. system in processor 0 is solved -> u0
- u0[0] -> b1[0] and u0[1] -> b2[1]
- Problem is solved in processor 1 and 2 -> u1 and u2
- u1[0] -> b0[0] and u2[1] -> b0[1]

"""

comm = MPI.Comm.Clone(MPI.COMM_WORLD)
rank = comm.Get_rank()
size = comm.Get_size()
assert size == 3 

if rank == 0:
    A = np.identity(2)
    b = np.array([1,0], dtype=float)
    buffer = np.zeros(1)
elif rank == 1:
    A = 2*np.identity(2)
    b = np.array([0,1], dtype=float)
    buffer = np.zeros(2)
else: 
    A = 3*np.identity(2)
    b = np.array([1,1], dtype=float)
    buffer = np.zeros(2)

def update_bc(new_val, source = 0):
    if rank == 0: 
        #print(f"new_val = {new_val} inside update_bc for rank {rank}, source={source}")
        if source == 1:
            b[0] = new_val[0] 
        elif source == 2:
            b[1] = new_val[0]
        else:
            raise Exception(f"Wrong source rank when updating b for root - received from {source}")
    elif rank == 1:
        b[0] = new_val[0]
    else:
        b[1] = new_val[1]

for k in range(2):

    if rank == 0:
        if k > 0:
            for i in range(1,size):
                comm.Recv(buffer, source = i)
                #print(f"rank {rank} received array {buffer} from source {i}")
                update_bc(buffer.copy(), source = i)
        #print(f"k={k}, rank={rank}, b={b}")
        u = np.linalg.solve(A,b)
        for i in range(1,size):
            comm.Send(u.copy(), dest = i)

    elif rank in [1,2]:
        comm.Recv(buffer, source = 0)
        update_bc(buffer.copy())
        u = np.linalg.solve(A,b)
        msg = np.array([u[rank-1]])
        #print(f"rank {rank} sent array {msg}")
        comm.Send(msg, dest = 0)

    print(f"k = {k}, u_{rank} = {u}")   

# ----------------------------------------------
"""

    if rank == 0:
        if k > 0:
            print(f"b before: {b}")
            for i in range(1,3): # generalize
                comm.Recv(buffer, source = i)
                print(f"Received buffer_{i} = {buffer}")
                update_bc(buffer.copy(), source = i)
            print(f"b after: {b}")
            # lägg send fårn u2, u3 i samma if-sats?

        u = np.linalg.solve(A,b)
        #x = u.copy() # x in general not same as u
        #buffer = np.ndarray(x.copy())
        #buffer = x.copy()
        for i in range(1,size):
            comm.Send(u.copy(), dest = i)

    elif rank in [1,2]:
        comm.Recv(buffer, source = 0)
        #x = buffer.copy()
        #update_bc(x)
        update_bc(buffer)
        u = np.linalg.solve(A,b)
        #x[0] = u[rank - 1]
        #buffer = x.copy()
        #print(f"Sent buffer_{rank} = {buffer}")
        comm.Send(u[rank-1], dest = 0)

    print(f"k = {k}, u_{rank} = {u}")   
"""

# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

"""
# Test of update_bc:

if rank == 0:
    update_bc([10], source = 1)
    update_bc([20], source = 2)
    print(f"rank = {rank}, should be [10,20]: {b}")
elif rank == 1:
    update_bc([10,20])
    print(f"rank = {rank}, should be [10,1]: {b}")
else:
    update_bc([10,20])
    print(f"rank = {rank}, should be [1,20]: {b}")

"""

"""
class MPIExample:

    def __init__(self):
        self.rank = rank
        self.x = np.zeros(2)
        self.buffer = np.zeros(2)

        if rank == 0:
            self.A = np.identity(2)
            self.b = np.array([1,0])
        elif rank == 1:
            self.A = 2*np.identity(2)
            self.b = np.array([0,1])
        else: 
            self.A = 3*np.identity(2)
            self.b = np.array([1,1])

    def update_bc(self, new_val, source = 0):
        if self.rank == 0:
            if source == 1:
                self.b[0] = new_val[0]
            elif source == 2:
                self.b[1] = new_val[0]
            else:
                raise Exception("Wrong source rank when updating b for root")
        elif self.rank == 1:
            self.b[0] = new_val[0]
        else:
            self.b[1] = new_val[1]

    def run(self):

        for k in range(2):
            if self.rank == 0:
                if k > 0:
                    for i in range(1,3): # generalize
                        comm.Recv(self.buffer, source = i)
                        #print(f"Received buffer_{i} = {buffer}")
                        update_bc(self.buffer, source = i)
                    #print("b=",b)
                    # lägg send fårn u2, u3 i samma if-sats?

                u = np.linalg.solve(A,b)
                x = u.copy() # x in general not same as u
                #buffer = np.ndarray(x.copy())
                buffer = x.copy()
                for i in range(1,size):
                    comm.Send(buffer, dest = i)

            elif rank in [1,2]:
                comm.Recv(buffer, source = 0)
                #x = buffer.copy()
                #update_bc(x)
                update_bc(buffer)
                u = np.linalg.solve(A,b)
                x[0] = u[rank - 1]
                buffer = x.copy()
                #print(f"Sent buffer_{rank} = {buffer}")
                comm.Send(buffer, dest = 0)

            #print(f"k = {k}, u_{rank} = {u}")   




if __name__ == "__main__":

    comm = MPI.Comm.Clone(MPI.COMM_WORLD)
    rank = comm.Get_rank()
    size = comm.Get_size()
    assert size == 3 


"""