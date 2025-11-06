from mpi4py import MPI
import numpy as np
""" Get a communicator:
    The most common communicator is the
    one that connects all available processes
    which is called COMM_WORLD.
    Clone the communicator to avoid interference
    with other libraries or applications
"""
comm = MPI.Comm.Clone( MPI.COMM_WORLD )

rank = comm.Get_rank()
if rank == 0:
    # send 10 numbers to rank 1 (dest=1)
    data = np.arange(10, dtype='d')
    #comm.Send([data,MPI.DOUBLE], dest=1, tag=42 )
    comm.Send([data,MPI.DOUBLE], dest=1, tag=42 )
if rank == 1:
    # receive 10 numbers from rank 0 (source=0)
    data = np.empty(10, dtype='d')
    comm.Recv(data, source=0, tag=42 )
    print("P[",rank,"] received data =",data)
