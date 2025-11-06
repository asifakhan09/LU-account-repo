from mpi4py import MPI
import numpy as np


comm = MPI.Comm.Clone(MPI.COMM_WORLD)
rank = comm.Get_rank()


data = np.array([range(1, 10)])

""" Make sure that every send has a matching recv! """

if rank == 0:
    # send 10 numbers to rank 1 (dest=1)
    # method 'send' for Python objects (pickle under the hood):
    comm.send(data, dest=1)

if np.linalg.norm(data) > 1:
    # receive 10 numbers from rank 0 (source=0)
    # method 'recv' for Python objects (pickle under the hood):
    data = comm.recv(source=0)
    print("P[", rank, "] received data =", data)
