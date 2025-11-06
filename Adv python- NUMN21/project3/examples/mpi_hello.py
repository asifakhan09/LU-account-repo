from mpi4py import MPI

""" Get a communicator :
The most common communicator is the
one that connects all available processes
which is called COMM_WORLD
"""
comm = MPI.Comm.Clone(MPI.COMM_WORLD)
print(
    " Hello World : process ",
    comm.Get_rank(),
    " out of",
    comm.Get_size(),
    " is reporting for duty ! ",
)
