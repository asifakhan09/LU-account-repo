from mpi4py import MPI

comm = MPI.Comm.Clone(MPI.COMM_WORLD)

rank = comm.Get_rank()
size = comm.Get_size()

left = rank - 1 if rank > 0 else size - 1
right = rank + 1 if rank < size - 1 else 0

s = rank

if rank == 0:
    comm.send(s, dest=right)
    s += comm.recv(source=left)
else:
    s += comm.recv(source=left)
    comm.send(s, dest=right)

# rank 0 now has the correct result
# send that to all others using the ring structure
if rank == 0:
    comm.send(s, dest=right)
    s = comm.recv(source=left)
else:
    s = comm.recv(source=left)
    comm.send(s, dest=right)

print(f"P[{rank}] s = {s}")

s = comm.allreduce(rank, op=MPI.SUM)

if rank == 0:
    print(f"Correct result is {s}")
