# project venv in "labs"
. ../.venv/bin/activate

# run program:
# note: aliases dont work?
mpirun -n 4 python3 examples/mpi_hello.py