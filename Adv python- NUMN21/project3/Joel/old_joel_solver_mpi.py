# Run with:
# mpirun -np 3 python3 Joel/joel_solver_mpi.py

import sys 
sys.path.append(".")

import numpy as np
from matplotlib import pyplot as plt
from marcus.room_class import Room, BoxBoundary, Dirichlet, Neumann
from mpi4py import MPI

comm = MPI.Comm.Clone(MPI.COMM_WORLD)
rank = comm.Get_rank()
size = comm.Get_size()
assert size == 3

resolution = 40 # 1/dx, dx is mesh width
dx = 1/resolution
w = 0.8 # For relaxation
n_iter = 10 # Number of iterations

# Creating room and b.c. objects for each domain
if rank == 0: # omega 2, root

    omega = Room(2,1,resolution,ghost_mode=True)

    # Initial values for artificial Dirichlet conditions
    gamma1_right = 15*np.ones(resolution)
    gamma2_left = 15*np.ones(resolution)

    # Boundary conditions
    bc = BoxBoundary(
        T = Dirichlet(40*np.ones(resolution)),   
        B = Dirichlet(5*np.ones(resolution)),
        L = Dirichlet(np.concatenate((15*np.ones(resolution),gamma1_right))),
        R = Dirichlet(np.concatenate((gamma2_left,15*np.ones(resolution))))
    )

    buffer = np.zeros((resolution,resolution))

elif rank == 1: # omega 1

    omega = Room(1,1,resolution,ghost_mode=True)
    bc = BoxBoundary(
        T = Dirichlet(15*np.ones(resolution)), 
        B = Dirichlet(15*np.ones(resolution)),
        L = Dirichlet(40*np.ones(resolution)),        
        R = Neumann(0) # No calculations with this value
    )

    gamma1_left = np.zeros(resolution)
    buffer = np.zeros((2*resolution,2*resolution))

elif rank == 2: # omega 3

    omega = Room(1,1,resolution,ghost_mode=True)
    bc = BoxBoundary(
        T = Dirichlet(15*np.ones(resolution)), 
        B = Dirichlet(15*np.ones(resolution)),
        L = Neumann(0), # No calculations with this value
        R = Dirichlet(40*np.ones(resolution))
    ) 

    gamma2_right = np.zeros(resolution)
    buffer = np.zeros((2*resolution,2*resolution))

def update_bc(new_sol, source = 0):
    # Maybe unnecessary to do this in separate function, but iteration loop becomes cleaner
    if rank == 0:
        if source == 1:
            gamma1_right = new_sol[:,-1]
            bc.L = Dirichlet(np.concatenate((15*np.ones(resolution),gamma1_right)))
        elif source == 2:
            gamma2_left = new_sol[:,0]
            bc.R = Dirichlet(np.concatenate((gamma2_left,15*np.ones(resolution))))
    elif rank == 1:
        gamma1_left = (new_sol[resolution:2*resolution,1] - new_sol[resolution:2*resolution,0]) / dx
        bc.R = Neumann(gamma1_left)
    elif rank == 2:
        gamma2_right = (new_sol[0:resolution,resolution-2] - new_sol[0:resolution,resolution-1]) / dx
        bc.L = Neumann(gamma2_right)


for k in range(n_iter):

    if k > 0: # for relaxation
        uold = u.copy()

    if rank == 0:
        if k > 0:
            for i in range(1,size): # omega 2 borders all other rooms
                comm.Recv(buffer, source = i)
                update_bc(buffer.copy(), source = i)
        u = omega.solve(bc)
        for i in range(1,size):
            comm.Send(u.copy(), dest = i)

    elif rank in [1,2]:
        comm.Recv(buffer, source = 0)
        update_bc(buffer.copy())
        u = omega.solve(bc)
        comm.Send(u.copy(), dest = 0)

    if k > 0: # relaxation
        u = w*u + (1-w)*uold


# Compiling and plotting results on root
# The send of the final results from the workers that matches the following receive comes
# from the last iteration of the for loop
if rank == 0:

    #Boundary Temperatures
    T_WALL = 15.0
    T_HEATER = 40.0
    T_WINDOW = 5.0

    # U: ny x nx
    Lx, Ly = 3.0, 2.0
    nx = int(Lx * resolution) 
    ny = int(Ly * resolution) 
    U = np.full((ny, nx), -10)
    U[:,resolution:2*resolution] = np.flip(u,0) # omega 2
    for i in range(1,size):
        comm.Recv(buffer, source = i)
        if i == 1: # omega 1
            U[0:resolution,0:resolution] = np.flip(buffer,0)
        elif i == 2: # omega 3
            U[resolution:2*resolution,2*resolution:3*resolution] = np.flip(buffer,0)

    ###### Plotting #########

    plt.figure(figsize=(9, 6))
    #Create a mask for outside, "hides" all values in U that are -10.0 
    masked_array = np.ma.array(U, mask=(U == -10.0)) 

    plt.imshow(masked_array, 
            origin="lower", 
            extent=[0, Lx, 0, Ly], 
            cmap="hot", 
            vmin=T_WINDOW, 
            vmax=T_HEATER)
    plt.colorbar(label = r"Temperature $(\mathbf{^{\circ}C})$")

    #Add black lines for room boundaries
    plt.axvline(x=1.0, ymin=0.0, ymax=1.0, color='black', linestyle='--')
    plt.axvline(x=2.0, ymin=0.0, ymax=1.0, color='black', linestyle='--')
    plt.axhline(y=1.0, xmin=2.0/Lx, xmax=3.0/Lx, color='black', linestyle='--') 
    plt.axhline(y=1.0, xmin=0.0, xmax=1.0/Lx, color='black', linestyle='--') 

    plt.title("Temperature Distribution in 3-Room Apartment")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

   




r"""

    - project 2

    - Robert computer skills course

"""
