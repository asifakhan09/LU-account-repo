import sys 
sys.path.append(".")

import numpy as np
from matplotlib import pyplot as plt
from marcus.room_class import Room, BoxBoundary, Dirichlet, Neumann

resolution = 50 # 1/dx, dx is mesh width. Should be 20.
dx = 1/resolution
w = 0.8 # For relaxation
n_iter = 10 # Number of iterations

omega13 = Room(1,1,resolution,ghost_mode=True) # Room object for omega 1 and 3 

# Boundary conditions for omega 1
bc1 = BoxBoundary(
    T = Dirichlet(15*np.ones(resolution)), 
    B = Dirichlet(15*np.ones(resolution)),
    L = Dirichlet(40*np.ones(resolution)),        
    R = Neumann(0) # No calculations with this value
)

# Boundary conditions for omega 3
bc3 = BoxBoundary(
    T = Dirichlet(15*np.ones(resolution)), 
    B = Dirichlet(15*np.ones(resolution)),
    L = Neumann(0), # No calculations with this value
    R = Dirichlet(40*np.ones(resolution))
) 

# Initializing arrays for artificial Neumann conditions (is this necessary?)
gamma1_left = np.zeros(resolution)
gamma2_right = np.zeros(resolution)

omega2 = Room(2,1,resolution,ghost_mode=True) # Room object for omega 2

# Initial values for artificial Dirichlet conditions
gamma1_right = 15*np.ones(resolution)
gamma2_left = 15*np.ones(resolution)

# Boundary conditions for omega 2
bc2 = BoxBoundary(
    T = Dirichlet(40*np.ones(resolution)),   
    B = Dirichlet(5*np.ones(resolution)),
    L = Dirichlet(np.concatenate((15*np.ones(resolution),gamma1_right))),
    R = Dirichlet(np.concatenate((gamma2_left,15*np.ones(resolution))))
)


for k in range(n_iter):

    # Solving problem in omega2
    if k > 0:
        u2_old = u2.copy() # Saving old values for relaxation
    u2 = omega2.solve(bc2)

    # Neumann values for omega 1 and 3. Convention: Derivatives are pointing OUTWARDS (i.e. FROM the domain)
    gamma1_left = (u2[resolution:2*resolution,1] - u2[resolution:2*resolution,0]) / dx
    gamma2_right = (u2[0:resolution,resolution-2] - u2[0:resolution,resolution-1]) / dx

    # Update B.C. for omega 1 
    bc1.R = Neumann(gamma1_left)
    
    # Update B.C. for omega 3
    bc3.L = Neumann(gamma2_right)

    # Solving problem in omega 1 and 3
    if k > 0:
        u1_old = u1.copy()
        u3_old = u3.copy()
    u1 = omega13.solve(bc1)
    u3 = omega13.solve(bc3)

    # Updating bc for omega 2
    gamma1_right = u1[:,-1]
    gamma2_left = u3[:,0]

    bc2.L = Dirichlet(np.concatenate((15*np.ones(resolution),gamma1_right)))
    bc2.R = Dirichlet(np.concatenate((gamma2_left,15*np.ones(resolution))))

    
    # Unsure if relaxation is at the right place in the iteration
    if k > 0:
        u1 = w*u1 + (1-w)*u1_old
        u2 = w*u2 + (1-w)*u2_old
        u3 = w*u3 + (1-w)*u3_old
    

# Jonathan's plotting code

#Boundary Temperatures
T_WALL = 15.0
T_HEATER = 40.0
T_WINDOW = 5.0

# U: ny x nx
Lx, Ly = 3.0, 2.0
nx = int(Lx * resolution) 
ny = int(Ly * resolution) 
U = np.full((ny, nx), -10)
U[0:resolution,0:resolution] = np.flip(u1,0)
U[:,resolution:2*resolution] = np.flip(u2,0)
U[resolution:2*resolution,2*resolution:3*resolution] = np.flip(u3,0)

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
plt.colorbar(label="Temperature ($\mathbf{^{\circ}C}$)")

#Add black lines for room boundaries
plt.axvline(x=1.0, ymin=0.0, ymax=1.0, color='black', linestyle='--')
plt.axvline(x=2.0, ymin=0.0, ymax=1.0, color='black', linestyle='--')
plt.axhline(y=1.0, xmin=2.0/Lx, xmax=3.0/Lx, color='black', linestyle='--') 
plt.axhline(y=1.0, xmin=0.0, xmax=1.0/Lx, color='black', linestyle='--') 

plt.title("Temperature Distribution in 3-Room Apartment")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

print("u2 shape:",np.shape(u2))
print("res",resolution)