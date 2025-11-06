import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix, csr_matrix

############ Parameters and Geometry Setup #########

#Parameters
h = 1.0 / 20  #mesh width
Lx, Ly = 3.0, 2.0  #Total domain bounding: 3x2

#Grid dimensions (including boundary nodes)
nx = int(Lx / h) + 1
ny = int(Ly / h) + 1

#Boundary Temperatures
T_WALL = 15.0
T_HEATER = 40.0
T_WINDOW = 5.0

#Geometry check function
def inside(i: int, j: int, h: float) -> bool:
    """
    Checks if a grid point (i,j) is strictly inside the apartment domain
    """
    x, y = i * h, j * h
    
    #Define three rooms
    is_omega_1 = (0.0 <= x <= 1.0) and (0.0 <= y <= 1.0) 
    is_omega_2 = (1.0 <= x <= 2.0) and (0.0 <= y <= 2.0)  
    is_omega_3 = (2.0 <= x <= 3.0) and (1.0 <= y <= 2.0)  

    if is_omega_1 or is_omega_2 or is_omega_3:
        return True
    
    return False




########## Set up and fill matrix + rhs vector #########

#Create mapping for all internal grid points (unknowns)
mapping = {}
counter = 0
#iterate over internal y- and x-indices
for j in range(1, ny - 1): 
    for i in range(1, nx - 1):
        #Check if the point is part of the apartment
        if inside(i, j, h):
            mapping[(i, j)] = counter
            counter += 1

N_unknowns = counter
print(f"Total number of unknowns (internal grid points): {N_unknowns}")

#Initialize Sparse System Matrix A and RHS vector b
A = lil_matrix((N_unknowns, N_unknowns))
b = np.zeros(N_unknowns)

#Build System
for (i, j), k in mapping.items():

    #diagonal values
    A[k, k] = -4.0
    
    #Check 4 neighbors
    for (ii, jj) in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
        
        #Check if neighbor is inside the overall bounding box [0, 3]x[0, 2]
        if 0 <= ii < nx and 0 <= jj < ny:
            
            x, y = ii * h, jj * h
            
            #Case 1: Neighbor is an interior (unknown) point
            if inside(ii, jj, h) and (ii, jj) in mapping:
                A[k, mapping[(ii, jj)]] = 1.0
                
            #Case 2 and 3: Neighbor is a fixed boundary or outside the domain (treat outside domain as just wall)
            else:
                T_BC = T_WALL
                
                #Check if the neighbor is part of boundary
                if inside(ii, jj, h): 
                    
                    #Room 1 Left wall
                    if (x == 0.0) and (0.0 <= y <= 1.0): T_BC = T_HEATER
                    #Room 2 Top wall
                    elif (y == Ly) and (1.0 <= x <= 2.0): T_BC = T_HEATER
                    #Room 3 Right wall
                    elif (x == Lx) and (1.0 <= y <= Ly): T_BC = T_HEATER 
                        
                    #Room 2 Bottom wall 
                    elif (y == 0.0) and (1.0 <= x <= 2.0): T_BC = T_WINDOW
                    
                    #Normal walls cover all others, so if not gone inside any if above, T_BC = T_WALL.
                    
                    #Add boundary contribution to the RHS vector
                    b[k] -= T_BC

                #Case 3: Neighbor is outside apartment domain
                else: 
                    T_BC = T_WALL
                    b[k] -= T_BC
            


########## Solve #########

#Convert A to CSR and use sciypy solve
A_csr = csr_matrix(A)
u = spsolve(A_csr, b)

#Initialize final grid for plotting (includes boundarie) and insert solution
U = np.full((ny, nx), np.nan)
for (i, j), k in mapping.items():
    U[j, i] = u[k]

#Fill boundaries and holes (outsid4 apartment) for plotting 
for j in range(ny):
    for i in range(nx):
        x, y = i * h, j * h
        
        #boundary nodes are inside but not in mapping
        is_boundary_node = inside(i, j, h) and (i, j) not in mapping
        
        if is_boundary_node:
            T_BC = T_WALL
            
            #Fill boundary values 
            if (x == 0.0) and (0.0 <= y <= 1.0): T_BC = T_HEATER
            elif (y == Ly) and (1.0 <= x <= 2.0): T_BC = T_HEATER
            elif (x == Lx) and (1.0 <= y <= Ly): T_BC = T_HEATER
            elif (y == 0.0) and (1.0 <= x <= 2.0): T_BC = T_WINDOW
            
            U[j, i] = T_BC

        #Fill outside of apartment with tag for plotting purposes
        is_in_hole_region = (0.0 <= x <= 2.0) and (1.0 < y <= 2.0)
        if is_in_hole_region and not inside(i, j, h):
            U[j, i] = -10.0 #random tag, could be any value as long as its can not be mistaken for value in apart (5-40 degrees)



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

print(np.shape(U))
