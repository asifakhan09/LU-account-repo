import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

H = 1.0 / 20 
T_WALL = 15.0
T_HEATER = 40.0
T_WINDOW = 5.0

######## Geometry and boundary definitions for each subdomain #######

def get_subdomain_params(subdomain_id: int):
    """
    Defines the geometry, grid size (nx_s, ny_s), and coordinate offset 
    (i_offset, j_offset) for the three rooms based on the global grid (0-indexed).

    Returns: (Lx_s, Ly_s, i_off, j_off, name) 
        Lx_s: width of subdomain (ex for room 1 is 1.0)
        Ly_s: height of subdomain (ex for room 2 is 2.0)
        i_off: X-offset (for global index). The starting global column index (i) of the room.
               This is the coordinate of the room's left boundary. Example for room2: 
               Starts at x=1, so the index offset is 1.0/H=20. When solver processes local
               point (i_s,j_s) in O2, its global position is (i_s+20, j_s).
        j_off: Y-Offset (for global index): The starting global row index (j) of the room.
               This is the coordinate of the room's bottom boundary.
        name: Subdomain domain name, string identifier 
    """

    if subdomain_id == 1:
        # Omega_1: [0, 1] x [0, 1]
        return 1.0, 1.0, 0, 0, "O1"
    elif subdomain_id == 2:
        # Omega_2: [1, 2] x [0, 2]
        return 1.0, 2.0, int(1.0 / H), 0, "O2"
    elif subdomain_id == 3:
        # Omega_3: [2, 3] x [1, 2]
        return 1.0, 1.0, int(2.0 / H), int(1.0 / H), "O3"
    else:
        raise ValueError("Invalid subdomain_id. Must be 1, 2, or 3.")
    


def get_fixed_bc(x: float, y: float, name: str) -> float:
    """
    Returns the fixed Dirichlet temperature (40, 5, or 15) for external boundary point.

    params: 
        x (floar): x-coordinate of point
        y (float): y-coordinate of point
        name (str): subdomain identifier ("O1", "O2" or "O3")

    returns:
    The boundaries fixed temps, i.e either 40.0, 5.0 or 15.0
    """
    Lx, Ly = 3.0, 2.0 # Global bounds for BC check

    # T_HEATER (40 deg C) locations (Gamma_H):
    if name == "O1" and (x == 0.0) and (0.0 <= y <= 1.0): return T_HEATER # O1 left wall
    if name == "O2" and (y == Ly) and (1.0 <= x <= 2.0): return T_HEATER # O2 top wall
    if name == "O3" and (x == Lx) and (1.0 <= y <= Ly): return T_HEATER # O3 right wall
    
    # T_WINDOW (5 deg C) location (Gamma_WF):
    if name == "O2" and (y == 0.0) and (1.0 <= x <= 2.0): return T_WINDOW # O2 bottom wall
    
    # The default fixed BC is T_WALL, ensuring all unhandled external boundaries get 15.
    return T_WALL 



######### Matrix Construction and Solution #########

def solve_subdomain(subdomain_id: int, boundary_values: dict, interface_flux: dict = None):
    """
    Constructs and solves system for a single subdomain.

    params:
        subdomain_id (int): 1, 2, or 3.
        boundary_values (dict): Dictionary containing Dirichlet values for the interfaces 
                                (used by O2). Keys: (global_i, global_j).
        interface_flux (dict, optional): Required for Neumann rooms (O1 and O3).
                                         Keys: (global_i, global_j), Value: flux g=du/dn.

    Returns:
        np array: The solved temperature vector u_s for the subdomain.
        dict: Mapping from linear index to (i_s, j_s) tuple for the unknowns.
        tuple: (nx_s, ny_s) grid dimensions.
    """
    Lx_s, Ly_s, i_off, j_off, name = get_subdomain_params(subdomain_id)
    
    # Subdomain grid dimensions (including boundaries)
    nx_s = int(Lx_s / H) + 1
    ny_s = int(Ly_s / H) + 1
    
    
    ### Create mapping for interior points and Neumann interface points. 
    # Iterate over the grid points within the subdomain and create the mapping dictionary: (local_i, local_j) -> linear_index.
    # In O2 only the interior points are unknowns. The interface points are known Dirichlet values.
    # In O1 and O3 both the interior points and the interface points are unknowns and are included in mapping.
    mapping = {}
    counter = 0
    # Loop over y-indices internal to the subdomain grid
    for j_s in range(1, ny_s - 1): 
        
        # O2 (Dirichlet): Interior points only (1 to nx_s - 2)
        if name == "O2":
            i_start, i_end = 1, nx_s - 1
        # O1 (Neumann): Interior and right interface point (1 to nx_s - 1)
        elif name == "O1":
            i_start, i_end = 1, nx_s
        # O3 (Neumann): Left interface point and interior (0 to nx_s - 2)
        elif name == "O3":
            i_start, i_end = 0, nx_s - 1
        
        for i_s in range(i_start, i_end):
            mapping[(i_s, j_s)] = counter
            counter += 1
    
    #initalize sparse matrix A and rhs vector b for subdomain given number of unknowns
    N_unknowns = counter
    A = lil_matrix((N_unknowns, N_unknowns))
    b = np.zeros(N_unknowns)



    ### Build system. 
    # Iterate over every unknown point (i_s,j_s) in mapping and determine its row in A and values in b.
    for (i_s, j_s), k in mapping.items():
        
        # Global grid indices
        i_g, j_g = i_s + i_off, j_s + j_off
        x_g, y_g = i_g * H, j_g * H
        
        # Check if this unknown is a Neumann Interface point
        is_neumann_interface = False

        if name == "O1" and i_s == nx_s - 1: # O1 Right boundary
            # The interface for O1 is only active up to y=1
            ##if j_g < ny_off_1: 
            if 0 < j_g < 1.0:
                is_neumann_interface = True
                interior_neighbor_i_s = i_s - 1 

                # Normal n points pos x. g = du/dx = u_(NC).
                # Slide equation RHS is -h u_(NC). So the sign is -1. See Project3_FD slides 26-27
                normal_sign = -1.0 

        elif name == "O3" and i_s == 0: # O3 Left boundary
            ##if j_g >= ny_off_1: # Only active from y=1 up to y=2
            if 1.0 < y_g < 2.0: 
                is_neumann_interface = True
                interior_neighbor_i_s = i_s + 1 

                # Normal n points neg x. g = -du/dx, so u_(NC) = du/dx = -g.
                # Slide equation RHS is -h u_(NC) = -h(-g) = hg. Thus, the sign is 1.
                normal_sign = 1.0

        
        if is_neumann_interface:
            
            # v_(i,j+1) + v_(i,j-1) + v_(i-1,j) - 3v_(i,j) = -h u_(NC)
            # Where (u_(NC) is calculated from flux g based on the normal vector. 
            # See slides 26-27 in Project3_FD
            
            # Diagonal coefficient (-4)
            ##A[k, k] = -4.0
            A[k, k] = -3.0
            
            # Interior neighbor coefficient (+1)
            ##A[k, mapping[(interior_neighbor_i_s, j_s)]] = 2.0
            A[k, mapping[(interior_neighbor_i_s, j_s)]] = 1.0
            
            # RHS contribution from flux (+/- hg)
            if interface_flux is None or (i_g, j_g) not in interface_flux:
                 g = 0.0
            else:
                 g = interface_flux[(i_g, j_g)]
            
            # RHS: +/- hg
            ##b[k] = normal_sign * 2.0 * H * g
            b[k] = normal_sign * H * g
            
            
            # Vertical neighbors (+1), must handle fixed boundaries (corners)
            for (di, dj) in [(0, 1), (0, -1)]:
                i_neighbor_s, j_neighbor_s = i_s + di, j_s + dj
                i_neighbor_g, j_neighbor_g = i_g + di, j_g + dj
                
                neighbor_coord_s = (i_neighbor_s, j_neighbor_s)
                
                # Case A: Neighbor is an unknown (interior point or other interface point)
                if neighbor_coord_s in mapping:
                    A[k, mapping[neighbor_coord_s]] = 1.0
                    
                # Case B: Neighbor is on a fxed external boundary (corner node on the wall) 
                else:
                    x_g, y_g = i_neighbor_g * H, j_neighbor_g * H
                    T_BC = get_fixed_bc(x_g, y_g, name)
                    # Move to RHS
                    b[k] -= T_BC
            


        else:
            # All interior points for all rooms as well as all unknowns for O2 goes here
            # Set diagonals to -4, form central difference. 
            #If a neighbor is another unknown (in the mapping), the corresponding off-diagonal entry in A is set to 1.0.
            # If a neighbor is a known value (fixed wall or dirichlet interface), value is moved to rhs (b).

            A[k, k] = -4.0
            
            # Check 4 neighbors
            for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                i_neighbor_s, j_neighbor_s = i_s + di, j_s + dj
                i_neighbor_g, j_neighbor_g = i_g + di, j_g + dj
                
                # Check if neighbor is inside the subdomain's full grid (0 <= index < N)
                is_on_subdomain_grid = (0 <= i_neighbor_s < nx_s) and (0 <= j_neighbor_s < ny_s)
    
                if is_on_subdomain_grid:
                    
                    neighbor_coord_s = (i_neighbor_s, j_neighbor_s)
                    
                    # Case 1: Neighbor is an interior unknown point (includes Neumann interfaces)
                    if neighbor_coord_s in mapping:
                        A[k, mapping[neighbor_coord_s]] = 1.0
                        
                    # Case 2: Neighbor is on a Fixed External Boundary or a Dirichlet Interface (O2)
                    else:
                        x_g, y_g = i_neighbor_g * H, j_neighbor_g * H
                        neighbor_coord_g = (i_neighbor_g, j_neighbor_g)
                        
                        # Case 2.1: Check if Drichlet interface (O2 only)
                        if name == "O2" and neighbor_coord_g in boundary_values:
                            T_interface = boundary_values[neighbor_coord_g]
                            b[k] -= T_interface
    
                        # Case 2.2: check if fixed external boundary (Wall, Heater, Window)
                        else:
                            T_BC = get_fixed_bc(x_g, y_g, name)
                            b[k] -= T_BC

    ### Solve system
    A_csr = csr_matrix(A)
    u_s = spsolve(A_csr, b)
    
    return u_s, mapping, (nx_s, ny_s)



def calculate_dirichlet_flux(u2_s: np.array, map2: dict, nx2: int, ny2: int, 
                              interface_values: dict) -> dict:
    """
    Calculates the heat flux (Neumann BC) g = du/dn at the interfaces Gamma1 and Gamma2
    from the Dirichlet solution u_2 on O2.
    
    The flux is calculated using finite difference approx between the interface temperature 
    u_Gamma and the adjacent interior temperature u_int. The sign of the flux is determined 
    by the outward normal vector of the  receiving subdomain (O1 or O3).

    params:
        u2_s (np.array): The solved temperature for O2.
        map2 (dict): The mapping of O2äs local grid points to linear indices.
        nx2,ny2 (int): Number of x- and y-points in O2.
        interface_values (dict): The Dirichlet boundary values u_Gamma used on O2
                                 
    Returns:
        dict: A dictionary of flux values. 
              Keys are the global grid coordinates of the interface points (global_i, global_j), 
              and values are the calculated flux g = du/dn.
    """
    
    flux_values = {}
    _, _, _, j_off, _ = get_subdomain_params(2) # O2 params
    
    nx_off_1 = int(1.0 / H)
    nx_off_2 = int(2.0 / H)
    ny_off_1 = int(1.0 / H)

    ### Gamma_1 (Left interface of O2) 
    i_g1 = nx_off_1 # Global x index for x=1
    for j_s in range(1, ny2 - 1): # j_s in O2 grid (y in (0, 2))
        j_g = j_s + j_off
        if j_g >= ny_off_1: continue # O1 is only up to y=1
        
        # Interface point (x=1, y=j_g*H)
        u_interface = interface_values.get((i_g1, j_g))
        if u_interface is None: continue

        # Interior layer is at x=1+H (i_s = 1)
        i_s_interior = 1
        k_interior = map2.get((i_s_interior, j_s))
        
        if k_interior is not None:
            u_interior = u2_s[k_interior]
            
            # Flux for O1 (Normal direction points Left, du/dx is negative)
            # g_O1 = (u_int - u_Gamma) / (-H)
            g = (u_interior - u_interface) / H 
            flux_values[(i_g1, j_g)] = g

    ### Gamma_2 (Right interface of O2) 
    i_g2 = nx_off_2 # Global x index for x=2
    for j_s in range(1, ny2 - 1): # j_s in O2 grid (y in (0, 2))
        j_g = j_s + j_off
        if j_g < ny_off_1: continue # O3 is only from y=1 to y=2

        # Interface point (x=2, y=j_g*H)
        u_interface = interface_values.get((i_g2, j_g))
        if u_interface is None: continue

        # Interior layer is at x=2-H (i_s = nx2 - 2)
        i_s_interior = nx2 - 2
        k_interior = map2.get((i_s_interior, j_s))
        
        if k_interior is not None:
            u_interior = u2_s[k_interior]

            # Flux for O3 (Normal direction points Right, du/dx is positive)
            # g_O3 = (u_Gamma - u_int) / H$
            g = (u_interface - u_interior) / H
            flux_values[(i_g2, j_g)] = g
            
    return flux_values