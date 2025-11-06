import numpy as np
import matplotlib.pyplot as plt
from jonathan_domain_setup import solve_subdomain, get_subdomain_params, calculate_dirichlet_flux, get_fixed_bc, H, T_WALL, T_HEATER, T_WINDOW

MAX_ITER = 30 
OMEGA = 0.8 
T_INITIAL_GUESS = 15.0 # Initial guess for all interface values



### Initial Data and Storage ####

# Storage for solutions u_s, mapping, and size for each subdomain
SUBDOMAIN_DATA = {} 

# Interface boundary values (Dirichlet conditions T^k for O2)
# Key: (global_i, global_j), Value: Temperature T
INTERFACE_POINTS = {}

# Global indices for interfaces
nx_off_1 = int(1.0 / H) # Global i index for x=1 (20)
nx_off_2 = int(2.0 / H) # Global i index for x=2 (40)
ny_off_1 = int(1.0 / H) # Global j index for y=1 (20)

# Gamma_1 (interface with O1)
# j_g runs from 1 to ny_off_1 - 1 (1 to 19)
for j_g in range(1, ny_off_1):
    INTERFACE_POINTS[(nx_off_1, j_g)] = T_INITIAL_GUESS

# Gamma_2 (interface with O3)
# O3 starts at y=1 (j_g=20). The interface nodes start at j_g = 21.
# j_g runs from ny_off_1 + 1 to int(2.0 / H) - 1 (21 to 39)
for j_g in range(ny_off_1 + 1, int(2.0 / H)):
    INTERFACE_POINTS[(nx_off_2, j_g)] = T_INITIAL_GUESS
    
# Check that we have 38 interface points (19 for each interface)
print(f"Total initial interface points defined: {len(INTERFACE_POINTS)}") 
    




### Main Dirichlet-Neumann Iteration Loop ###

def run_dn_iteration(interface_values: dict, interface_flux: dict) -> tuple[dict, dict]:
    """Performs one full Dirichlet-Neumann sweep."""
    

    ## Step 1: Solve O2 (Dirichlet at Gamma_1 and Gamma_2) ##

    # O2 uses interface_values (T^k) as Dirichlet conditions
    u2_new, map2, (nx2, ny2) = solve_subdomain(subdomain_id=2, boundary_values=interface_values)
    
    # Calculate the Dirichlet flux (g = du/dx) from the new O2 solution and the relaxed interface values
    o2_flux = calculate_dirichlet_flux(u2_new, map2, nx2, ny2, interface_values)
    
    # Store the O2 
    SUBDOMAIN_DATA[2] = (u2_new, map2, (nx2, ny2))
    

    ## Step 2: Solve O1 and O3 (Neumann at Gamma_1 and Gamma_2) ##

    # O1 and O3 use the calculated O2 flux (g) as the Neumann condition
    u1_new, map1, (nx1, ny1) = solve_subdomain(subdomain_id=1, boundary_values=interface_values, 
                                               interface_flux=o2_flux)
    
    u3_new, map3, (nx3, ny3) = solve_subdomain(subdomain_id=3, boundary_values=interface_values, 
                                               interface_flux=o2_flux)
    
    # Store solutions
    SUBDOMAIN_DATA[1] = (u1_new, map1, (nx1, ny1))
    SUBDOMAIN_DATA[3] = (u3_new, map3, (nx3, ny3))
    


    ## Step 3: Relaxation (Update the interface values for the next iteration) ##
    
    # Initialize the new interface values based on the keys of the old interface values
    new_interface_values = interface_values.copy()
    
    # The new Dirichlet value at the interface is the solution of the Neumann problem 
    # at the interface nodes (so u1_new and u3_new)
    
    # Relaxation for O1 interface nodes (x=1)
    for (i_s, j_s), k1 in map1.items():

        # Check if the O1 point is on the interface (i_s = nx1 - 1, which is nx_off_1)
        if i_s == nx_off_1: 
            i_g, j_g = i_s + 0, j_s + 0 # Global coord
            
            # Only update if it is a point in INTERFACE_POINTS
            if (i_g, j_g) in interface_values: 
                u_star = u1_new[k1] # Solution from Neumann problem
                u_old = interface_values[(i_g, j_g)]
                
                # Relaxation
                u_relaxed = OMEGA * u_star + (1.0 - OMEGA) * u_old
                new_interface_values[(i_g, j_g)] = u_relaxed

    # Relaxation for O3 interface nodes (x=2)
    for (i_s, j_s), k3 in map3.items():
        # Check if the O3 point is on the interface (i_s = 0)
        if i_s == 0: 
            # O3 coordinates are (i_s + nx_off_2, j_s + ny_off_1)
            i_g, j_g = i_s + nx_off_2, j_s + ny_off_1 
            
            # Only update if it is a point in INTERFACE_POINTS
            if (i_g, j_g) in interface_values:
                u_star = u3_new[k3] # Solution from Neumann problem
                u_old = interface_values[(i_g, j_g)]
                
                # Relaxation
                u_relaxed = OMEGA * u_star + (1.0 - OMEGA) * u_old
                new_interface_values[(i_g, j_g)] = u_relaxed
    
    return new_interface_values, o2_flux




### Run the iteration ###

print(f"Starting Dirichlet-Neumann Iteration (Omega={OMEGA}, H={H})")
current_interface_values = INTERFACE_POINTS.copy()
# Flux is initialized to zero (Dirichlet-Neumann requires a flux of 0 for the first Neumann solve)
current_interface_flux = {coord: 0.0 for coord in INTERFACE_POINTS} 
norm_history = []
k = 0 # Initialize iteration counter outside the loop

for k in range(1, MAX_ITER + 1):
    # Ensure consistency before difference calculation
    old_values = np.array(list(current_interface_values.values()))
    
    current_interface_values, current_interface_flux = run_dn_iteration(
        current_interface_values, current_interface_flux
    )
    
    # Ensure consistency after update
    new_values = np.array(list(current_interface_values.values()))



### Reconstruct and Plot Final Solution ###

def plot_final_solution(k_final):
    Lx, Ly = 3.0, 2.0
    nx = int(Lx / H) + 1
    ny = int(Ly / H) + 1
    
    U_global = np.full((ny, nx), np.nan)
    
    # Helper to check global domain inside status
    def is_inside(i: int, j: int) -> bool:
        x, y = i * H, j * H
        is_omega_1 = (0.0 <= x <= 1.0) and (0.0 <= y <= 1.0)
        is_omega_2 = (1.0 <= x <= 2.0) and (0.0 <= y <= 2.0)
        is_omega_3 = (2.0 <= x <= 3.0) and (1.0 <= y <= 2.0)
        return is_omega_1 or is_omega_2 or is_omega_3

    # Map subdomain solutions back to the global grid
    for sub_id in SUBDOMAIN_DATA:
        u_s, mapping, (nx_s, ny_s) = SUBDOMAIN_DATA[sub_id]
        Lx_s, Ly_s, i_off, j_off, name = get_subdomain_params(sub_id)
        
        # Map ALL solved points (interior + Neumann interfaces)
        for (i_s, j_s), k_local in mapping.items():
            i_g, j_g = i_s + i_off, j_s + j_off
            U_global[j_g, i_g] = u_s[k_local]

        # Fill subdomain FIXED boundaries for visualization
        for i_s in range(nx_s):
            for j_s in range(ny_s):
                i_g, j_g = i_s + i_off, j_s + j_off
                x_g, y_g = i_g * H, j_g * H
                
                # Check if it's a fixed boundary node (not an unknown, not an interface point)
                is_fixed_boundary_node = is_inside(i_g, j_g) and U_global[j_g, i_g] is np.nan
                    
                if is_fixed_boundary_node:
                    T_BC = get_fixed_bc(x_g, y_g, name)
                    
                    # Special checks for the hole walls (T=15)
                    if name == "O1" and j_s == ny_s - 1:
                        if x_g < 1.0 and (y_g == 1.0): T_BC = T_WALL 
                    elif name == "O3" and j_s == 0:
                        if x_g > 2.0 and (y_g == 1.0): T_BC = T_WALL 
                        
                    U_global[j_g, i_g] = T_BC

    # Fill the hole (-10.0 for masking)
    for j in range(ny):
        for i in range(nx):
            if not is_inside(i, j):
                U_global[j, i] = -10.0

    # Plotting
    plt.figure(figsize=(9, 6))
    masked_array = np.ma.array(U_global, mask=(U_global == -10.0)) 

    plt.imshow(masked_array, 
               origin="lower", 
               extent=[0, Lx, 0, Ly], 
               cmap="hot", 
               vmin=T_WINDOW, 
               vmax=T_HEATER)
    plt.colorbar(label="Temperature ($\mathbf{^{\circ}C}$)")

    # Add room boundaries
    plt.axvline(x=1.0, ymin=0.0, ymax=0.5, color='black', linestyle='--')
    plt.axvline(x=2.0, ymin=0.0, ymax=1.0, color='black', linestyle='--')
    plt.axhline(y=1.0, xmin=2.0/Lx, xmax=3.0/Lx, color='black', linestyle='--') 

    plt.title(f"D-N Iteration ($\mathbf{{k={k_final}}}$) Temperature Distribution (True Neumann)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

plot_final_solution(k)