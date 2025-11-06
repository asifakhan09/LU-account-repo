import numpy as np
import matplotlib.pyplot as plt
from domain_setup import solve_subdomain,calculate_dirichlet_flux, get_room_geometry, get_room_type, get_fixed_bc, H, T_HEATER, T_WINDOW, ROOM_CONFIGS

MAX_ITER = 30
OMEGA = 0.8
T_INITIAL_GUESS = 15.0

### Build interface points automatically from ROOM_CONFIG ###

INTERFACE_POINTS = {}

for room in ROOM_CONFIGS:
    room_type = get_room_type(room)
    nx_s, ny_s, i_off, j_off = get_room_geometry(room, H)

    # Right interface points
    if room_type == "Neumann_right" or (room_type == "Dirichlet" and any(r['interface'] == 'left' for r in ROOM_CONFIGS if r != room)):
        i_global = i_off + nx_s - 1
        # loop over vertical neighbors
        for j_local in range(1, ny_s - 1):
            j_global = j_off + j_local
            INTERFACE_POINTS[(i_global, j_global)] = T_INITIAL_GUESS

    # Left interface points
    if room_type == "Neumann_left" or (room_type == "Dirichlet" and any(r['interface'] == 'right' for r in ROOM_CONFIGS if r != room)):
        i_global = i_off
        for j_local in range(1, ny_s - 1):
            j_global = j_off + j_local
            INTERFACE_POINTS[(i_global, j_global)] = T_INITIAL_GUESS


# Categorize rooms
DIRICHLET_ROOMS = [r for r in ROOM_CONFIGS if get_room_type(r) == "Dirichlet"]
NEUMANN_ROOMS = [r for r in ROOM_CONFIGS if get_room_type(r).startswith("Neumann")]

# Storage for solutions
SUBDOMAIN_DATA = {}


### Helper functions ###

def compute_flux_all_dirichlet(interface_values):
    """Compute flux for all Dirichlet rooms"""
    flux = {}
    for room in DIRICHLET_ROOMS:
        u, mapping, (nx_s, ny_s), i_off, j_off = SUBDOMAIN_DATA[room["id"]]
        # Reuse the existing function for each Dirichlet room
        local_flux = calculate_dirichlet_flux(
            u, mapping, nx_s, ny_s, interface_values, i_off=i_off, j_off=j_off
        )
        flux.update(local_flux)
    return flux


def relax_all_neumann_rooms(interface_values, interface_flux):
    """Update interface values for all Neumann rooms"""
    new_interface_values = interface_values.copy()
    for room in NEUMANN_ROOMS:
        u_s, mapping, _, i_off, j_off = SUBDOMAIN_DATA[room["id"]]
        for (i_s, j_s), k_local in mapping.items():
            i_g, j_g = i_s + i_off, j_s + j_off
            if (i_g, j_g) in interface_values:
                u_star = u_s[k_local]
                u_old = interface_values[(i_g, j_g)]
                u_relaxed = OMEGA * u_star + (1.0 - OMEGA) * u_old
                new_interface_values[(i_g, j_g)] = u_relaxed
    return new_interface_values


### Dirichlet–Neumann iteration ###

def run_dn_iteration(interface_values, interface_flux):
    #Solve Dirichlet rooms
    for room in DIRICHLET_ROOMS:
        u_s, mapping, (nx_s, ny_s), i_off, j_off = solve_subdomain(
            room, boundary_values=interface_values
        )
        SUBDOMAIN_DATA[room["id"]] = (u_s, mapping, (nx_s, ny_s), i_off, j_off)

    # Compute flux from all Dirichlet rooms
    new_flux = compute_flux_all_dirichlet(interface_values)

    # Solve Neumann rooms using this flux
    for room in NEUMANN_ROOMS:
        u_s, mapping, (nx_s, ny_s), i_off, j_off = solve_subdomain(
            room, boundary_values=interface_values, interface_flux=new_flux
        )
        SUBDOMAIN_DATA[room["id"]] = (u_s, mapping, (nx_s, ny_s), i_off, j_off)

    # Relax interface values based on Neumann rooms
    new_interface_values = relax_all_neumann_rooms(interface_values, new_flux)
    return new_interface_values, new_flux


### Main iteration loop ###

current_interface_values = INTERFACE_POINTS.copy()
current_interface_flux = {coord: 0.0 for coord in INTERFACE_POINTS}

for k in range(1, MAX_ITER + 1):
    old_vals = np.array(list(current_interface_values.values()))
    current_interface_values, current_interface_flux = run_dn_iteration(
        current_interface_values, current_interface_flux
    )
    new_vals = np.array(list(current_interface_values.values()))
    diff = np.linalg.norm(new_vals - old_vals)
    print(f"Iteration {k}: interface norm change = {diff:.6e}")


### Visualization ###

def plot_final_solution():
    # compute max domain size
    Lx = max(r["bounds"]["x_max"] for r in ROOM_CONFIGS)
    Ly = max(r["bounds"]["y_max"] for r in ROOM_CONFIGS)
    nx = int(round(Lx / H)) + 1
    ny = int(round(Ly / H)) + 1

    U_global = np.full((ny, nx), np.nan)

    def is_inside(i, j):
        x, y = i * H, j * H
        for r in ROOM_CONFIGS:
            if (r["bounds"]["x_min"] <= x <= r["bounds"]["x_max"]) and (
                r["bounds"]["y_min"] <= y <= r["bounds"]["y_max"]
            ):
                return True
        return False

    # Place solved unknowns
    for _, data in SUBDOMAIN_DATA.items():
        u_s, mapping, (nx_s, ny_s), i_off, j_off = data
        for (i_s, j_s), k_local in mapping.items():
            i_g, j_g = i_s + i_off, j_s + j_off
            U_global[j_g, i_g] = u_s[k_local]

    # Fill fixed boundary nodes
    for r in ROOM_CONFIGS:
        nx_s, ny_s, i_off, j_off = get_room_geometry(r, H)
        for i_s in range(nx_s):
            for j_s in range(ny_s):
                i_g, j_g = i_s + i_off, j_s + j_off
                if not is_inside(i_g, j_g):
                    continue
                if np.isnan(U_global[j_g, i_g]):
                    x_g, y_g = i_g * H, j_g * H
                    U_global[j_g, i_g] = get_fixed_bc(x_g, y_g)

    for j in range(ny):
        for i in range(nx):
            if not is_inside(i, j):
                U_global[j, i] = -10.0

    plt.figure(figsize=(10, 6))
    #for masking holes (non rooms), set to white
    masked_array = np.ma.array(U_global, mask=(U_global == -10.0))
    plt.imshow(
        masked_array,
        origin="lower",
        extent=[0, Lx, 0, Ly],
        cmap="jet",
        vmin=T_WINDOW,
        vmax=T_HEATER,
    )
    plt.colorbar(label="Temperature (°C)")

    # Draw room borders
    for r in ROOM_CONFIGS:
        x0, x1 = r["bounds"]["x_min"], r["bounds"]["x_max"]
        y0, y1 = r["bounds"]["y_min"], r["bounds"]["y_max"]
        plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k--", lw=1)

    plt.title(f"Dirichlet–Neumann Iteration, {len(ROOM_CONFIGS)} rooms")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


plot_final_solution()

