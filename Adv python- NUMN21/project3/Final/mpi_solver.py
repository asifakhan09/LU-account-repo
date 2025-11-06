from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
from apartment import Room, Apartment, TOL
from subdomain_solver import SubdomainSolver
from argparse import ArgumentParser
import time
T_WALL, T_WINDOW, T_HEATER = 15.0, 5.0, 40.0
start_time = time.time()

"""
MPI driver for the Dirichlet-Neumann solver.

Run:
    To run, do: 
        mpiexec -n 4 python mpi_solver.py -r 40 -w 0.8 -mi 30
    in terminal. You can change params if you want. 
    Params:
        n: number of processors
        r: sets resolution. r = 40 -> H = 1/40
        w: omega (relaxation factor)
        mi: max iterations of Dirichlet/Neumann
"""


def build_apartment_and_rooms(h):
    """ Helper function to set up apartment. 

    Params: 
        h: float 
            Step size (global param)
    
    Returns: 
        apartment (Apartment object): Instance of the Apartment class in apartment.py, built from all rooms.
    """
    rooms = []

    bounds1 = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
    bc1 = {
        "left": [{"y_min": 0.0, "y_max": 1.0, "type": "Heater", "value": T_HEATER}],
        "right": [{"y_min": 0.0, "y_max": 1.0, "type": "Interface"}],
        "top": [{"x_min": 0.0, "x_max": 1.0, "type": "Wall", "value": T_WALL}],
        "bottom": [{"x_min": 0.0, "x_max": 1.0, "type": "Wall", "value": T_WALL}],
    }
    rooms.append(Room(1, bounds1, "right", bc1, h))

    bounds2 = {"x_min": 1.0, "x_max": 2.0, "y_min": 0.0, "y_max": 2.0}
    bc2 = {
        "left": [
            {"y_min": 0.0, "y_max": 1.0, "type": "Interface"},
            {"y_min": 1.0, "y_max": 2.0, "type": "Wall", "value": T_WALL},
        ],
        "right": [
            {"y_min": 0.0, "y_max": 1.0, "type": "Wall", "value": T_WALL},
            {"y_min": 1.0, "y_max": 2.0, "type": "Interface"},
        ],
        "top": [{"x_min": 1.0, "x_max": 2.0, "type": "Heater", "value": T_HEATER}],
        "bottom": [{"x_min": 1.0, "x_max": 2.0, "type": "Window", "value": T_WINDOW}],
    }
    rooms.append(Room(2, bounds2, None, bc2, h))

    bounds3 = {"x_min": 2.0, "x_max": 3.0, "y_min": 1.0, "y_max": 2.0}
    bc3 = {
        "left": [{"y_min": 1.0, "y_max": 2.0, "type": "Interface"}],
        "right": [{"y_min": 1.0, "y_max": 2.0, "type": "Heater", "value": T_HEATER}],
        "top": [{"x_min": 2.0, "x_max": 3.0, "type": "Wall", "value": T_WALL}],
        "bottom": [{"x_min": 2.0, "x_max": 3.0, "type": "Wall", "value": T_WALL}],
    }
    rooms.append(Room(3, bounds3, "left", bc3, h))

    bounds4 = {"x_min": 2.0, "x_max": 2.5, "y_min": 0.5, "y_max": 1.0}
    bc4 = {
        "left": [{"y_min": 0.5, "y_max": 1.0, "type": "Interface"}],
        "right": [{"y_min": 0.5, "y_max": 1.0, "type": "Wall", "value": T_WALL}],
        "top": [{"x_min": 2.0, "x_max": 2.5, "type": "Wall", "value": T_WALL}],
        "bottom": [{"x_min": 2.0, "x_max": 2.5, "type": "Heater", "value": T_HEATER}],
    }
    rooms.append(Room(4, bounds4, "left", bc4, h))

    apartment = Apartment(rooms, h)
    return apartment


def partition_room_ids(all_rooms, size, rank):
    """
    Partition the list of rooms among MPI ranks. Each room is assigned 
    to (exactly) one rank based on its index modulo `size`. So a rank
    can have multiple rooms, but a room can only be assigned to one rank.

    Params:
        all_rooms: list[Room]
            List of all Room objects in the apartment.
        size: int
            Total number of MPI processes.
        rank: int
            Rank ID of the current MPI process.

    Returns:
        list
            Subset of Room objects assigned to this rank.
    """
    return [r for idx, r in enumerate(all_rooms) if idx % size == rank]


def compute_local_fluxes(solver, local_rooms):
    """
    Compute the Neumann flux contributions from Dirichlet subdomains owned by this rank.
    For each Dirichlet room, the flux at interface is computed as the difference
    between the interior solution (u_int) and the current interface temperature 
    (T_interface), divided by the grid spacing (h). This gives the normal derivative
    (heat flux) across the interface boundary, which will later be used to solve the 
    Neumann problems in adjacent subdomains.

    A fuzzy equality check (1e-12) is used to determine whether a node lies on the left
    or right boundary of the room to avoid floating point precision issues.

    Params:
        solver: DirichletNeumannSolver
                The solver object containing subdomain data, interface values, and grid spacing.
        local_rooms: list[Room]
                List of Room objects assigned to this MPI rank.

    Returns:
        new_flux: dict
            Dictionary mapping interface node coordinates (i_g, j_g) to computed flux values.
            These fluxes represent the normal derivative across Dirichlet interfaces.
    """

    new_flux = {}
    for room in local_rooms:
        if room.get_type() != "Dirichlet":
            continue
        if room.id not in solver.subdomain_data:
            continue
        u_s, mapping, (nx_s, _), _, j_off = solver.subdomain_data[room.id]

        # loop over interface_points currently present in solver.interface_values
        for (i_g, j_g), T_interface in solver.interface_values.items():
            x = i_g * solver.h
            y = j_g * solver.h
            bc_type, _ = room.get_bc_at(x, y)
            if bc_type != "Interface":
                continue

            # left interface of that dirichlet room
            if abs(x - room.bounds["x_min"]) < TOL: #fuzzy equality check (float errors)
                i_s_interior = 1
                j_s_interior = j_g - j_off
                k = mapping.get((i_s_interior, j_s_interior))
                if k is None:
                    continue
                u_int = u_s[k]
                new_flux[(i_g, j_g)] = (u_int - T_interface) / solver.h

            # right interface
            elif abs(x - room.bounds["x_max"]) < TOL:
                i_s_interior = nx_s - 2
                j_s_interior = j_g - j_off
                k = mapping.get((i_s_interior, j_s_interior))
                if k is None:
                    continue
                u_int = u_s[k]
                new_flux[(i_g, j_g)] = (T_interface - u_int) / solver.h

    return new_flux 


def compute_local_neumann_updates(solver, local_rooms):

    """
    Collect interface values (u_star) from Neumann subdomains on this rank.
    After solving the Neumann problems locally this function extracts the computed
    temperature values at interface nodes. These values are used to update the 
    global interface values during the relaxation step on the root process.

    Params:
        solver : SubdomainSolver
            The solver object containing subdomain data, interface values, and grid spacing.
        local_rooms : list[Room]
            List of Room objects assigned to this MPI rank.

    Returns:
        updates: dict
            Dictionary mapping interface node coordinates (i_g, j_g) to solution 
            values (u_star) computed from the Neumann subdomains.
    """
    updates = {}
    for room in local_rooms:
        if not room.get_type().startswith("Neumann"):
            continue
        if room.id not in solver.subdomain_data:
            continue
        u_s, mapping, (_, _), i_off, j_off = solver.subdomain_data[room.id]

        # iterate unknown nodes in mapping and pick those that are interface nodes
        for (i_s, j_s), k in mapping.items():
            i_g = i_s + i_off
            j_g = j_s + j_off
            x = i_g * solver.h
            y = j_g * solver.h
            bc_type, _ = room.get_bc_at(x, y)
            if bc_type != "Interface":
                continue
            # store candidate u_s
            updates[(i_g, j_g)] = float(u_s[k])
    return updates


def plot_solution(solver, U_global):
    """
    Plot assembled global temperature field.

    Params: 
        solver: SubdomainSolver
            The solver object containing subdomain data, interface values, and grid spacing.
        
        U_global: np.array
            2D numpy array containing the assembled global temperature field of the apartment. 
    
    Returns: 
        None
    """
    x_max = max(r.bounds["x_max"] for r in solver.apartment.rooms)
    y_max = max(r.bounds["y_max"] for r in solver.apartment.rooms)

    masked = np.ma.array(U_global, mask=(U_global == -10.0))
    plt.figure(figsize=(9, 6))
    plt.imshow(
        masked,
        origin="lower",
        extent=[0, x_max, 0, y_max],
        cmap="hot",
        vmin=5,
        vmax=40
    )
    plt.colorbar(label="Temperature (°C)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Temperature Distribution — Final DN Iteration")

    #draw rom outlines
    for r in solver.apartment.rooms:
        x0, x1 = r.bounds["x_min"], r.bounds["x_max"]
        y0, y1 = r.bounds["y_min"], r.bounds["y_max"]
        plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k--", linewidth=1)

    plt.show()


def mpi_run(h,omega,max_iter):
    """
    Run the Dirichlet/Neumann domain decomposition solver in parallel using MPI.
    This function distributes the subdomains across the MPI ranks,
    performs parallel Dirichlet/Neumann iterations, and assembles the final
    temperature field on the root process (after convergence or after the maximum
    number of iterations).

    Algoritm steps:
        1. Initialize MPI and distribute room assignments to ranks.
        2. Solve Dirichlet problems locally on each rank using the current interface values.
        3. Compute flux contributions from Dirichlet solutions and gather them on root.
        4. Root merges fluxes, broadcasts updated flux field to all ranks.
        5. Solve Neumann problems locally using merged flux.
        6. Gather Neumann interface value updates (u_star) at root.
        7. Root applies relaxation to update global interface values and broadcasts them.
        8. Repeat steps 2 to 7 until convergence or max iterations reached.
        9. Root assembles the global solution and plots it.

    Params:
        h: float
            Grid spacing.
        omega: float
            Relaxation parameter used when updating interface values.
        max_iter: int
            Maximum number of Dirichlet/Neumann iterations.

    Returns
        None
            The final solution is assembled and plotted on the root rank (rank 0).
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Build identical apartment + solver on each rank
    apartment = build_apartment_and_rooms(h)
    solver = SubdomainSolver(apartment, h)

    # Partition rooms 
    local_rooms = partition_room_ids(apartment.rooms, size, rank)
    print("local rooms:", local_rooms)

    #Print assignments
    if rank == 0:
        print(f"[root] MPI size={size}. Rooms: {[r.id for r in apartment.rooms]}")
    local_room_ids = [r.id for r in local_rooms]
    print(f"[rank {rank}] assigned rooms: {local_room_ids}")

    # initialize interface values on root, then broadcast
    if rank == 0:
        solver.initialize_interfaces()
        interface_values = solver.interface_values.copy()
        interface_flux = solver.interface_flux.copy()
    else:
        interface_values = None
        interface_flux = None

    # Broadcast initial interface_values
    interface_values = comm.bcast(interface_values, root=0)
    interface_flux = comm.bcast(interface_flux, root=0)

    # Main DN iterations (same ordering as before: Dirichlet -> flux -> Neumann -> relax)
    for it in range(max_iter):
        
        # Solve Dirichlet rooms owned by this rank
        for room in local_rooms:
            if room.get_type().startswith("Dirichlet"):
            # use current global interface_values and interface_flux
                solver.subdomain_data[room.id] = solver.solve_subdomain(room, solver.interface_values, solver.interface_flux)

        # Compute local fluxes from Dirichlet rooms (this rank)
        local_flux = compute_local_fluxes(solver, local_rooms)

        # Gather all local_flux dicts to root
        all_local_fluxes = comm.gather(local_flux, root=0)

        # Root merges flux dicts
        if rank == 0:
            merged_flux = {}
            for d in all_local_fluxes:
                if d:
                    merged_flux.update(d)
            # store in root solver
            solver.interface_flux = merged_flux.copy()
        else:
            merged_flux = None
        
        #broadcast and update merged flux
        merged_flux = comm.bcast(merged_flux, root=0)
        solver.interface_flux = merged_flux.copy()

        # Solve Neumann rooms owned by this rank (use merged_flux)
        for room in local_rooms:
            if room.get_type().startswith("Neumann"):
                solver.subdomain_data[room.id] = solver.solve_subdomain(room, solver.interface_values,solver.interface_flux)

        # Each rank computes its Neumann updates (u_star) for interface nodes
        local_updates = compute_local_neumann_updates(solver, local_rooms)

        # Gather local_updates to root
        all_local_updates = comm.gather(local_updates, root=0)

        # Root performs relaxation and updates interface_values, then broadcasts them
        if rank == 0:

            old_interface_values = solver.interface_values.copy()

            # merge updates 
            merged_updates = {}
            for d in all_local_updates:
                if d:
                    merged_updates.update(d)

            # relax
            for (i_g, j_g), u_s in merged_updates.items():
                if (i_g, j_g) in solver.interface_values:
                    u_old = solver.interface_values[(i_g, j_g)]
                    solver.interface_values[(i_g, j_g)] = omega * u_s + (1.0 - omega) * u_old
            
            keys = sorted(solver.interface_values.keys())
            old_vec = [old_interface_values.get(k, solver.interface_values[k]) for k in keys]
            new_vec = [solver.interface_values[k] for k in keys]
            residual = (sum((nv - ov) ** 2 for ov, nv in zip(old_vec, new_vec))) ** 0.5
            print(f"[root] Iter {it+1}/{max_iter} — Residual: {residual:.3e}, updated interface nodes: {len(merged_updates)}")

            
            interface_values_to_send = solver.interface_values.copy()
        else:
            interface_values_to_send = None

        # broadcast new global interface_values, and update local copy for next iteration
        interface_values_to_send = comm.bcast(interface_values_to_send, root=0)
        solver.interface_values = interface_values_to_send.copy()

    # End iterations

    # Gather all subdomain_data dicts to root so root can assemble final solution
    all_subdomain_data = comm.gather(solver.subdomain_data, root=0)

    if rank == 0:
        # merge dictionaries from every rank
        merged_subdomain_data = {}
        for d in all_subdomain_data:
            if d:
                merged_subdomain_data.update(d)

        # attach to root solver and assemble/plot
        solver.subdomain_data = merged_subdomain_data
        U_global = solver.assemble_global_solution()
        print("[root] MPI solve finished and plotted.")
        print("--- %s seconds ---" % (time.time() - start_time))
        
        plot_solution(solver, U_global)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-r", "--resolution", type = int, default = 40, help = "Grid resolution, 1/H")
    parser.add_argument("-w", "--omega", type = float, default = 0.8, help = "Relaxation factor")
    parser.add_argument("-mi", "--m_iterations", type = int, default = 10, help = "Number of Dirichlet/Neumann iterations")

    args = parser.parse_args()

    mpi_run(1/args.resolution, args.omega, args.m_iterations)

