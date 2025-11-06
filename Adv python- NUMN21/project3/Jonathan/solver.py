import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from project3.Final.apartment import Apartment
from project3.Final.global_params import H, TOL, OMEGA, MAX_ITER


class DirichletNeumannSolver:

    def __init__(self, apartment: Apartment):
        self.apartment = apartment
        self.interface_values = {}
        self.interface_flux = {}
        self.subdomain_data = {}  # store solutions per room
    
    def initialize_interfaces(self):
        self.apartment.build_interface_points()
        self.interface_values = self.apartment.interface_points.copy()
        self.interface_flux = {coord: 0.0 for coord in self.interface_values}

    def update_interface_flux(self):
        """
        Compute interface flux (Neumann BC) based on the Dirichlet subdomain solutions.
        Equivalent to calculate_dirichlet_flux() in the old solver.
        """
        new_flux = {}

        for room in self.apartment.rooms:
            room_type = room.get_type()
            if not room_type.startswith("Dirichlet"):
                continue

            u_s, mapping, (nx_s, _), _, j_off = self.subdomain_data[room.id]

            # loop through interface boundary points for this room
            for (i, j), T_interface in self.interface_values.items():
                x = i * H
                y = j * H

                bc_type, _ = room.get_bc_at(x, y)
                if bc_type != "Interface":
                    continue  # not an interface point for this room

                # Determine whether this interface is left or right for this room
                if abs(x - room.bounds["x_min"]) < TOL:  # left
                    i_s_interior = 1
                    j_s_interior = j - j_off
                    k_interior = mapping.get((i_s_interior, j_s_interior))
                    if k_interior is None:
                        continue
                    u_interior = u_s[k_interior]
                    # Normal points left → flux = (u_interior - u_interface)/H
                    new_flux[(i, j)] = (u_interior - T_interface) / H

                elif abs(x - room.bounds["x_max"]) < TOL:  # right
                    i_s_interior = nx_s - 2
                    j_s_interior = j - j_off
                    k_interior = mapping.get((i_s_interior, j_s_interior))
                    if k_interior is None:
                        continue
                    u_interior = u_s[k_interior]
                    # Normal points right → flux = (u_interface - u_interior)/H
                    new_flux[(i, j)] = (T_interface - u_interior) / H

        self.interface_flux = new_flux

    def update_interface_values(self):
        """
        Relax the interface temperature values using the Neumann solutions.
        Equivalent to relaxation step in old solver (O1 and O3).
        """
        new_values = self.interface_values.copy()

        for room in self.apartment.rooms:
            room_type = room.get_type()
            if not room_type.startswith("Neumann"):
                continue

            u_s, mapping, (nx_s, _), _, j_off = self.subdomain_data[room.id]

            for (i, j) in list(self.interface_values.keys()):
                x = i * H
                y = j * H
                bc_type, _ = room.get_bc_at(x, y)
                if bc_type != "Interface":
                    continue

                # Determine whether this interface is left or right for this room
                if abs(x - room.bounds["x_min"]) < TOL:  # left
                    i_s = 0
                    j_s = j - j_off
                elif abs(x - room.bounds["x_max"]) < TOL:  # right
                    i_s = nx_s - 1
                    j_s = j - j_off
                else:
                    continue

                k = mapping.get((i_s, j_s))
                if k is None:
                    continue

                u_star = u_s[k]                        # Neumann subdomain value
                u_old = self.interface_values[(i, j)]  # previous interface value
                u_relaxed = OMEGA * u_star + (1 - OMEGA) * u_old
                new_values[(i, j)] = u_relaxed

        self.interface_values = new_values

    def solve_subdomain(self, room, boundary_values, interface_flux=None):
        """
        Direct lift of your original solve_subdomain, 
        only changing how BCs and geometry are accessed.
        """

        # Determine room type (Dirichlet vs Neumann sides) and get geometry of room
        room_type = room.get_type()
        nx_s, ny_s, i_off, j_off = room.get_geometry()

        mapping = {}
        counter = 0

        for j_s in range(1, ny_s - 1):
            if room_type == "Dirichlet":
                i_start, i_end = 1, nx_s - 1
            elif room_type == "Neumann_right":
                i_start, i_end = 1, nx_s
            elif room_type == "Neumann_left":
                i_start, i_end = 0, nx_s - 1
            else:
                #i_start, i_end = 1, nx_s - 1  # fallback
                raise ValueError("Unknown room type")

            for i_s in range(i_start, i_end):
                mapping[(i_s, j_s)] = counter
                counter += 1

        A = lil_matrix((counter, counter))
        b = np.zeros(counter)

        for (i_s, j_s), k in mapping.items():
            i_g, j_g = i_s + i_off, j_s + j_off

            # Detect Neumann interface 
            is_neumann_interface = False
            if room_type == "Neumann_right" and i_s == nx_s - 1:
                is_neumann_interface = True
                interior_neighbor_i_s = i_s - 1
                normal_sign = -1.0  # outward normal 
            elif room_type == "Neumann_left" and i_s == 0:
                is_neumann_interface = True
                interior_neighbor_i_s = i_s + 1
                normal_sign = 1.0  # outward normal 

            # If Neumann interface node 
            if is_neumann_interface:
                A[k, k] = -3.0
                A[k, mapping[(interior_neighbor_i_s, j_s)]] = 1.0

                g = 0.0
                if interface_flux is not None and (i_g, j_g) in interface_flux:
                    g = interface_flux[(i_g, j_g)]
                b[k] = normal_sign * H * g

                # handle vertical neighbors
                for (di, dj) in [(0, 1), (0, -1)]:
                    i_ns, j_ns = i_s + di, j_s + dj
                    i_ng, j_ng = i_g + di, j_g + dj
                    neighbor_coord_s = (i_ns, j_ns)

                    if neighbor_coord_s in mapping:
                        A[k, mapping[neighbor_coord_s]] = 1.0
                    else:
                        bc_type, bc_val = self.apartment.get_bc_at(i_ng * H, j_ng * H)
                        if bc_type == "Fixed" and bc_val is not None:
                            b[k] -= bc_val

            else:
                # Interior/Dirichlet node 
                A[k, k] = -4.0
                for (di, dj) in [(1,0),(-1,0),(0,1),(0,-1)]:
                    i_ns, j_ns = i_s + di, j_s + dj
                    i_ng, j_ng = i_g + di, j_g + dj
                    neighbor_s = (i_ns, j_ns)

                    if 0 <= i_ns < nx_s and 0 <= j_ns < ny_s:
                        if neighbor_s in mapping:
                            A[k, mapping[neighbor_s]] = 1.0
                        else:
                            bc_type, bc_val = self.apartment.get_bc_at(i_ng * H, j_ng * H)
                            if room_type == "Dirichlet" and (i_ng, j_ng) in boundary_values:
                                b[k] -= boundary_values[(i_ng, j_ng)]
                            elif bc_type == "Fixed" and bc_val is not None:
                                b[k] -= bc_val
                    else:
                        bc_type, bc_val = self.apartment.get_bc_at(i_ng * H, j_ng * H)
                        if bc_type == "Fixed" and bc_val is not None:
                            b[k] -= bc_val

        A_csr = csr_matrix(A)
        u_s = spsolve(A_csr, b)

        return u_s, mapping, (nx_s, ny_s), i_off, j_off

    def run(self):
        """Dirichlet–Neumann iteration with correct ordering for good convergence."""

        self.initialize_interfaces()
        self.norm_history = []

        for iteration in range(MAX_ITER):
            # Solve Dirichlet rooms 
            for room in [r for r in self.apartment.rooms if r.get_type() == "Dirichlet"]:
                u_s, mapping, (nx_s, ny_s), i_off, j_off = self.solve_subdomain(
                    room, boundary_values=self.interface_values, interface_flux=self.interface_flux
                )
                self.subdomain_data[room.id] = (u_s, mapping, (nx_s, ny_s), i_off, j_off)

            # Compute flux solved Dirichlet rooms
            self.update_interface_flux()   # must compute flux using subdomain_data & offsets

            # Solve Neumann rooms using flux
            for room in [r for r in self.apartment.rooms if r.get_type().startswith("Neumann")]:
                u_s, mapping, (nx_s, ny_s), i_off, j_off = self.solve_subdomain(
                    room, boundary_values=self.interface_values, interface_flux=self.interface_flux
                )
                self.subdomain_data[room.id] = (u_s, mapping, (nx_s, ny_s), i_off, j_off)

            # Relax interface Dirichlet values using Neumann solutions
            old_values = np.array(list(self.interface_values.values()))
            self.update_interface_values()   # uses subdomain_data for Neumann rooms
            new_values = np.array(list(self.interface_values.values()))

            # compute residual to check convergence
            residual = np.linalg.norm(new_values - old_values)
            self.norm_history.append(residual)
            print(f"Iteration {iteration+1}/{MAX_ITER} — Residual: {residual:.3e}")

            if hasattr(self, "tol") and residual < self.tol:
                print(f"Converged after {iteration+1} iterations.")
                break

        print("Dirichlet–Neumann iteration finished.")


    def assemble_global_solution(self):
        """
        Assemble the global temperature field from all room subdomain solutions.
        Returns U_global as a 2D numpy array.
        """
        # Determine full domain extent
        x_max = max(r.bounds["x_max"] for r in self.apartment.rooms)
        y_max = max(r.bounds["y_max"] for r in self.apartment.rooms)
        nx = int(round(x_max / H)) + 1
        ny = int(round(y_max / H)) + 1

        # Initialize with NaN for masking
        U_global = np.full((ny, nx), np.nan)

        # Fill in subdomain solutions (interior + Neumann interfaces)
        for room in self.apartment.rooms:
            u_s, mapping, (nx_s, ny_s), i_off, j_off = self.subdomain_data[room.id]

            # Map local solved values
            for (i_s, j_s), k in mapping.items():
                i_g = i_s + i_off
                j_g = j_s + j_off
                U_global[j_g, i_g] = u_s[k]

            # Fill fixed boundary values for visualization
            for i_s in range(nx_s):
                for j_s in range(ny_s):
                    i_g = i_s + i_off
                    j_g = j_s + j_off
                    x_g = i_g * H
                    y_g = j_g * H

                    if np.isnan(U_global[j_g, i_g]):
                        bc_type, value = room.get_bc_at(x_g, y_g)
                        if bc_type == "Fixed":
                            U_global[j_g, i_g] = value
                        elif bc_type == "Interface":
                            # fill from interface values if available
                            if (i_g, j_g) in self.interface_values:
                                U_global[j_g, i_g] = self.interface_values[(i_g, j_g)]

        # Mask areas not inside any room
        for j in range(ny):
            for i in range(nx):
                x, y = i * H, j * H
                if not any(r.contains(x, y) for r in self.apartment.rooms):
                    U_global[j, i] = -10.0  # outside mask value


        return U_global
    
    def plot_solution(self, U_global):
        """
        Plot assembled global temperature field.
        """
        x_max = max(r.bounds["x_max"] for r in self.apartment.rooms)
        y_max = max(r.bounds["y_max"] for r in self.apartment.rooms)

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

        # Optional: draw room outlines
        for r in self.apartment.rooms:
            x0, x1 = r.bounds["x_min"], r.bounds["x_max"]
            y0, y1 = r.bounds["y_min"], r.bounds["y_max"]
            plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k--", linewidth=1)

        plt.show()



