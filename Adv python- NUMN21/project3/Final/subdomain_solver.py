import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from apartment import Apartment

class SubdomainSolver:

    """
    Solver for subdomain problems within apartment.

    This class manages the interface values, fluxes, and subdomain solutions. 
    Each subdomain corresponds to a room, and the solver can:
      - Initialize interface nodes
      - Solve individual subdomain problems
      - Assemble the global solution after all subdomains are solved.
    """

    def __init__(self, apartment: Apartment, h):
        self.apartment = apartment
        self.h = h
        self.interface_values = {}
        self.interface_flux = {}
        self.subdomain_data = {}  # store solutions per room

    def initialize_interfaces(self):
        """
        Initialize interface values and fluxes.

        This method builds the global interface point dictionary and
        sets initial fluxes to zero at all interface nodes.
        """
        self.apartment.build_interface_points()
        self.interface_values = self.apartment.interface_points.copy()
        self.interface_flux = {coord: 0.0 for coord in self.interface_values}
   

    def solve_subdomain(self, room, boundary_values, interface_flux=None):
        """
        Solves a single subdomain (room).

        Method builds the local linear system for a subdomain based on its
        boundary type (Dirichlet or Neumann), applies boundary/interface conditions,
        and solves the system.

        Params:
            room : Room
                The room object to be solved.
            boundary_values : dict
                Dictionary of current global interface values.
            interface_flux : dict, optional
                Dictionary of current interface flux values (used for Neumann sides).

        Returns:
            u_s: np.array
                Local solution vector for the subdomain.
            mapping: dict
                Maps local coordinates (i_s, j_s) to indices in u_s.
            (nx_s, ny_s): tuple of int
                Local grid shape in x and y directions.
            i_off, j_off: int
                Global index offsets for mapping local to global coordinates.
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
                b[k] = normal_sign * self.h * g

                # handle vertical neighbors
                for (di, dj) in [(0, 1), (0, -1)]:
                    i_ns, j_ns = i_s + di, j_s + dj
                    i_ng, j_ng = i_g + di, j_g + dj
                    neighbor_coord_s = (i_ns, j_ns)

                    if neighbor_coord_s in mapping:
                        A[k, mapping[neighbor_coord_s]] = 1.0
                    else:
                        bc_type, bc_val = self.apartment.get_bc_at(i_ng * self.h, j_ng * self.h)
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
                            bc_type, bc_val = self.apartment.get_bc_at(i_ng * self.h, j_ng * self.h)
                            if room_type == "Dirichlet" and (i_ng, j_ng) in boundary_values:
                                b[k] -= boundary_values[(i_ng, j_ng)]
                            elif bc_type == "Fixed" and bc_val is not None:
                                b[k] -= bc_val
                    else:
                        bc_type, bc_val = self.apartment.get_bc_at(i_ng * self.h, j_ng * self.h)
                        if bc_type == "Fixed" and bc_val is not None:
                            b[k] -= bc_val

        A_csr = csr_matrix(A)
        u_s = spsolve(A_csr, b)

        return u_s, mapping, (nx_s, ny_s), i_off, j_off


    def assemble_global_solution(self):
        """
        Assemble the global temperature field from local subdomain solutions.
        Reconstructs the domain solution grid by:
          - Mapping local subdomain solutions to global coordinates
          - Filling in fixed boundary and interface values
          - Masking points outside the apartment layout

        Returns:
            np.ndarray
                2D array of the global temperature field with masked (outside) values.
        """
        # Determine full domain extent
        x_max = max(r.bounds["x_max"] for r in self.apartment.rooms)
        y_max = max(r.bounds["y_max"] for r in self.apartment.rooms)
        nx = int(round(x_max / self.h)) + 1
        ny = int(round(y_max / self.h)) + 1

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
                    x_g = i_g * self.h
                    y_g = j_g * self.h

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
                x, y = i * self.h, j * self.h
                if not any(r.contains(x, y) for r in self.apartment.rooms):
                    U_global[j, i] = -10.0  # outside mask value

        return U_global