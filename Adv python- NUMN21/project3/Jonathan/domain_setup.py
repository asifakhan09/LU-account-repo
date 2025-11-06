import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


H = 1.0 / 20 
T_WALL = 15.0
T_HEATER = 40.0
T_WINDOW = 5.0


ROOM_CONFIGS = [
    
    # ROOM 1 — bottom left, [0,1] x [0,1], Neumann on right
    {
        "id": 1,
        "bounds": {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0},
        "interface": "right", 
        "bc": {
            "left": [
                {"y_min": 0.0, "y_max": 1.0, "type": "Heater", "value": T_HEATER}
            ],
            "right": [
                {"y_min": 0.0, "y_max": 1.0, "type": "Interface"}
            ],
            "top": [
                {"x_min": 0.0, "x_max": 1.0, "type": "Wall", "value": T_WALL}
            ],
            "bottom": [
                {"x_min": 0.0, "x_max": 1.0, "type": "Wall", "value": T_WALL}
            ],
        },
    },

    # ROOM 2 — central, [1,2] x [0,2], Dirichlet 
    # left side: interface (0–1) + wall (1–2)
    # right side: wall (0–1) + interface (1–2)
    {
        "id": 2,
        "bounds": {"x_min": 1.0, "x_max": 2.0, "y_min": 0.0, "y_max": 2.0},
        "interface": None,  # Dirichlet
        "bc": {
            "left": [
                {"y_min": 0.0, "y_max": 1.0, "type": "Interface"},
                {"y_min": 1.0, "y_max": 2.0, "type": "Wall", "value": T_WALL},
            ],
            "right": [
                {"y_min": 0.0, "y_max": 1.0, "type": "Wall", "value": T_WALL},
                {"y_min": 1.0, "y_max": 2.0, "type": "Interface"},
            ],
            "top": [
                {"x_min": 1.0, "x_max": 2.0, "type": "Heater", "value": T_HEATER}
            ],
            "bottom": [
                {"x_min": 1.0, "x_max": 2.0, "type": "Window", "value": T_WINDOW}
            ],
        },
    },

    # ROOM 3 — top right, [2,3] x [1,2], Neumann on left

    {
        "id": 3,
        "bounds": {"x_min": 2.0, "x_max": 3.0, "y_min": 1.0, "y_max": 2.0},
        "interface": "left",
        "bc": {
            "left": [
                {"y_min": 1.0, "y_max": 2.0, "type": "Interface"}
            ],
            "right": [
                {"y_min": 1.0, "y_max": 2.0, "type": "Heater", "value": T_HEATER}
            ],
            "top": [
                {"x_min": 2.0, "x_max": 3.0, "type": "Wall", "value": T_WALL}
            ],
            "bottom": [
                {"x_min": 2.0, "x_max": 3.0, "type": "Wall", "value": T_WALL}
            ],
        },
    },

    # ROOM 4 — optional example, [3,4] x [0,1], Neumann on left
    {
        "id": 4,
        "bounds": {"x_min": 2.0, "x_max": 2.5, "y_min": 0.5, "y_max": 1.0},
        "interface": "left",
        "bc": {
            "left": [
                {"y_min": 0.5, "y_max": 1.0, "type": "Interface"}
            ],
            "right": [
                {"y_min": 0.5, "y_max": 1.0, "type": "Wall", "value": T_WALL}
            ],
            "top": [
                {"x_min": 2.0, "x_max": 2.5, "type": "Wall", "value": T_WALL}
            ],
            "bottom": [
                {"x_min": 2.0, "x_max": 2.5, "type": "Heater", "value": T_HEATER}
            ],
        },
    },
]

### Helper functions ###

def get_room_geometry(room_config, H):
    bounds = room_config['bounds']
    Lx_s = bounds['x_max'] - bounds['x_min']
    Ly_s = bounds['y_max'] - bounds['y_min']
    nx_s = int(round(Lx_s / H)) + 1
    ny_s = int(round(Ly_s / H)) + 1
    i_off = int(round(bounds['x_min'] / H))
    j_off = int(round(bounds['y_min'] / H))
    return nx_s, ny_s, i_off, j_off


def get_room_type(room_config):
    if room_config.get('interface') == 'right':
        return 'Neumann_right'
    elif room_config.get('interface') == 'left':
        return 'Neumann_left'
    else:
        return 'Dirichlet'


def get_fixed_bc(x: float, y: float) -> float | None:
    """
    Returns the fixed boundary temperature at (x, y) based on ROOM_CONFIGS.
    - Handles piecewise BC segments for each boundary.
    - Collects matches from all rooms and decides after.
    - Interface-interface corners are treated as interface (None).
    - If no match but on boundary, fallback to T_WALL.
    """

    tol = 1e-12
    matches = []

    for room in ROOM_CONFIGS:
        bx = room["bounds"]

        # Check if the point is within the bounding box of the room
        if not (bx["x_min"] - tol <= x <= bx["x_max"] + tol and
                bx["y_min"] - tol <= y <= bx["y_max"] + tol):
            continue

        # Left boundary 
        if abs(x - bx["x_min"]) < tol:
            for seg in room["bc"]["left"]:
                if seg["y_min"] - tol <= y <= seg["y_max"] + tol:
                    matches.append(seg["type"] if seg["type"] == "Interface" else seg["value"])

        # Right boundary
        if abs(x - bx["x_max"]) < tol:
            for seg in room["bc"]["right"]:
                if seg["y_min"] - tol <= y <= seg["y_max"] + tol:
                    matches.append(seg["type"] if seg["type"] == "Interface" else seg["value"])

        # Bottom boundary
        if abs(y - bx["y_min"]) < tol:
            for seg in room["bc"]["bottom"]:
                if seg["x_min"] - tol <= x <= seg["x_max"] + tol:
                    matches.append(seg["type"] if seg["type"] == "Interface" else seg["value"])

        # Top boundary
        if abs(y - bx["y_max"]) < tol:
            for seg in room["bc"]["top"]:
                if seg["x_min"] - tol <= x <= seg["x_max"] + tol:
                    matches.append(seg["type"] if seg["type"] == "Interface" else seg["value"])


    # Decide based on collected matches 
    if not matches:
        # no match found, check if on boundary and fallback to wall ( ex corners)
        for room in ROOM_CONFIGS:
            bx = room["bounds"]
            if (abs(x - bx["x_min"]) < tol or abs(x - bx["x_max"]) < tol or
                abs(y - bx["y_min"]) < tol or abs(y - bx["y_max"]) < tol):
                return T_WALL
        return None

    # If any fixed BC value exists, return the first one 
    for m in matches:
        if m != "Interface":
            return m

    # All matches are Interface
    return None






# Solver for subdomain

def solve_subdomain(room_config, boundary_values: dict, interface_flux: dict = None):
    nx_s, ny_s, i_off, j_off = get_room_geometry(room_config, H)
    room_type = get_room_type(room_config)

    mapping = {}
    counter = 0

    for j_s in range(1, ny_s - 1):
        if room_type == 'Dirichlet':
            i_start, i_end = 1, nx_s - 1
        elif room_type == 'Neumann_right':
            i_start, i_end = 1, nx_s  # include right interface
        elif room_type == 'Neumann_left':
            i_start, i_end = 0, nx_s - 1  # include left interface
        else:
            raise ValueError("Unknown room type")

        for i_s in range(i_start, i_end):
            mapping[(i_s, j_s)] = counter
            counter += 1

    N_unknowns = counter
    A = lil_matrix((N_unknowns, N_unknowns))
    b = np.zeros(N_unknowns)

    # Loop over unknowns
    for (i_s, j_s), k in mapping.items():
        i_g = i_s + i_off
        j_g = j_s + j_off
        x_g = i_g * H
        y_g = j_g * H

        # Detect Neumann interface points for left/right
        is_neumann_interface = False
        if room_type == 'Neumann_right' and i_s == nx_s - 1:
            is_neumann_interface = True
            interior_neighbor_i_s = i_s - 1
            normal_sign = -1.0
        elif room_type == 'Neumann_left' and i_s == 0:
            is_neumann_interface = True
            interior_neighbor_i_s = i_s + 1
            normal_sign = 1.0

        if is_neumann_interface:
            # Neumann interface equation
            A[k, k] = -3.0
            A[k, mapping[(interior_neighbor_i_s, j_s)]] = 1.0
            g_val = 0.0
            if interface_flux and (i_g, j_g) in interface_flux:
                g_val = interface_flux[(i_g, j_g)]
            b[k] = normal_sign * H * g_val

            # Vertical neighbors
            for (di, dj) in [(0, 1), (0, -1)]:
                i_ns, j_ns = i_s + di, j_s + dj
                i_ng, j_ng = i_g + di, j_g + dj
                neighbor_coord_s = (i_ns, j_ns)
                neighbor_coord_g = (i_ng, j_ng)

                if neighbor_coord_s in mapping:
                    A[k, mapping[neighbor_coord_s]] = 1.0
                elif neighbor_coord_g in boundary_values:
                    b[k] -= boundary_values[neighbor_coord_g]
                else:
                    T_BC = get_fixed_bc(i_ng * H, j_ng * H)
                    if T_BC is not None:
                        b[k] -= T_BC

        else:
            # Interior equation
            A[k, k] = -4.0
            for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                i_ns, j_ns = i_s + di, j_s + dj
                i_ng, j_ng = i_g + di, j_g + dj
                neighbor_coord_s = (i_ns, j_ns)
                neighbor_coord_g = (i_ng, j_ng)

                # If inside subdomain grid
                if 0 <= i_ns < nx_s and 0 <= j_ns < ny_s:
                    if neighbor_coord_s in mapping:
                        A[k, mapping[neighbor_coord_s]] = 1.0
                    elif neighbor_coord_g in boundary_values:
                        b[k] -= boundary_values[neighbor_coord_g]
                    else:
                        T_BC = get_fixed_bc(i_ng * H, j_ng * H)
                        if T_BC is not None:
                            b[k] -= T_BC
                else:
                    T_BC = get_fixed_bc(i_ng * H, j_ng * H)
                    if T_BC is not None:
                        b[k] -= T_BC

    A_csr = csr_matrix(A)
    u_s = spsolve(A_csr, b)
    return u_s, mapping, (nx_s, ny_s), i_off, j_off


# Interface flux calculation

def calculate_dirichlet_flux(u2_s: np.array, map2: dict, nx2: int, ny2: int, interface_values: dict, i_off=0, j_off=0):
    flux_values = {}
    nx_off_1 = int(1.0 / H)
    nx_off_2 = int(2.0 / H)
    ny_off_1 = int(1.0 / H)

    # Left interface x=1
    i_g1 = nx_off_1
    for j_s in range(1, ny2 - 1):
        j_g = j_s + j_off
        if j_g >= ny_off_1:
            continue
        u_interface = interface_values.get((i_g1, j_g))
        if u_interface is None:
            continue
        k_int = map2.get((1, j_s))
        if k_int is not None:
            u_int = u2_s[k_int]
            g = (u_int - u_interface) / H
            flux_values[(i_g1, j_g)] = g

    # Right interface x=2
    i_g2 = nx_off_2
    for j_s in range(1, ny2 - 1):
        j_g = j_s + j_off
        if j_g < ny_off_1:
            continue
        u_interface = interface_values.get((i_g2, j_g))
        if u_interface is None:
            continue
        k_int = map2.get((nx2 - 2, j_s))
        if k_int is not None:
            u_int = u2_s[k_int]
            g = (u_interface - u_int) / H
            flux_values[(i_g2, j_g)] = g

    return flux_values
