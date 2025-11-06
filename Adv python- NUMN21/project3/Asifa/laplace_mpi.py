#!/usr/bin/env python3
import numpy as np
from mpi4py import MPI
from scipy.linalg import solve
import sys

# =========================
# Problem / iteration setup
# =========================
h = 1.0 / 20.0      # use 1.0/3.0 for Task 1 matrices; 1.0/20.0 for Tasks 2–3
omega = 0.8
num_iters = 10

T_NORMAL = 15.0
T_HEATER = 40.0
T_WINDOW = 5.0

# Room sizes per Canvas figure:
#   Ω1: 1×1  (bottom-left, attaches to LOWER HALF of Ω2's left)
#   Ω2: 1×2  (vertical middle)
#   Ω3: 1×1  (top-right, attaches to UPPER HALF of Ω2's right)
ROOMS = {
    0: {"name": "Ω1", "Lx": 1.0, "Ly": 1.0},  # rank 0
    1: {"name": "Ω2", "Lx": 1.0, "Ly": 2.0},  # rank 1
    2: {"name": "Ω3", "Lx": 1.0, "Ly": 1.0},  # rank 2
}

# =========
# MPI setup
# =========
comm = MPI.Comm.Clone(MPI.COMM_WORLD)
rank = comm.Get_rank()
size = comm.Get_size()
if size != 3:
    if rank == 0:
        print("Run with exactly 3 processes: mpirun -np 3 python laplace_mpi.py")
    sys.exit(0)

room = ROOMS[rank]
Lx, Ly = room["Lx"], room["Ly"]

# =========================
# Grid helpers / assembly
# =========================
def grid_info(Lx, Ly, h):
    nx_pts = int(round(Lx / h)) + 1
    ny_pts = int(round(Ly / h)) + 1
    nx_int = nx_pts - 2
    ny_int = ny_pts - 2
    return nx_pts, ny_pts, nx_int, ny_int

nx_pts, ny_pts, nx_int, ny_int = grid_info(Lx, Ly, h)
N = max(nx_int * ny_int, 0)

def idx(i, j, nx_int):
    return j * nx_int + i

def flat_to_grid(u_flat, nx_int, ny_int):
    return u_flat.reshape((ny_int, nx_int))

def assemble_A(nx_int, ny_int, h):
    if nx_int <= 0 or ny_int <= 0:
        return np.zeros((0, 0))
    s = 1.0 / (h*h)
    N = nx_int * ny_int
    A = np.zeros((N, N), dtype=float)
    for j in range(ny_int):
        for i in range(nx_int):
            k = idx(i, j, nx_int)
            A[k, k] = -4.0 * s
            if i-1 >= 0:       A[k, idx(i-1, j, nx_int)] =  1.0 * s
            if i+1 < nx_int:   A[k, idx(i+1, j, nx_int)] =  1.0 * s
            if j-1 >= 0:       A[k, idx(i, j-1, nx_int)] =  1.0 * s
            if j+1 < ny_int:   A[k, idx(i, j+1, nx_int)] =  1.0 * s
    return A

A_base = assemble_A(nx_int, ny_int, h)

def dirichlet_rhs(nx_int, ny_int, h, g_left, g_right, g_bottom, g_top):
    if nx_int <= 0 or ny_int <= 0:
        return np.zeros(0)
    s = 1.0 / (h*h)
    b = np.zeros(nx_int * ny_int, dtype=float)
    for j in range(ny_int):
        for i in range(nx_int):
            k = idx(i, j, nx_int)
            if i == 0:            b[k] += g_left   * s
            if i == nx_int - 1:   b[k] += g_right  * s
            if j == 0:            b[k] += g_bottom * s
            if j == ny_int - 1:   b[k] += g_top    * s
    return b

def apply_neumann_on_side(A, b, nx_int, ny_int, h, side, g_vals):
    """
    Modify A,b for a Neumann side with flux g_vals (array).
    For left/right: len(g_vals)==ny_int; for bottom/top: len(g_vals)==nx_int.
    Diagonal becomes -3/h^2 at those interface nodes, and RHS adds -(g)/h.
    """
    if nx_int <= 0 or ny_int <= 0:
        return A, b
    s = 1.0 / (h*h)
    A = A.copy()
    b = b.copy()
    if side == 'left':
        i = 0
        for j in range(ny_int):
            k = idx(i, j, nx_int)
            A[k, k] = -3.0 * s
            b[k] += -(g_vals[j] / h)
    elif side == 'right':
        i = nx_int - 1
        for j in range(ny_int):
            k = idx(i, j, nx_int)
            A[k, k] = -3.0 * s
            b[k] += -(g_vals[j] / h)
    elif side == 'bottom':
        j = 0
        for i in range(nx_int):
            k = idx(i, j, nx_int)
            A[k, k] = -3.0 * s
            b[k] += -(g_vals[i] / h)
    elif side == 'top':
        j = ny_int - 1
        for i in range(nx_int):
            k = idx(i, j, nx_int)
            A[k, k] = -3.0 * s
            b[k] += -(g_vals[i] / h)
    else:
        raise ValueError("side must be one of {'left','right','bottom','top'}")
    return A, b

# =========================
# Outer-wall temperatures
# =========================
def room_walls(room_rank):
    """
    Match Figure (Canvas):
      - Ω1: left outer wall is heater (ΓH), others normal; right is interface to Ω2 (lower half).
      - Ω2: bottom ΓWF (window), top ΓH (heater); sides are partly external (normal) and partly interfaces.
      - Ω3: right outer wall is heater (ΓH), others normal; left is interface to Ω2 (upper half).
    """
    walls = {"left": T_NORMAL, "right": T_NORMAL, "bottom": T_NORMAL, "top": T_NORMAL}
    if room_rank == 0:         # Ω1
        walls["left"] = T_HEATER
    elif room_rank == 1:       # Ω2
        walls["bottom"] = T_WINDOW
        walls["top"]    = T_HEATER
        # left/right default to normal; interface portions will be overridden via Dirichlet traces
    elif room_rank == 2:       # Ω3
        walls["right"] = T_HEATER
    return walls

walls = room_walls(rank)
b_base = dirichlet_rhs(nx_int, ny_int, h,
                       g_left=walls["left"], g_right=walls["right"],
                       g_bottom=walls["bottom"], g_top=walls["top"])

# =========================
# Interface helpers
# =========================
def interface_trace_from_room_left(u_flat):
    """For Ω1: take right-most interior column (temperatures next to Ω2)."""
    if nx_int <= 0 or ny_int <= 0: return np.zeros(0)
    return flat_to_grid(u_flat, nx_int, ny_int)[:, -1].copy()

def interface_trace_from_room_right(u_flat):
    """For Ω3: take left-most interior column (temperatures next to Ω2)."""
    if nx_int <= 0 or ny_int <= 0: return np.zeros(0)
    return flat_to_grid(u_flat, nx_int, ny_int)[:, 0].copy()

# Ω2 sizes (for slicing) + Ω1,Ω3 vertical sizes (so the halves match)
_, _, nx_int2, ny_int2 = grid_info(ROOMS[1]["Lx"], ROOMS[1]["Ly"], h)  # Ω2
_, _, nx_int1, ny_int1 = grid_info(ROOMS[0]["Lx"], ROOMS[0]["Ly"], h)  # Ω1
_, _, nx_int3, ny_int3 = grid_info(ROOMS[2]["Lx"], ROOMS[2]["Ly"], h)  # Ω3

# HALF slices on Ω2 (lower part for Ω1, upper part for Ω3)
lower_slice = slice(0, ny_int1)                 # rows for interface with Ω1
upper_slice = slice(ny_int2 - ny_int3, ny_int2) # rows for interface with Ω3

def add_interface_dirichlet_on_halves_for_Omega2(b2, d_left_lower, d_right_upper, nx_int2, ny_int2, h):
    """Add Dirichlet contributions from Ω1 (left-lower) and Ω3 (right-upper) to Ω2 RHS."""
    s = 1.0 / (h*h)
    # left boundary of Ω2: ONLY lower_slice uses data from Ω1
    for jj, j in enumerate(range(lower_slice.start, lower_slice.stop)):
        k = idx(0, j, nx_int2)
        b2[k] += d_left_lower[jj] * s
    # right boundary of Ω2: ONLY upper_slice uses data from Ω3
    for jj, j in enumerate(range(upper_slice.start, upper_slice.stop)):
        k = idx(nx_int2 - 1, j, nx_int2)
        b2[k] += d_right_upper[jj] * s
    return b2

# =========================
# Iteration
# =========================
u_old = np.full(N, T_NORMAL) if N > 0 else np.zeros(0)
u_new = u_old.copy()

for it in range(num_iters):
    # ----- Send Dirichlet traces to Ω2 -----
    if rank == 0:
        # Ω1 → Ω2 (lower half of Ω2's LEFT boundary)
        d1 = interface_trace_from_room_left(u_old)  # length ny_int1
        comm.Send([d1, MPI.DOUBLE], dest=1, tag=10)
    if rank == 2:
        # Ω3 → Ω2 (upper half of Ω2's RIGHT boundary)
        d3 = interface_trace_from_room_right(u_old) # length ny_int3
        comm.Send([d3, MPI.DOUBLE], dest=1, tag=11)

    # ----- Ω2 receives, solves, returns flux on the two half-interfaces -----
    if rank == 1:
        d_left_lower  = np.empty(ny_int1, dtype='d')
        d_right_upper = np.empty(ny_int3, dtype='d')
        comm.Recv([d_left_lower,  MPI.DOUBLE], source=0, tag=10)
        comm.Recv([d_right_upper, MPI.DOUBLE], source=2, tag=11)

        # Build RHS for Ω2: base outer walls + half-interface Dirichlet
        b2 = b_base.copy()
        b2 = add_interface_dirichlet_on_halves_for_Omega2(b2, d_left_lower, d_right_upper,
                                                          nx_int, ny_int, h)

        u2_new = solve(A_base, b2, assume_a='sym') if N > 0 else np.zeros(0)

        # Compute fluxes ONLY on the interface rows we couple
        U2 = flat_to_grid(u2_new, nx_int, ny_int) if N > 0 else np.zeros((0,0))

        # left flux to Ω1 on lower_slice: gL = (u2[0] - d_left_lower) / h
        gL_to_O1 = np.array([(U2[j, 0] - d_left_lower[jj]) / h
                             for jj, j in enumerate(range(lower_slice.start, lower_slice.stop))], dtype='d')

        # right flux to Ω3 on upper_slice: gR = (d_right_upper - u2[-1]) / h
        gR_to_O3 = np.array([(d_right_upper[jj] - U2[j, nx_int-1]) / h
                             for jj, j in enumerate(range(upper_slice.start, upper_slice.stop))], dtype='d')

        comm.Send([gL_to_O1, MPI.DOUBLE], dest=0, tag=20)
        comm.Send([gR_to_O3, MPI.DOUBLE], dest=2, tag=21)

    # ----- Ω1 & Ω3 receive flux and solve with Neumann on their interface side -----
    if rank in (0, 2):
        need_len = ny_int1 if rank == 0 else ny_int3
        g = np.empty(need_len, dtype='d')
        tag = 20 if rank == 0 else 21
        comm.Recv([g, MPI.DOUBLE], source=1, tag=tag)
        side = 'right' if rank == 0 else 'left'
        A_mod, b_mod = apply_neumann_on_side(A_base, b_base, nx_int, ny_int, h, side=side, g_vals=g)
        u_side_new = solve(A_mod, b_mod, assume_a='sym') if N > 0 else np.zeros(0)
        u_new = u_side_new.copy()

    if rank == 1:
        u_new = u2_new.copy()

    # Relax
    if N > 0:
        u_new = omega * u_new + (1.0 - omega) * u_old
    u_old = u_new.copy()

# =========================
# Gather & L-shape plotting
# =========================
fields = comm.gather((rank, nx_int, ny_int, u_old), root=0)

if rank == 0:
    # sort by rank: 0=Ω1,1=Ω2,2=Ω3
    fields_sorted = sorted(fields, key=lambda t: t[0])
    nxi1, nyi1 = fields_sorted[0][1], fields_sorted[0][2]   # Ω1
    nxi2, nyi2 = fields_sorted[1][1], fields_sorted[1][2]   # Ω2
    nxi3, nyi3 = fields_sorted[2][1], fields_sorted[2][2]   # Ω3

    U1 = flat_to_grid(fields_sorted[0][3], nxi1, nyi1) if nxi1*nyi1>0 else None
    U2 = flat_to_grid(fields_sorted[1][3], nxi2, nyi2) if nxi2*nyi2>0 else None
    U3 = flat_to_grid(fields_sorted[2][3], nxi3, nyi3) if nxi3*nyi3>0 else None

    # L-shape canvas:
    # width = Ω1 + Ω2 + Ω3 interior columns, height = Ω2 interior rows
    glob_w = nxi1 + nxi2 + nxi3
    glob_h = nyi2
    Uglob = np.full((glob_h, glob_w), T_NORMAL, float)

    # place Ω2 in the middle full height
    Uglob[:, nxi1:nxi1+nxi2] = U2

    # place Ω1 to the LEFT of Ω2, bottom-aligned (lower half)
    Uglob[0:nyi1, 0:nxi1] = U1

    # place Ω3 to the RIGHT of Ω2, top-aligned (upper half)
    Uglob[glob_h-nyi3:glob_h, nxi1+nxi2:nxi1+nxi2+nxi3] = U3

    # Save PNG (no GUI required)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(Uglob, origin='lower', aspect='auto')
        ax.set_title(f"Temperature (interior), h=1/{int(round(1/h))}, ω={omega}, iters={num_iters}")
        ax.set_xlabel("x (interior cols)")
        ax.set_ylabel("y (interior rows)")
        fig.colorbar(im, ax=ax, label="Temperature")
        plt.tight_layout()
        plt.savefig("temperature.png", dpi=150)
        print("Saved: temperature.png")
    except Exception as e:
        print("Plotting failed:", e)

    print("Global interior shape (H,W):", Uglob.shape)
    print("Min/Max temperature:", float(Uglob.min()), float(Uglob.max()))