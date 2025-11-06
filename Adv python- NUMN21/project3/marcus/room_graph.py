import sys
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from mpi4py import MPI
from numpy.typing import NDArray

sys.path.append(".")
from marcus.room_class import BoxBoundary, Dirichlet, Neumann, Room

OPPOS = {
    "T": "B",
    "B": "T",
    "L": "R",
    "R": "L",
}


@dataclass
class Interface:
    side_from: Literal["T", "B", "L", "R"]
    node_to: "RoomNode"
    range_from: tuple[int, int]
    range_to: tuple[int, int]
    mode: Literal["N", "D"]

    @property
    def side_to(self):
        return OPPOS[self.side_from]


@dataclass
class RoomNode:
    label: str
    thread: int
    bc: BoxBoundary
    room: Room
    # current state
    u: NDArray
    # Neighbors:
    send_to: list[Interface] = field(default_factory=list)

    def get_interface_vec(self, u: NDArray, side: Literal["T", "B", "L", "R"]):
        if side == "T":
            return u[0, :]
        elif side == "B":
            return u[-1, :]
        elif side == "L":
            return u[:, 0]
        elif side == "R":
            return u[:, -1]
        else:
            raise ValueError(f"unknown side {side}")

    def get_almost_interface_vec(self, u: NDArray, side: str):
        if side == "T":
            return u[1, :]
        elif side == "B":
            return u[-2, :]
        elif side == "L":
            return u[:, 1]
        elif side == "R":
            return u[:, -2]
        else:
            raise ValueError(f"unknown side {side}")

    def step(self):
        """Solve the compute the room temperature and compute new interface values"""
        u = self.room.solve(self.bc)

        assert np.all(~np.isnan(u)), "solution should not be NaN"
        assert np.all(np.isfinite(u)), "solution should not be inf"

        for inter in self.send_to:
            rf = inter.range_from
            rt = inter.range_to
            ivec = self.get_interface_vec(u, inter.side_from)[rf[0] : rf[1]]

            # padded send
            send_vec = np.full(rt[1], np.nan)
            if inter.mode == "D":
                send_vec[rt[0] :] = ivec
            elif inter.mode == "N":
                ivec2 = self.get_almost_interface_vec(u, inter.side_from)[rf[0] : rf[1]]
                send_vec[rt[0] :] = (ivec2 - ivec) * self.room.resolution
            else:
                raise ValueError("nope")

            inter.node_to.receive(send_vec, inter.side_to)

        return self.relax(u)

    def relax(self, u: NDArray, relax_factor: float = 0.8):
        if self.u is not None:
            normdiff = np.linalg.norm(u - self.u)
            # self.u = u
            self.u = relax_factor * u + (1 - relax_factor) * self.u
        else:
            normdiff = None
            self.u = u
        return normdiff, self.u

    def receive(self, vec: NDArray, side: str):
        """Update a bc"""

        try:
            if side == "T":
                self.bc.T.update(vec)
            elif side == "B":
                self.bc.B.update(vec)
            elif side == "L":
                self.bc.L.update(vec)
            elif side == "R":
                self.bc.R.update(vec)
            else:
                raise ValueError(f"unexpected side: {side}")
        except AssertionError as e:
            print(f"update failed at '{self.label}'->{side}: {e}")

    def step_mpi(self):
        """?"""


class RoomGraph:
    """How can a bunch of rooms communicate?

    - Thinking that sending the "interface values" is probably fine
    - But, maybe we shouldnt send/bcast all values to all nodes
    - So, can we somehow cleanly pass the relevant vector to the each node?
    """

    def __init__(
        self,
        nodes: list[tuple[RoomNode, tuple[int, int]]],
        tot_h: int,
        tot_w: int,
    ) -> None:
        self.nodes = [n for n, _ in nodes]
        self.origins = [c for _, c in nodes]
        self.shape = (tot_h, tot_w)

    def fake_mpi(self, n_iter=10, plot: bool = False):
        """Iterate without MPI."""
        for i in range(n_iter):
            ts = time.time()
            norms = []
            for node in self.nodes:
                nd, u = node.step()
                if nd is not None:
                    norms.append(nd)
            t_iter = time.time() - ts
            if norms:
                nmean = sum(norms) / len(norms)
            else:
                nmean = np.nan
            print(f"{i=:2d} ({t_iter=:.1f} s): {nmean:.2e}")

        U = self.assemble()
        print(f"Done: {U.shape}")
        if plot:
            self.plot(U)

    def assemble(self):
        U = np.full(self.shape, np.nan)
        for n, (r, c) in zip(self.nodes, self.origins):
            h, w = n.room.n_grid

            assert n.u is not None, "needs a solution first"
            U[r : r + h, c : c + w] = n.u

        return U

    def plot(self, U: NDArray):
        Lx, Ly = 3.0, 2.0

        plt.figure(figsize=(9, 6))
        # Create a mask for outside, "hides" all values in U that are -10.0

        plt.imshow(
            U,
            origin="upper",
            extent=(0, Lx, 0, Ly),
            cmap="hot",
            # vmin=T_WIN,
            # vmax=T_HEATER,
        )
        plt.colorbar(label=r"Temperature $(\mathbf{^{\circ}C})$")

        # Add black lines for room boundaries
        plt.axvline(x=1.0, ymin=0.0, ymax=1.0, color="black", linestyle="--")
        plt.axvline(x=2.0, ymin=0.0, ymax=1.0, color="black", linestyle="--")
        plt.axhline(y=1.0, xmin=2.0 / Lx, xmax=3.0 / Lx, color="black", linestyle="--")
        plt.axhline(y=1.0, xmin=0.0, xmax=1.0 / Lx, color="black", linestyle="--")
        # if self.n_rooms == 4:
        #     plt.axhline(
        #         y=0.5,
        #         xmin=2.0 / Lx,
        #         xmax=2.5 / Lx,
        #         color="black",
        #         linestyle="--",
        #     )
        #     plt.axvline(
        #         x=2.5,
        #         ymin=0.5 / Ly,
        #         ymax=1.0 / Ly,
        #         color="black",
        #         linestyle="--",
        #     )

        plt.title("Temperature Distribution in 3-Room Apartment")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def solve_mpi(self, plot: bool = False):
        comm = MPI.Comm.Clone(MPI.COMM_WORLD)

        n_thread = comm.Get_size()
        rank = comm.Get_rank()
        assert n_thread == len(self.nodes), "expects one process per room"

        my_node = self.nodes[rank]
        my_node.solve()


RES = 30
GM = False
# Grid / geometry
H = 1.0 / RES

# Physical properties
T_WALL = 15.0
T_HEATER = 40.0
T_WIN = 5.0

# Solver settings
OMEGA = 0.8
MAX_ITER = 15
TOL = 1e-12


if __name__ == "__main__":
    # Rooms
    om1 = RoomNode(
        "bottom left",
        1,
        bc=BoxBoundary(
            Dirichlet(T_WALL),
            Dirichlet(T_WALL),
            Dirichlet(T_HEATER),
            Neumann(np.zeros(RES)),
        ),
        room=Room(1, 1, RES, GM),
        u=T_WALL * np.ones((RES, RES)),
    )
    om2 = RoomNode(
        "center",
        2,
        bc=BoxBoundary(
            Dirichlet(T_HEATER),
            Dirichlet(T_WIN),
            Dirichlet(np.concatenate([T_WALL * np.ones(RES), 0 * np.ones(RES)])),
            Dirichlet(np.concatenate([0 * np.ones(RES), T_WALL * np.ones(RES)])),
        ),
        room=Room(2, 1, RES, GM),
        u=T_WALL * np.ones((2 * RES, RES)),
    )
    om3 = RoomNode(
        "top right",
        3,
        bc=BoxBoundary(
            Dirichlet(T_WALL),
            Dirichlet(T_WALL),
            Neumann(np.zeros(RES)),
            Dirichlet(T_HEATER),
        ),
        room=Room(1, 1, RES, GM),
        u=T_WALL * np.ones((RES, RES)),
    )
    om4 = RoomNode(
        "small",
        4,
        bc=BoxBoundary(
            Dirichlet(T_WALL),
            Dirichlet(T_HEATER),
            Neumann(np.zeros(RES // 2)),
            Dirichlet(T_WALL),
        ),
        room=Room(0.5, 0.5, RES, GM),
        u=T_WALL * np.ones((RES // 2, RES // 2)),
    )

    all_rooms = [
        (om1, (RES, 0)),
        (om2, (0, RES)),
        (om3, (0, 2 * RES)),
    ]

    # Connections
    om1.send_to = [
        Interface("R", om2, (0, RES), (RES, 2 * RES), "D"),
    ]
    om2.send_to = [
        Interface("L", om1, (RES, 2 * RES), (0, RES), "N"),
        Interface("R", om3, (0, RES), (0, RES), "N"),
    ]
    om3.send_to = [
        Interface("L", om2, (0, RES), (0, RES), "D"),
    ]

    room4 = True
    if room4:
        om2.send_to.append(
            Interface("R", om4, (RES, 3 * RES // 2), (0, RES // 2), "N"),
        )
        om4.send_to.append(
            Interface("L", om2, (0, RES // 2), (RES, 3 * RES // 2), "D"),
        )
        all_rooms.append((om4, (RES, 2 * RES)))

    # rooms and their origin
    graph = RoomGraph(
        all_rooms,
        tot_h=2 * RES,
        tot_w=3 * RES,
    )
    # simulate MPI solution
    graph.fake_mpi(n_iter=MAX_ITER, plot=True)
