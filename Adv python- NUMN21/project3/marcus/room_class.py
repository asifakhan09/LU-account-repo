"""A room class: trying to prioritize readability"""

from dataclasses import dataclass

from matplotlib import pyplot as plt
import numpy as np
from rich import print
from scipy import sparse
from scipy.sparse.linalg import spsolve


class BoundaryCondition:
    def __init__(self, value: np.ndarray | float):
        self._value = value

    def get_value(self, idx):
        if isinstance(self._value, np.ndarray):
            return self._value[idx]
        else:
            return self._value

    def update(self, vec: np.ndarray):
        assert isinstance(self._value, np.ndarray), f"can only update array, {self}"
        mask = ~np.isnan(vec)
        try:
            self._value[: len(mask)][mask] = vec[mask]
        except ValueError as e:
            print(f"update failed | {mask.shape=} {vec.shape=} {self._value.shape=}")
            raise e

    def get_value2(self):
        return self._value

    def get_label(self, idx):
        letter = type(self).__name__[0]
        return f"{letter}{self.get_value(idx)}"

    def flipped(self):
        v = self._value
        if isinstance(v, np.ndarray):
            return type(self)(np.flip(v, 0))  # flip the vector
        else:
            return type(self)(v)  # scalar


class Dirichlet(BoundaryCondition):
    def __str__(self) -> str:
        return "D"


class Neumann(BoundaryCondition):
    def __str__(self) -> str:
        return "N"


@dataclass
class BoxBoundary:
    T: BoundaryCondition
    B: BoundaryCondition
    L: BoundaryCondition
    R: BoundaryCondition

    def flipped_v(self):
        """Swap T and B, also flip the vectors in L and R if applicable"""
        return BoxBoundary(
            T=self.B,
            B=self.T,
            L=self.L.flipped(),
            R=self.R.flipped(),
        )

    def flipped_h(self):
        """Swap L and R, also flip the vectors in T and B if applicable"""
        return BoxBoundary(
            T=self.T.flipped(),
            B=self.B.flipped(),
            L=self.R,
            R=self.L,
        )

    def rotated_90(self):
        """Swap BCs:
        T->L (flip!),
        L->B,
        B->R (flip!),
        R->T"""
        return BoxBoundary(
            T=self.R,
            B=self.L,
            L=self.T.flipped(),
            R=self.B.flipped(),
        )

    def transposed(self):
        """Swap BCs: T->L, L->T, B->R, R->T"""
        return BoxBoundary(
            T=self.L,
            B=self.R,
            L=self.T,
            R=self.B,
        )


@dataclass
class Room:
    height: float
    width: float
    resolution: int
    ghost_mode: bool = False

    def show_grid(self, bc: BoxBoundary, cell_w=3):
        """Print linear indices and BC in a grid"""
        n_row, n_col = self.n_grid
        print("-" * (n_col * cell_w // 2) + " ROOM " + "-" * (n_col * cell_w // 2))

        print(
            " " * cell_w
            + "".join(f" {bc.T.get_label(c).ljust(cell_w)}" for c in range(n_col))
        )
        for r in range(n_row):
            # row = " ".join("*".ljust(cell_w) for _ in range(w))
            row = " ".join(str(self.lin_idx(r, c)).ljust(cell_w) for c in range(n_col))
            print(
                f"{bc.L.get_label(r).ljust(cell_w)} {row} {bc.R.get_label(r).ljust(cell_w)}"
            )
        print(
            " " * cell_w
            + "".join(f" {bc.B.get_label(c).ljust(cell_w)}" for c in range(n_col))
        )
        print(f"points: {self.n_points}, grid: {self.n_grid}")

    def lin_idx(self, r: int, c: int) -> int:
        """Compute 'linear' index from row and column indices"""
        return int(r * self.width * self.resolution + c)

    def grid_idx(self, x: float, y: float) -> tuple[int, int]:
        """Compute 'grid' indices from coordinates"""
        return int(x * self.resolution), int(y * self.resolution)

    @property
    def n_grid(self):
        """number or rows and columns"""
        return int(self.height * self.resolution), int(self.width * self.resolution)

    @property
    def n_points(self) -> int:
        """How many internal points"""
        return int((self.width * self.height) * self.resolution**2)

    @property
    def h(self):
        """Grid step size"""
        return 1 / self.resolution

    def grid_iter(self):
        """Iterate over internal points in row-first-order"""
        n_rows, n_col = self.n_grid

        for r in range(n_rows):
            for c in range(n_col):
                i = self.lin_idx(r, c)
                yield (i, r, c)

    def neighbors(self, r: int, c: int):
        """Return 4-neighbor grid-idx (up, down, left, right)."""

        return [
            (r - 1, c),  # above
            (r + 1, c),  # below
            (r, c - 1),  # left
            (r, c + 1),  # right
        ]

    def build_sys(self, bc: BoxBoundary) -> tuple[sparse.csr_array, np.ndarray]:
        """Build the matrix A and the RHS vector b."""
        n_interior = self.n_points
        Ah2 = sparse.lil_matrix((n_interior, n_interior))
        bh2 = np.zeros(n_interior)

        n_rows, n_cols = self.n_grid

        def get_bc(r: int, c: int):
            """Get the boundary condition (coordinate outside grid), else None

            Parameters
            ----------

            r, c: int
                row/col index, should be a point in grid or one step outside,
                but not diagonally outside corner

            Returns
            -------
            bc_here: BoundaryCondition
                The boundary condition for this point
            bc_vec_idx: int
                Which index of the bc-vector to take
            opposite_neigh_idx: int
                Linear index of the point one step in from the edge. (for ghost-mode)
            """
            if r == -1:
                assert c >= 0, "should not be out on 2 axes"
                return bc.T, c, self.lin_idx(1, c)
            if r == n_rows:
                assert c < n_cols, "should not be out on 2 axes"
                return bc.B, c, self.lin_idx(n_rows - 2, c)
            if c == -1:
                assert r >= 0, "should not be out on 2 axes"
                return bc.L, r, self.lin_idx(r, 1)
            if c == n_cols:
                assert r < n_rows, "should not be out on 2 axes"
                return bc.R, r, self.lin_idx(r, n_cols - 2)

            return None, 0, 0

        def incrementA(i1: int, i2: int, v: float):
            """Basically does `Ah2[i1, i2]+=v`."""
            cur = Ah2[i1, i2]
            assert isinstance(cur, float)
            Ah2[i1, i2] = cur + v

        for i, r, c in self.grid_iter():
            # diagonal default
            incrementA(i, i, -4)

            # check each neighbor
            for nr, nc in self.neighbors(r, c):
                # what bc?
                bc_here, bc_vec_idx, opp = get_bc(nr, nc)

                if bc_here is None:
                    # neighbor in grid
                    ni = self.lin_idx(nr, nc)
                    incrementA(i, ni, 1)

                elif isinstance(bc_here, Dirichlet):
                    bh2[i] -= bc_here.get_value(bc_vec_idx)
                elif isinstance(bc_here, Neumann):
                    if self.ghost_mode:
                        # using two-sided approx
                        # -> have 2 on the "outside" and the "opposite neigh"
                        assert opp != i, "Didnt expect this.."
                        incrementA(i, opp, 1)

                        bh2[i] -= 2 * self.h * bc_here.get_value(bc_vec_idx)
                    else:
                        # using one-sided approx
                        # -> have -3 on diagonal
                        # NOTE: Ah2[i, i] = -3 seems reasonable,
                        # but maybe we need to increment twice for corners...
                        # will not happen in apartment, but using +=1 is (probably) safer
                        incrementA(i, i, 1)
                        bh2[i] -= self.h * bc_here.get_value(bc_vec_idx)

        h2 = self.h**2
        return Ah2.tocsr() / h2, bh2 / h2

    def update_sys_or_maybe_just_b(self, todo="TODO"):
        """Maybe we can update just the changed parts during iteration, should be faster"""

    def fast_b(self, bc: BoxBoundary):
        """build the RHS vector b.

        TODO: can probably be even faster
        """
        n_interior = self.n_points
        bh2 = np.zeros(n_interior)

        n_rows, n_cols = self.n_grid

        def get_bc(r: int, c: int):
            if r == -1:
                assert c >= 0, "should not be out on 2 axes"
                return bc.T, c, self.lin_idx(1, c)
            if r == n_rows:
                assert c < n_cols, "should not be out on 2 axes"
                return bc.B, c, self.lin_idx(n_rows - 2, c)
            if c == -1:
                assert r >= 0, "should not be out on 2 axes"
                return bc.L, r, self.lin_idx(r, 1)
            if c == n_cols:
                assert r < n_rows, "should not be out on 2 axes"
                return bc.R, r, self.lin_idx(r, n_cols - 2)

            return None, 0, 0

        for i, r, c in self.grid_iter():
            if 0 < r < n_rows - 1 and 0 < c < n_cols - 1:
                continue

            # check each neighbor
            for nr, nc in self.neighbors(r, c):
                # what bc?
                bc_here, bc_vec_idx, _ = get_bc(nr, nc)

                if isinstance(bc_here, Dirichlet):
                    bh2[i] -= bc_here.get_value(bc_vec_idx)
                elif isinstance(bc_here, Neumann):
                    if self.ghost_mode:
                        bh2[i] -= 2 * self.h * bc_here.get_value(bc_vec_idx)
                    else:
                        bh2[i] -= self.h * bc_here.get_value(bc_vec_idx)

        h2 = self.h**2
        return bh2 / h2

    def solve(self, bc):
        A, b = self.build_sys(bc)
        u = spsolve(A, b)
        assert isinstance(u, np.ndarray)
        n_rows, n_cols = self.n_grid
        return u.reshape((n_rows, n_cols))


def plot_solution(u: np.ndarray, title: str):
    plt.figure()
    plt.imshow(u, cmap="hot")
    plt.colorbar(label=r"Temp ($\mathbf{^{\circ}C}$)")
    plt.title(title)
    plt.show(block=False)


# resolution
RES = 30

EXAMPLES = {
    "half-warm": BoxBoundary(
        T=Neumann(0),
        B=Neumann(0),
        L=Dirichlet((np.arange(RES) > RES // 2) * 10),
        R=Neumann(-5),
    ),
    "linear": BoxBoundary(
        T=Neumann(0),
        B=Neumann(0),
        L=Dirichlet(1),
        R=Dirichlet(5),
    ),
    "corner": BoxBoundary(
        T=Neumann(5),
        B=Dirichlet(-2.1),
        L=Neumann(4),
        R=Dirichlet(8),
    ),
}

if __name__ == "__main__":
    np.set_printoptions(precision=3)

    bc = EXAMPLES["corner"]

    u = Room(1, 1, resolution=RES).solve(bc)
    u_ghost = Room(1, 1, resolution=RES, ghost_mode=True).solve(bc)
    if RES <= 10:
        print("u")
        print(u)
        print("u_ghost")
        print(u_ghost)

    plot_solution(u, "Temp")
    plot_solution(u_ghost, "Temp_ghost")
    # ==== TASK 1 ===
    print("\n\n=== Task 1 ===")
    bc = BoxBoundary(
        T=Dirichlet(40),
        B=Dirichlet(40),
        L=Dirichlet(40),
        R=Neumann(0),
    )
    room = Room(1, 1, resolution=3, ghost_mode=True)
    room.show_grid(bc)
    A, b = room.build_sys(bc)
    print(A.toarray() / 9)
    print(b / 9)

    input("quit?")
