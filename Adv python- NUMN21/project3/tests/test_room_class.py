import itertools
import sys

import numpy as np
from pytest import approx, mark

sys.path.append(".")

from marcus.room_class import Room, BoxBoundary, Dirichlet, Neumann

# NOTE: all these exampees have scalar values
# in the flip tests, a vector BC would need to flip the vector
# So maybe should refactor to use the flip/rot methods on the class!
EXAMPLE_BCS = [
    (Dirichlet(0.5), Dirichlet(-0.1), Dirichlet(4), Dirichlet(8)),
    (Neumann(0.5), Dirichlet(-2.0), Neumann(4), Dirichlet(2)),
    (Neumann(0.5), Dirichlet(-2.1), Neumann(-40), Neumann(8)),
    (Dirichlet(9), Dirichlet(-2.1), Neumann(-40), Neumann(8)),
]

# Try many combos
EXAMPLE_PROD = list(itertools.product(EXAMPLE_BCS, [True, False]))
# Somewhat readable names...
IDS = [
    f"{''.join(str(b) for b in e[0])}|" + f"{'Ghost' if e[1] else 'Simple'}"
    for e in EXAMPLE_PROD
]
assert len(EXAMPLE_PROD) == len(IDS)

np.set_printoptions(precision=3)


@mark.parametrize("gm", [True, False], ids=["Ghost", "Simple"])
def test_dirichlet_const(gm):
    u = Room(1, 1, resolution=5, ghost_mode=gm).solve(
        bc=BoxBoundary(
            T=Dirichlet(1),
            B=Dirichlet(1),
            L=Dirichlet(1),
            R=Dirichlet(1),
        ),
    )

    assert u == approx(1.0), "expects a constant 1.0"


@mark.parametrize("gm", [True, False], ids=["Ghost", "Simple"])
def test_linear_h(gm):
    """zero vertical flow -> should be linear horizontal gradient"""
    room = Room(3, 10, resolution=1, ghost_mode=gm)
    u = room.solve(
        bc=BoxBoundary(
            T=Neumann(0),
            B=Neumann(0),
            L=Dirichlet(1),
            R=Dirichlet(2),
        ),
    )
    # middle row!
    u = u[len(u) // 2]

    true_linear = np.linspace(1.0, 2.0, num=int(room.width))
    assert u == approx(true_linear, abs=0.5, rel=0.5), "expects a Horizontal gradient"


@mark.parametrize("gm", [True, False], ids=["Ghost", "Simple"])
def test_linear_v(gm):
    """zero horizontal flow -> should be linear vertical gradient"""
    room = Room(10, 3, resolution=1, ghost_mode=gm)
    u = room.solve(
        bc=BoxBoundary(
            T=Neumann(0),
            B=Neumann(0),
            L=Dirichlet(1),
            R=Dirichlet(2),
        ),
    )
    # middle col!
    u = u[:, u.shape[1] // 2]

    true_linear = np.linspace(1, 2, num=int(room.height))
    assert u == approx(true_linear, abs=0.5, rel=0.5), "expects a Vertical gradient"


@mark.parametrize("gm", [True, False], ids=["Ghost", "Simple"])
def test_no_flux(gm):
    u = Room(1, 1, resolution=5, ghost_mode=gm).solve(
        bc=BoxBoundary(
            T=Neumann(0),
            B=Neumann(0),
            L=Neumann(0),
            R=Neumann(0),
        ),
    )

    assert u == approx(u.mean()), "Any constant value is good"


@mark.parametrize("gm", [True, False], ids=["Ghost", "Simple"])
def test_dirichlet_symmetric(gm):
    u = Room(1, 1, resolution=5, ghost_mode=gm).solve(
        bc=BoxBoundary(
            T=Dirichlet(1),
            B=Dirichlet(1),
            L=Dirichlet(0),
            R=Dirichlet(3),
        ),
    )

    print(u)
    assert np.allclose(u[0], u[-1]), "first and last row should be the same"


@mark.parametrize("bcs,gm", EXAMPLE_PROD, ids=IDS)
def test_residual(bcs, gm):
    """use some random BC combo and make sure solution is stable"""
    room = Room(1, 1, resolution=5, ghost_mode=gm)

    A, b = room.build_sys(
        bc=BoxBoundary(*bcs),
    )
    u = room.solve(
        bc=BoxBoundary(*bcs),
    )
    r = A @ u.flatten() - b
    assert r == approx(0), "residual Au-b should be approx zero"


@mark.parametrize("bcs,gm", EXAMPLE_PROD, ids=IDS)
def test_transpose(bcs, gm):
    res = 5
    room = Room(1, 1, resolution=res, ghost_mode=gm)
    bc = BoxBoundary(*bcs)

    u = room.solve(bc)
    u_t = room.solve(bc.transposed())

    assert u_t == approx(u.T), "transposed BC should give transposed solution"


@mark.parametrize("bcs,gm", EXAMPLE_PROD, ids=IDS)
def test_flip_v(bcs, gm):
    res = 6
    room = Room(1, 1, resolution=res, ghost_mode=gm)
    u = room.solve(bc=BoxBoundary(*bcs))

    u_flip_v = room.solve(bc=BoxBoundary(bcs[1], bcs[0], bcs[2], bcs[3]))

    assert u_flip_v == approx(np.flip(u, 0)), (
        "switch T <-> B should give vertically flipped solution"
    )


@mark.parametrize("bcs,gm", EXAMPLE_PROD, ids=IDS)
def test_flip_h(bcs, gm):
    res = 6
    room = Room(1, 1, resolution=res, ghost_mode=gm)
    u = room.solve(bc=BoxBoundary(*bcs))

    u_flip_h = room.solve(bc=BoxBoundary(bcs[0], bcs[1], bcs[3], bcs[2]))

    assert u_flip_h == approx(np.flip(u, 1)), (
        "switch L <-> R should give horizontally flipped solution"
    )


@mark.parametrize("bcs,gm", EXAMPLE_PROD, ids=IDS)
def test_rot(bcs, gm):
    res = 6
    room = Room(1, 1, resolution=res, ghost_mode=gm)
    u = room.solve(bc=BoxBoundary(*bcs))

    u_90 = room.solve(bc=BoxBoundary(bcs[3], bcs[2], bcs[0], bcs[1]))
    u_180 = room.solve(bc=BoxBoundary(bcs[1], bcs[0], bcs[3], bcs[2]))

    assert np.allclose(np.rot90(u), u_90)
    assert np.allclose(np.rot90(u, k=2), u_180)


@mark.skip()
def test_more_neumann():
    """What to expect..."""
    assert False


@mark.parametrize("bcs,gm", EXAMPLE_PROD, ids=IDS)
def test_fastb(bcs, gm):
    res = 6
    room = Room(1, 1, resolution=res, ghost_mode=gm)
    _, b = room.build_sys(bc=BoxBoundary(*bcs))
    bf = room.fast_b(bc=BoxBoundary(*bcs))

    assert bf == approx(b)
