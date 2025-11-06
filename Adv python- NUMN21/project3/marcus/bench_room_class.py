import sys
import timeit
from typing import Callable

import numpy as np
from rich import print
from scipy.sparse.linalg import spsolve
from numpy.testing import assert_allclose

sys.path.append(".")

from marcus.room_class import EXAMPLES, Room


def measure(title: str, f: Callable):
    ts = np.array(timeit.repeat(f, number=1, repeat=NREP))
    tmean = ts.mean()
    tstd = ts.std()
    print(f"  {title}: {1000 * tmean:4.1f} ms (stddev {1000 * tstd:.1f})")


NREP = 5
RES_TRY = [5, 10, 20, 40]
if __name__ == "__main__":
    bc = EXAMPLES["corner"]

    for res in RES_TRY:
        print(f"\n{res = }")
        room = Room(1, 1, res)

        # a small test that the fast b works!
        A, b = room.build_sys(bc)
        bf = room.fast_b(bc)
        assert_allclose(bf, b)

        measure("build sys ", lambda: room.build_sys(bc))
        measure("fast b    ", lambda: room.fast_b(bc))
        measure("spsolve   ", lambda: spsolve(A, b))
