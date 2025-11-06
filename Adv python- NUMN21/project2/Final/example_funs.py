"""
It could be interesting to compare our methods on a few more functions.

https://en.wikipedia.org/wiki/Test_functions_for_optimization

"""

import abc
import numpy as np


class ExampleFun:
    def __init__(self, global_min: np.ndarray) -> None:
        assert global_min.ndim == 1
        self.global_min = global_min

    @property
    def ndim(self):
        return self.global_min.size

    @abc.abstractmethod
    def f(self, x: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.f(x)

    def __str__(self) -> str:
        return f"{type(self).__name__} ({self.ndim}d)"


class ExampleFunWithGrad(ExampleFun):
    @abc.abstractmethod
    def g(self, x: np.ndarray) -> np.ndarray:
        pass


class Rosenbrock2d(ExampleFunWithGrad):
    def __init__(self) -> None:
        super().__init__(global_min=np.ones(2))

    def f(self, x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    def g(self, x):
        dfdx = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
        dfdy = 200 * (x[1] - x[0] ** 2)
        return np.array([dfdx, dfdy], dtype=float)


class Sphere(ExampleFunWithGrad):
    """Should be easy!"""

    def __init__(self, ndim: int) -> None:
        super().__init__(global_min=np.zeros(ndim))

    def f(self, x: np.ndarray):
        return (x**2).sum()

    def g(self, x: np.ndarray):
        return 2 * x


class RosenbrockNd(ExampleFun):
    def __init__(self, ndim: int) -> None:
        assert ndim >= 2
        super().__init__(global_min=np.ones(ndim))

    def f(self, x: np.ndarray):
        t1 = np.sum((x[1:] - x[:-1] ** 2) ** 2, dtype=float)
        t2 = np.sum((1 - x[:-1]) ** 2, dtype=float)
        return 100 * t1 + t2


class ThreeHumpCamel(ExampleFun):
    def __init__(self) -> None:
        super().__init__(global_min=np.zeros(2))

    def f(self, x):
        return (
            2 * x[0] ** 2 - 1.05 * x[0] ** 4 + (x[0] ** 6) / 6 + x[0] * x[1] + x[1] ** 2
        )


class Booth(ExampleFun):
    """Seems quite easy?"""

    def __init__(self) -> None:
        super().__init__(global_min=np.array([1, 3]))

    def f(self, x):
        return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


if __name__ == "__main__":
    import sys

    sys.path.append(".")
    import plotly_plot

    funs = [
        Sphere(2),
        Rosenbrock2d(),
        ThreeHumpCamel(),
        Booth(),
    ]
    fig = plotly_plot.many_heat_maps(funs, resolution=100, xr=(-4, 4), log_scale=True)
    fig.show()
