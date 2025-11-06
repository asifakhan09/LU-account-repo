import gzip
import pickle
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from numpy import typing as npt


@dataclass(slots=True)
class MnistSubSet:
    X: npt.NDArray[np.float32]
    Y: npt.NDArray[np.int64]

    def __str__(self) -> str:
        return " | ".join(
            f"{f.name}: {getattr(self, f.name).shape} {getattr(self, f.name).dtype}"
            for f in fields(self)
        )

    def __len__(self):
        return len(self.X)

    def get_image(self, idx: int):
        """Get a sample from X, reshped to square"""
        return self.X[idx, :].reshape((28, 28))


@dataclass(slots=True)
class MnistDataSet:
    train: MnistSubSet
    val: MnistSubSet
    test: MnistSubSet

    @classmethod
    def fromTuple(
        cls, data: tuple[tuple, ...], maximum: tuple[int, int, int] | None = None
    ):
        """From the loaded pickle data"""
        if maximum is None:
            return cls(*[MnistSubSet(d[0], d[1]) for d in data])
        else:
            return cls(
                *[MnistSubSet(d[0][:m, :], d[1][:m]) for d, m in zip(data, maximum)]
            )

    def __iter__(self):
        return iter([self.train, self.val, self.test])

    def __str__(self) -> str:
        return "MnistDataSet\n" + "\n".join(
            f"  {f.name:<6} -> {getattr(self, f.name)}" for f in fields(self)
        )


def load_mnist(data_root: str, maximum: tuple[int, int, int] | None = None):
    """Load compressed dataset"""
    path = Path(data_root) / "mnist.pkl.gz"
    with gzip.open(path) as f:
        data = pickle.load(f, encoding="bytes")

    return MnistDataSet.fromTuple(data, maximum)
