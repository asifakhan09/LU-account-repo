import numpy as np
import pytest

import marcus_nn_classes as nn


@pytest.mark.parametrize(
    "x,y,corr",
    [
        ([1], [1], 0),
        ([1], [0], 0.5),
    ],
)
def test_mse_fw(x, y, corr):
    f = nn.MSELoss()
    loss = f.forward(np.array(x), np.array(y))
    assert loss == pytest.approx(corr)
