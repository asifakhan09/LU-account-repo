import sys
sys.path.append(".")

import numpy as np
import Final.Final_nn_classes as nn


class TestClass:

    def test_n_param(self):
        layer = nn.FFLayerSimple(1,1,nn.Sigmoid(),10)
        assert layer.n_param == 2