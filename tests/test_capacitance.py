import numpy as np
import pytest
import module.base.capacitance as capacitance

class TestCapacitance:
    
    def test_capacitites(self):
        """
        Tests if capacities are right for a dummy system.
        """
        capacities = capacitance.build_network(2,2,1,[[0,0,0],[1,1,0]])

        cap_mat = np.array(
            [[ 8.47203543, -1.37756674, -1.37756674,  0.        ],
             [-1.37756674,  7.09446869,  0.        , -1.37756674],
             [-1.37756674,  0.        ,  7.09446869, -1.37756674],
             [ 0.        , -1.37756674, -1.37756674,  8.47203543]])

        assert capacities["node"] == pytest.approx(1.3775667360664756), "wrong node capacity"
        assert capacities["lead"] == pytest.approx(1.3775667360664756), "wrong lead capacity"
        assert capacities["self"] == pytest.approx(4.339335218609398), "wrong self capacity"
        assert np.allclose(capacities["cap_mat"], cap_mat), "wrong capacity matrix"

    def test_capacities2(self):
        capacities = capacitance.build_network(2,2,1,[])

        cap_mat = np.array(
            [[ 7.09446869, -1.37756674, -1.37756674,  0.        ],
             [-1.37756674,  7.09446869,  0.        , -1.37756674],
             [-1.37756674,  0.        ,  7.09446869, -1.37756674],
             [ 0.        , -1.37756674, -1.37756674,  7.09446869]])
        assert np.allclose(capacities["cap_mat"], cap_mat), "wrong capacity matrix"
