from module.simulation.quick_meanfield2 import QuickMeanField2 as QuickMeanField2
from module.base.network import Network as Network
import pytest
import numpy as np

class TestMeanfield:

    def test_instance(self):
        """
        Tests if the initial class members are calculated correctly
        """
        net = Network(2,2,1,[[0,0,0]])
        mf = QuickMeanField2(net)

        
        # shapes
        assert mf.island_indices.shape == (4,6), "wrong shape of island_indices"
        assert mf.neighbour_indices.shape == (4,6), "wrong shape of neighbour_indices"
        assert mf.neighbour_mask.shape == (4,6), "wrong shape of neighbour_indices"


        # island indices
        assert np.all(mf.island_indices[0,:] == 0), "error in island_indices"
        assert np.all(mf.island_indices[1,:] == 1), "error in island_indices"
        assert np.all(mf.island_indices[2,:] == 2), "error in island_indices"
        assert np.all(mf.island_indices[3,:] == 3), "error in island_indices"

        # neighbour indices
        neighbours = np.array([[-1,  1, -1,  2, -1, -1],
                                 [ 0, -1, -1,  3, -1, -1],
                                 [-1,  3,  0, -1, -1, -1],
                                 [ 2, -1,  1, -1, -1, -1]])
        assert np.allclose(mf.neighbour_indices, neighbours), "error in nearest neighbours"

        # neighbour mask
        mask = np.array([[0,  1, 0,  1, 0, 0],
                        [ 1, 0, 0,  1, 0, 0],
                        [0,  1,  1, 0, 0, 0],
                        [ 1, 0,  1, 0, 0, 0]])
        assert np.allclose(mf.neighbour_mask, mask), "error in neighbour mask"


    def test_effective_operator(self):
        """
        This tests the effective operator and cheks that is broadcastable accordingly
        """
        net = Network(2,2,1,[[0,0,0],[1,1,0]])
        mf = QuickMeanField2(net)
        island_index = np.array([[0,1],[2,3]], dtype = "int")
        effective_states = np.array([[0,1],[2,-1]])

        states = np.array([[[16.17410522, 18.49276232, 23.45403068, 11.1422785 ],
                                        [10.16361653, 28.2023101,   5.33504647, 25.85101559]],

                                        [[ 9.14831688, 19.33463189, 21.3374962,  12.9502747 ],
                                        [17.87378285, 14.48118671,  6.5975802,  15.22911196]]])
        res = mf.effective_operator(states, island_index, effective_states)

        sol = np.array([[[16, 18.49276232, 23.45403068, 11.1422785 ],
                                        [10.16361653, 29,   5.33504647, 25.85101559]],

                                        [[ 9.14831688, 19.33463189, 23,  12.9502747 ],
                                        [17.87378285, 14.48118671,  6.5975802,  14]]])
        
        assert np.allclose(res, sol), "effective operator fails"

    def test_calc_probability(self):
        """
        Tests the calculation of probabilities
        """

        net = Network(2,2,1,[])
        mf = QuickMeanField2(net)


        mf.means = np.array([0.3,1.4,2.5,3.6])
        mf.vars = np.array([0.33,0.44,0.55,0.66])

        island_indices = np.array([[0,1],[2,3]])

        probs = mf.calc_probability(island_indices)

        assert probs.shape == (2,2,4), "wrong shape of probabilities"

        # check if distribution is correct
        mn = -1 * probs[0,0,0] + 0 * probs[0,0,1] + 1 * probs[0,0,2] + 2 * probs[0,0,3]
        var = (-1)**2 * probs[0,0,0] + (0)**2 * probs[0,0,1] + (1)**2 * probs[0,0,2] + (2)**2 * probs[0,0,3] - mn**2
        assert mn == pytest.approx(mf.means[0])
        assert var == pytest.approx(mf.vars[0])

        # check if distribution is correct
        mn = 1 * probs[1,0,0] + 2 * probs[1,0,1] + 3 * probs[1,0,2] + 4 * probs[1,0,3]
        var = (1)**2 * probs[1,0,0] + (2)**2 * probs[1,0,1] + (3)**2 * probs[1,0,2] + (4)**2 * probs[1,0,3] - mn**2
        assert mn == pytest.approx(mf.means[2])
        assert var == pytest.approx(mf.vars[2])

        