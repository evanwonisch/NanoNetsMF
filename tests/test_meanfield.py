from module.simulation.meanfield import MeanField as MeanField
from module.base.network import Network as Network
import pytest
import numpy as np

class TestMeanfield:

    def test_instance(self):
        """
        Tests if the initial class members are calculated correctly
        """
        net = Network(2,2,1,[[0,0,0]])
        mf = MeanField(net)

        
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

    def test_propability(self):
        """
        Tests if probabilities for microstates are calculated correctly and broadcastable.
        """
        net = Network(2,2,1,[[0,0,0],[1,1,0]])
        mf = MeanField(net)
        island_index = np.array([[0,1],[2,3]])
        groundstates = np.array([[False,True],[False,True]])

        macrostates = np.array([[[16.17410522, 18.49276232, 23.45403068, 11.1422785 ],
                        [10.16361653, 28.2023101,   5.33504647, 25.85101559]],

                        [[ 9.14831688, 19.33463189, 21.3374962,  12.9502747 ],
                        [17.87378285, 14.48118671,  6.5975802,  15.22911196]]])
        prob = mf.calc_probability(macrostates, island_index, groundstates)
        sol = np.array([[0.17410522, 1- 0.2023101],[0.3374962, 1 - 0.22911196]])

        assert np.allclose(prob, sol), "probabilities for microstates wrongly calculated"

    def test_effective_operator(self):
        """
        This tests the effective operator and cheks that is broadcastable accordingly
        """
        net = Network(2,2,1,[[0,0,0],[1,1,0]])
        mf = MeanField(net)
        island_index = np.array([[0,1],[2,3]])
        groundstates = np.array([[False,True],[False,True]])

        occupation_numbers = np.array([[[16.17410522, 18.49276232, 23.45403068, 11.1422785 ],
                                        [10.16361653, 28.2023101,   5.33504647, 25.85101559]],

                                        [[ 9.14831688, 19.33463189, 21.3374962,  12.9502747 ],
                                        [17.87378285, 14.48118671,  6.5975802,  15.22911196]]])
        res = mf.effective_operator(occupation_numbers, island_index, groundstates)

        sol = np.array([[[17, 18.49276232, 23.45403068, 11.1422785 ],
                                        [10.16361653, 28,   5.33504647, 25.85101559]],

                                        [[ 9.14831688, 19.33463189, 22,  12.9502747 ],
                                        [17.87378285, 14.48118671,  6.5975802,  15]]])
        
        assert np.allclose(res, sol), "effective operator fails"

    def test_effective_states(self):
        net = Network(2,2,1,[[0,0,0]])
        mf = MeanField(net)

        macrostate = np.array([0.5, 0.3, 0.5, 0.6])

        effective_states, p = mf.calc_effective_states(macrostate, neighbour_in_ground_state=True, island_in_ground_state = False)

        zs = np.zeros(4)

        sol_state = np.array([
                [zs,[1,0,0.5,0.6],zs,[1,0.3,0,0.6],zs,zs],
                [[0,1,0.5,0.6],zs,zs,[0.5,1,0.5,0],zs,zs],
                [zs,[0.5,0.3,1,0],[0,0.3,1,0.6],zs,zs,zs],
                [[0.5,0.3,0,1],zs,[0.5,0,0.5,1],zs,zs,zs]
        ])

        sol_p = np.array([
            [0, 0.7 * 0.5, 0, 0.5 * 0.5, 0, 0],
            [0.5 * 0.3, 0, 0, 0.4 * 0.3, 0, 0],
            [0, 0.5 * 0.4, 0.5 * 0.5, 0, 0, 0],
            [0.5 * 0.6, 0, 0.7 * 0.6, 0, 0, 0]
        ])

        assert np.allclose(effective_states, sol_state), "wrong effective state calculation"
        assert np.allclose(p, sol_p), "wrong probability calculation"

        # check for wrong inputs
        with pytest.raises(AssertionError):
            mf.calc_effective_states(np.zeros(5), True, False)

        with pytest.raises(AssertionError):
            mf.calc_effective_states(np.zeros((2,4)), True, False)

        with pytest.raises(AssertionError):
            mf.calc_effective_states(np.zeros(4), True, "test")