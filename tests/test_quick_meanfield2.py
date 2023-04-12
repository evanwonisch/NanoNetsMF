from module.simulation.quick_meanfield2 import QuickMeanField2 as QuickMeanField2
from module.simulation.meanfield import MeanField as MeanField
from module.simulation.set_meanfield2 import SetMeanField2
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

        # repeated island indices
        assert mf.r_island_indices.shape == (4, 6, 4, 4)
        assert np.all(mf.r_island_indices[0, :, :, :] == 0)
        assert np.all(mf.r_island_indices[1, :, :, :] == 1)
        assert np.all(mf.r_island_indices[2, :, :, :] == 2)
        assert np.all(mf.r_island_indices[3, :, :, :] == 3)

        # repeated neighbour indices
        assert mf.r_neighbour_indices.shape == (4, 6, 4, 4)
        assert np.all(mf.r_neighbour_indices[0, 1, :, :] == 1)
        assert np.all(mf.r_neighbour_indices[0, 3, :, :] == 2)
        assert np.all(mf.r_neighbour_indices[2, 1, :, :] == 3)

        # repeated microstates for island
        assert mf.effective_states_islands.shape == (4, 6, 4, 4)
        assert np.all(mf.effective_states_islands[:,:,0,:] == -1)
        assert np.all(mf.effective_states_islands[:,:,1,:] == 0)
        assert np.all(mf.effective_states_islands[:,:,2,:] == 1)
        assert np.all(mf.effective_states_islands[:,:,3,:] == 2)

        # repeated microstates for neighbours
        assert mf.effective_states_neighbours.shape == (4, 6, 4, 4)
        assert np.all(mf.effective_states_neighbours[:,:,:,0] == -1)
        assert np.all(mf.effective_states_neighbours[:,:,:,1] == 0)
        assert np.all(mf.effective_states_neighbours[:,:,:,2] == 1)
        assert np.all(mf.effective_states_neighbours[:,:,:,3] == 2)

    def test_calc_derivatives(self):
        net = Network(2,2,1,[[0,0,0],[0,0,0]])
        mf_0 = MeanField(net)
        mf = QuickMeanField2(net)

        currents, currents_dag, n_currents =  mf.calc_expectation_islands()

        # test shape 
        assert currents.shape == (net.N_particles, 6), "wrong shape"

        # test that no currents come from not-existing neighbours
        assert currents[0,0] == 0
        assert currents[0,2] == 0
        assert currents[0,4] == 0
        assert currents[0,5] == 0

        assert currents[1,1] == 0
        assert currents[1,2] == 0
        assert currents[1,4] == 0
        assert currents[1,5] == 0

        # for var = 0: reverts back to normal lawrence dist
        net.set_voltage_config([-0.01,0.03],0.04)
        mf.vars = np.zeros(4)
        mf.means = np.array([0,1,2.3,4])
        assert np.allclose(mf_0.calc_expected_island_rates(mf.means), mf.calc_expectation_islands()[0]), "quick_meanfield2 and meanfield do not converge into the same for var = 0"
        assert np.allclose(mf_0.calc_total_currents(mf.means), mf.calc_derivatives()[0]), "quick_meanfield2 and meanfield do not converge concerning the total currents"


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
        res, occ_num = mf.effective_operator(states, island_index, effective_states)

        sol = np.array([[[16, 18.49276232, 23.45403068, 11.1422785 ],
                                        [10.16361653, 29,   5.33504647, 25.85101559]],

                                        [[ 9.14831688, 19.33463189, 23,  12.9502747 ],
                                        [17.87378285, 14.48118671,  6.5975802,  14]]])
        
        sol_occ_num = np.array([[16, 29],[23, 14]])
        
        assert np.allclose(res, sol), "effective operator fails"
        assert np.allclose(occ_num, sol_occ_num), "effective operator fails in calculating occupation numbers"

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

        