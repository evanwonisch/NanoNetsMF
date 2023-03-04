from module.simulation.meanfield import MeanField as MeanField
from module.base.network import Network as Network
import pytest
import numpy as np

class TestMeanfield:

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