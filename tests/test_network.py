import numpy as np
import pytest

from module.base.network import Network
import module.components.CONST as CONST

class TestNetwork:

    def relative_error(self, value, target):
        return np.abs(value - target) / target

    def test_electrostatics(self):
        """
        Test if applying voltages and induced charges work right.
        """
        net = Network(2,2,1,[[0,0,0],[1,1,0]])
        net.set_voltage_config([1,2], 0)

        with pytest.raises(AssertionError):
            net.set_voltage_config([1],0)

        assert np.allclose(net.dq, np.array([1.37756674, 0.       ,  0.        , 2 * 1.37756674        ])), "wrong induced charge vector"

        assert net.inv_cap_mat.shape == (4,4), "wrong shape of capacity matrix"

        assert net.electrode_voltages[0] == 1, "wrong volatge configuration"
        assert net.electrode_voltages[1] == 2, "wrong voltage configuration"
        assert net.gate_voltage == 0, "wrong gate voltage"

    def test_indices(self):
        """
        Checks if index conversions and neighbours work right.
        """
        net = Network(3,3,3,[])

        # convert to cartesian
        linear_indices = np.arange(29)
        cartesian_indices = net.get_cartesian_indices(linear_indices)
        sol = np.array([[ 0,  0,  0],
                [ 1,  0,  0],
                [ 2,  0,  0],
                [ 0,  1,  0],
                [ 1,  1,  0],
                [ 2,  1,  0],
                [ 0,  2,  0],
                [ 1,  2,  0],
                [ 2,  2,  0],
                [ 0,  0,  1],
                [ 1,  0,  1],
                [ 2,  0,  1],
                [ 0,  1,  1],
                [ 1,  1,  1],
                [ 2,  1,  1],
                [ 0,  2,  1],
                [ 1,  2,  1],
                [ 2,  2,  1],
                [ 0,  0,  2],
                [ 1,  0,  2],
                [ 2,  0,  2],
                [ 0,  1,  2],
                [ 1,  1,  2],
                [ 2,  1,  2],
                [ 0,  2,  2],
                [ 1,  2,  2],
                [ 2,  2,  2],
                [-1, -1, -1],
                [-1, -1, -1]])
        assert np.allclose(cartesian_indices, sol), "wrong conversion to cartesian indices"

        # convert to linear indices
        linear_indices = net.get_linear_indices(cartesian_indices)
        sol = np.arange(29)
        sol[-1] = -1
        sol[-2] = -1
        assert np.allclose(linear_indices, sol), "wrong conversion to linear indices"

        # next neighbours
        linear_indices = np.arange(4)
        neighbours = net.get_nearest_neighbours(linear_indices)
        sol = np.array([[-1,  1, -1,  3, -1,  9],
                         [ 0,  2, -1,  4, -1, 10],
                         [ 1, -1, -1,  5, -1, 11],
                         [-1,  4,  0,  6, -1, 12]])
        assert np.allclose(neighbours, sol), "wrong nearest-neighbour indices"

    def test_internal_energy(self): 
        """
        Checks if free energy calculation behaves according to saved data
        """
        net = Network(5,5,1,[[0,0,0],[4,4,0]])
        net.set_voltage_config([0.1,-0.1],0)

        n = np.loadtxt("tests/testdata/internal_energy/n.csv")
        U_target = np.loadtxt("tests/testdata/internal_energy/U.csv")
        U = net.calc_internal_energy(n)

        assert np.allclose(U_target, U), "free energy calculation deviates from original"

    def test_energy_broadcasting(self):
        """
        Checks that energy calculations broadcast accordingly
        """
        N = 4
        net = Network(2,2,1,[])

        # with broadcasting
        occupations = np.array([[[1,0,0,0],[0,0,0,0]],[[0,0,0,0],[1,1,1,1]]])
        assert net.calc_internal_energy(occupations).shape == (2,2), "invalid shape broadcasting"

        # without broadcasting
        occupations = np.array([1,0,0,0])
        assert net.calc_internal_energy(occupations).shape == (), "invalid shape broadcasting"

    def test_internal_energy2(self):
        """
        Checks if free energy is equal for equivalent situations
        """
        net = Network(2,2,1, [])
        net.set_voltage_config([], 0)

        n = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        U = net.calc_internal_energy(n)

        assert U.shape == (4,), "wrong shape of free energies"
        assert self.relative_error(U[0], U[1]) < 1e-15, "energies should be equal"
        assert self.relative_error(U[0], U[2]) < 1e-15, "energies should be equal"
        assert self.relative_error(U[0], U[3]) < 1e-15, "energies should be equal"

        
    def test_rates_internal(self):
        """
        Cheks if rate calculation is according to saved data
        """
        net = Network(2,2,1,[[0,0,0]])
        net.set_voltage_config([1], 0)

        assert self.relative_error(1/CONST.electron_charge**2/CONST.tunnel_resistance, net.calc_rate_internal(-1)) < 1e-20, "wrong rate in linear regime"

        assert self.relative_error(net.calc_rate_internal(0), CONST.kb * CONST.temperature / CONST.electron_charge**2 / CONST.tunnel_resistance) < 1e-20, "wrong rate for dF = 0"

        # check other energies
        dF, sol = np.loadtxt("tests/testdata/rates/rates.csv", unpack = True)
        rates = net.calc_rate_internal(dF)

        assert np.allclose(rates, sol), "wrong rate calculations"

    def test_rates_islands(self):
        """
        Checks that tunnel rates are the same for equivalent jumps
        """
        net = Network(2,2,1, [])
        net.set_voltage_config([], 0)

        n = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[1,0,0,0]])
        alpha = np.array([0,1,2,0])
        beta = np.array([1,3,3,2])

        rates = net.calc_rate_island(n, alpha, beta)

        assert rates.shape == (4,), "invalid shape of rates"
        assert self.relative_error(rates[0], rates[1]) < 1e-22, "rates should be equal"
        assert self.relative_error(rates[0], rates[2]) < 1e-22, "rates should be equal"
        assert self.relative_error(rates[0], rates[3]) < 1e-22, "rates should be equal"

    def test_rates_electrode(self):
        """
        Cheks that rates for electordes behave accordingly.
        """
        net = Network(2,2,1,[[0,0,0]])
        net.set_voltage_config([-0.1], 0)

        n = np.random.randn(3,3,4)
        rates1 = net.calc_rate_to_electrode(n, 0)
        assert rates1.shape == (3,3), "wrong shape of rates to electrode"
        rates2 = net.calc_rate_from_electrode(n ,0)
        assert rates2.shape == (3,3), "wrong shape of rates from electorde"

        with pytest.raises(AssertionError):
            _ = net.calc_rate_to_electrode(n, 1)
            _ = net.calc_rate_from_electrode(n, 1)

    def test_rates_comparison(self):
        """
        Checks if the rates equal to the KMC-model calculations.
        """

        net = Network(2,2,1,[[0,0,0], [1,1,0]])
        net.set_voltage_config([0.1, 0.0], 0.0)
        state = np.zeros(4)

        # rates from KMC model

        # input electrode
        assert self.relative_error(net.calc_rate_from_electrode(state, 0), 18.08179571417591) < 1e-8
        assert net.calc_rate_to_electrode(state, 0) == pytest.approx(0)

        # output electrode
        assert net.calc_rate_from_electrode(state, 1) == pytest.approx(0)
        assert net.calc_rate_to_electrode(state, 1) == pytest.approx(0)

        # inter-particle-rates
        assert self.relative_error(net.calc_rate_island(state, 0, 1), 1.056664437782643e-76) < 1e-4
        assert self.relative_error(net.calc_rate_island(state, 0, 2), 1.056664437782643e-76) < 1e-4

    def test_rates_comparison2(self):
        """
        Checks if the rates equal to the KMC-model calculations.
        """
        net = Network(2,2,1,[[0,0,0], [1,1,0]])
        net.set_voltage_config([0.01, 0.0], 0.1)
        state = np.zeros(4)

        # rates from KMC model

        # input electrodes
        assert net.calc_rate_from_electrode(state, 0) == pytest.approx(0)
        assert self.relative_error(net.calc_rate_to_electrode(state, 0), 15.727086322720085) < 1e-7

        # output electrodes
        assert net.calc_rate_from_electrode(state, 1) == pytest.approx(0)
        assert self.relative_error(net.calc_rate_to_electrode(state, 1), 17.81773816105172) < 1e-8