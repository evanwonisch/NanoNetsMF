import numpy as np
import pytest

from module.base.network import Network
import module.components.CONST as CONST

class TestNetwork:

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

        n = np.loadtxt("tests/data/internal_energy/n.csv")
        U_target = np.loadtxt("tests/data/internal_energy/U.csv")
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
        assert U[0] == pytest.approx(U[1]), "energies should be equal"
        assert U[0] == pytest.approx(U[2]), "energies should be equal"
        assert U[0] == pytest.approx(U[3]), "energies should be equal"

        
    def test_rates_internal(self):
        """
        Cheks if rate calculation is according to saved data
        """
        net = Network(2,2,1,[[0,0,0]])
        net.set_voltage_config([1], 0)

        assert 1/CONST.electron_charge**2/CONST.tunnel_resistance == pytest.approx(net.calc_rate_internal(-1)), "wrong rate in linear regime"

        assert net.calc_rate_internal(0) == pytest.approx(CONST.kb * CONST.temperature / CONST.electron_charge**2 / CONST.tunnel_resistance), "wrong rate for dF = 0"

        # check other energies
        dF, sol = np.loadtxt("tests/data/rates/rates.csv", unpack = True)
        rates = net.calc_rate_internal(dF)

        assert np.allclose(rates, sol), "wrong rate calculations"

    def test_rates_islands(self):
        """
        Checks that tunnel rates are the same for equivalent jumps
        """
        net = Network(2,2,1, [])
        net.set_voltage_config([], 0)

        n = np.array([[2,0,0,0],[0,2,0,0],[0,0,2,0],[2,0,0,0]])
        alpha = np.array([0,1,2,0])
        beta = np.array([1,3,3,2])

        rates = net.calc_rate_island(n, alpha, beta)

        assert rates.shape == (4,), "invalid shape of rates"
        assert rates[0] == pytest.approx(rates[1]), "rates should be equal"
        assert rates[0] == pytest.approx(rates[2]), "rates should be equal"
        assert rates[0] == pytest.approx(rates[3]), "rates should be equal"

    def test_rates_electrode(self):
        """
        Cheks that rates for electordes behave accordingly.
        """
        net = Network(2,2,1,[[0,0,0]])
        net.set_voltage_config([-0.1], 0)

        n = np.array([0,0,0,0])
        electrode_index = 0

        rates_to_electrode = net.calc_rate_to_electrode(n, electrode_index)
        rates_from_electrode = net.calc_rate_from_electrode(n , electrode_index)

        assert rates_to_electrode == pytest.approx(18.07519097)
        assert rates_from_electrode == pytest.approx(3.202710038178024e-86)

        n = np.random.randn(3,3,4)
        rates1 = net.calc_rate_to_electrode(n, 0)
        assert rates1.shape == (3,3), "wrong shape of rates to electrode"
        rates2 = net.calc_rate_from_electrode(n ,0)
        assert rates2.shape == (3,3), "wrong shape of rates from electorde"

        with pytest.raises(AssertionError):
            _ = net.calc_rate_to_electrode(n, 1)
            _ = net.calc_rate_from_electrode(n, 1)