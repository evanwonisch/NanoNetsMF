import numpy as np

from module.base.network import Network
from module.simulation.masterequation import MasterEquation
import module.components.CONST as CONST
from module.components.discrete_gaussian1D import DiscreteGaussian1D

class SetMeanField2:
    """
    Implements a full second order meanfield approximation for solving the single electron transistor.
    """

    def __init__(self, input_voltage, gate_voltage, phase_space_min = -20, phase_space_max = 20):
        """
        The output voltage is held fixed to zero.

        Parameters:
            input_voltage   : voltage of input electrode
            gate_voltage    : voltage of gate  
        """
        self.net = Network(1, 1, 1, [[0,0,0], [0,0,0]])
        self.net.set_voltage_config([input_voltage, 0], gate_voltage)
        self.gaussian = DiscreteGaussian1D(phase_space_min, phase_space_max)
        self.phase_space = self.gaussian.phase_space

    def get_rate_to_island(self):
        """
        Calculates the total rate towards the island for the entire phase space snippet.
        """
        space = np.expand_dims(self.phase_space, axis = -1)
        return self.net.calc_rate_from_electrode(space, 0) + self.net.calc_rate_from_electrode(space, 1)
    
    def get_rate_from_island(self):
        """
        Calculates the total rate towards the electrodes for the entire phase space snippet.
        """
        space = np.expand_dims(self.phase_space, axis = -1)
        return self.net.calc_rate_to_electrode(space, 0) + self.net.calc_rate_to_electrode(space, 1)

    def I(self):
        """
        Calculates the total current flowing towards the particle for the entire phase space snippet.
        """
        return self.get_rate_to_island() - self.get_rate_from_island()

    def I_dag(self):
        """
        Calculates the adjoint current for the entire phase space snippet.
        """
        return self.get_rate_to_island() + self.get_rate_from_island()

    def dN1(self):
        """
        Phase space function calculating the derivative of the first moment.
        """
        return self.I()

    def dN2(self):
        """
        Phase space function calculating the derivative of the second moment.
        """
        return 2 * self.phase_space * self.I() + self.I_dag()
    
    def output_current(self):
        """
        Phase space function calculating the output current for the entire phase space snippet.
        """
        space = np.expand_dims(self.phase_space, axis = -1)
        return -(self.net.calc_rate_from_electrode(space, 1) - self.net.calc_rate_to_electrode(space, 1)) * CONST.electron_charge
    
    def calc_expected_output_current(self, mean, var):
        """
        Calculates the expected output current in nA for a given mean and variance
        """
        probs = self.gaussian.calc_prob(mean, var)
        return np.sum(self.output_current() * probs)
    
    def calc_expected_squared_output_current(self, mean, var):
        """
        Calculates the expected squared output current in (nA)^2 for a given mean and variance
        """
        probs = self.gaussian.calc_prob(mean, var)
        return np.sum(self.output_current() ** 2 * probs)
    
    def convergence_metric(self, mean, var):
        """
        Calculates the maximum rate of change of either the mean or the variance.
        """
        probs = self.gaussian.calc_prob(mean, var)
        dN1 = self.dN1()
        dN2 = self.dN2()
        gauss_dN = np.sum(dN1 * probs)
        gauss_dN2 = np.sum(dN2 * probs)
        gauss_dvar = gauss_dN2 - 2 * mean * gauss_dN

        return max(np.abs(gauss_dN), np.abs(gauss_dvar))

    def solve(self, N = 25, dt = 0.05, verbose = False):
        """
        Solves for the first and second order moment of the distribution function
        for the single-electron-transistor.

        If verbose, the convergence metric is printed.

        Returns:
            mean, variance
        """
        # initial condition
        gauss_mean = 0
        gauss_var = 2

        dN1 = self.dN1()
        dN2 = self.dN2()

        for _ in range(N):
            gauss_probs = self.gaussian.calc_prob(gauss_mean, gauss_var)
            gauss_dN = np.sum(dN1 * gauss_probs)
            gauss_dN2 = np.sum(dN2 * gauss_probs)
            gauss_dvar = gauss_dN2 - 2 * gauss_mean * gauss_dN

            gauss_mean += dt * gauss_dN
            gauss_var += dt * gauss_dvar

        if verbose:
            print("convergence:", self.convergence_metric(gauss_mean, gauss_var))


        ## Recalculate Mean and Variance
        ##
        ## because this gaussian merges into lawrence dist for too low variance,
        ## the input becomes variance independant at some point.
        ##
        ## this might lead to gausian(mean, var) having a lower variance as var..
        ##
        gauss_probs = self.gaussian.calc_prob(gauss_mean, gauss_var)
        mean = np.sum(self.phase_space * gauss_probs)
        var = np.sum(self.phase_space ** 2 * gauss_probs) - mean**2
        return mean, var
