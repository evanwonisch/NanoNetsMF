import numpy as np
from module.base.network import Network
import module.components.CONST as CONST

class MasterEquation:
    """
    Solves the Master Equation for the single electron transistor
    """
    def __init__(self, input_voltage, gate_voltage):
        """
        The output voltage is held fixed to zero.

        Parameters:
            input_voltage   : voltage of input electrode
            gate_voltage    : voltage of gate  
        """
        self.n = np.array([0])
        self.n2 = np.array([0.1])

        self.net = Network(1,1,1,[[0,0,0], [0,0,0]])
        self.net.set_voltage_config([input_voltage, 0], gate_voltage)

        self.phase_space = np.arange(-20, 20)
        self.N = len(self.phase_space)

        self.A = np.zeros((self.N, self.N))
        self.build_A()

    def get_state(self, index):
        """
        Calculates the occupation number at a given index in phase space.
        """
        return np.array([self.phase_space[index]])
    
    def get_rate_to_island(self, n):
        """
        Calculates the total rate towards the island for given occupation number.
        """
        return self.net.calc_rate_from_electrode(n, 0) + self.net.calc_rate_from_electrode(n, 1)
    
    def get_rate_from_island(self, n):
        """
        Calculates the total rate towards the electrodes for given occupation number.
        """
        return self.net.calc_rate_to_electrode(n, 0) + self.net.calc_rate_to_electrode(n, 1)
    
    def build_A(self):
        """
        The linear Master Equation is solved for a snippet of phase space.
        Thus a Matrix characterising the dynamics is defined.
        """
        for n in range(self.N):

            if not n == 0:
                self.A[n, n - 1] = self.get_rate_to_island(self.get_state(n) - 1)
                self.A[n, n] += -self.get_rate_from_island(self.get_state(n))

            if not n == self.N - 1: 
                self.A[n, n + 1] = self.get_rate_from_island(self.get_state(n) + 1)
                self.A[n, n] += -self.get_rate_to_island(self.get_state(n))

    def evolve(self, probs, dt = 0.05):
        """
        Evolves a given prob vector of shape = phase_space.shape along dt in time.
        """
        assert probs.shape == (self.N,)
        return probs + self.A @ probs * dt
    
    def convergence_metric(self, probs):
        """
        Returns the maxmimum rate of change for a probaility as convergence metric.
        """
        return np.max(np.abs(self.A @ probs))
    
    def expected_current(self, probs):
        """
        For given probabilities, calculates the expected output current.
        """
        rates = self.output_current(np.expand_dims(self.phase_space, axis = -1))
        return np.sum(rates * probs)
        
    def output_current(self, state):
        """
        The output current as a function of phase-space.
        """
        return -CONST.electron_charge * (self.net.calc_rate_from_electrode(state, 1) - self.net.calc_rate_to_electrode(state, 1))
    
    def solve(self, probs = None, N = 1500, verbose = False):
        """
        Solves the Master Equation for N iterations. Prints the convergence metric if verbose.
        """
        if probs is None:
            probs = np.abs(np.random.randn(self.N))
            probs = probs / np.sum(probs)

        for i in range(N):
            probs = self.evolve(probs, dt = 0.001)

        if verbose:
            print("convergence:", self.convergence_metric(probs))

        return probs