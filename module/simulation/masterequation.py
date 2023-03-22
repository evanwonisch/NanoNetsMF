import numpy as np
from module.base.network import Network
import module.components.CONST as CONST

class MasterEquation:
    def __init__(self, input_voltage, gate_voltage):
        self.n = np.array([0])
        self.n2 = np.array([0.1])

        self.net = Network(1,1,1,[[0,0,0], [0,0,0]])
        self.net.set_voltage_config([input_voltage, 0], gate_voltage)

        self.phase_space = np.arange(-20, 20)
        self.N = len(self.phase_space)

        self.A = np.zeros((self.N, self.N))
        self.build_A()

    def get_state(self, index):
        return np.array([self.phase_space[index]])
    
    def get_rate_to_island(self, n):
        return self.net.calc_rate_from_electrode(n, 0) + self.net.calc_rate_from_electrode(n, 1)
    
    def get_rate_from_island(self, n):
        return self.net.calc_rate_to_electrode(n, 0) + self.net.calc_rate_to_electrode(n, 1)
    
    def build_A(self):
        for n in range(self.N):

            if not n == 0:
                self.A[n, n - 1] = self.get_rate_to_island(self.get_state(n) - 1)
                self.A[n, n] += -self.get_rate_from_island(self.get_state(n))

            if not n == self.N - 1: 
                self.A[n, n + 1] = self.get_rate_from_island(self.get_state(n) + 1)
                self.A[n, n] += -self.get_rate_to_island(self.get_state(n))

    def evolve(self, probs, dt = 0.05):
        assert probs.shape == (self.N,)
        return probs + self.A @ probs * dt
    
    def convergence_metric(self, probs):
        return np.max(np.abs(self.A @ probs))
    
    def expected_current(self, probs):
        rates = self.output_rate(np.expand_dims(self.phase_space, axis = -1))
        return np.sum(rates * probs)
        
    def output_rate(self, state):
        return -CONST.electron_charge * (self.net.calc_rate_from_electrode(state, 1) - self.net.calc_rate_to_electrode(state, 1))
    
    def solve(self, probs = None, N = 1500, verbose = False):

        if probs is None:
            probs = np.abs(np.random.randn(self.N))
            probs = probs / np.sum(probs)

        for i in range(N):
            probs = self.evolve(probs, dt = 0.001)

        if verbose:
            print("convergence:", self.convergence_metric(probs))

        return probs