import numpy as np
from module.base.network import Network
import module.components.CONST as CONST

class SetMeanField:
    def __init__(self, input_voltage, gate_voltage):
        self.n = np.array([0])
        self.n2 = np.array([0.1])

        self.net = Network(1,1,1,[[0,0,0], [0,0,0]])
        self.net.set_voltage_config([input_voltage, 0], gate_voltage)

    def I1(self, n):
        """
        Phase-space function for the evaulation of the time derivative of n.
        This represents the current from the input electrode.
        """
        return CONST.electron_charge * (self.net.calc_rate_from_electrode(n, 0) - self.net.calc_rate_to_electrode(n, 0))
    
    def I2(self, n):
        """
        Phase-space function for the evaulation of the time derivative of n.
        This represents the current from the output electrode.
        """
        return CONST.electron_charge * (self.net.calc_rate_from_electrode(n, 1) - self.net.calc_rate_to_electrode(n, 1))
    
    def L(self, n):
        """
        Phase-space function to evaluate the time derivative of the second order expectation of n.
        """

        sq = np.squeeze(n, axis = -1)

        L1 = (2 * sq + 1) * self.net.calc_rate_from_electrode(n, 0) - (2 * sq - 1) * self.net.calc_rate_to_electrode(n, 0)
        L2 = (2 * sq + 1) * self.net.calc_rate_from_electrode(n, 1) - (2 * sq - 1) * self.net.calc_rate_to_electrode(n, 1)

        return L1 + L2
    
    def get_samples(self):
        var = np.clip(self.n2 - self.n ** 2, a_min = 0.01, a_max = None)
        std = np.sqrt(var)
        n = np.random.normal(loc = self.n, scale = std, size = (50000, 1))
        n1 = np.ceil(n)
        n2 = np.floor(n)
        samples = np.concatenate((n1, n2))

        return samples
    
    def calc_dn(self):

        # p_ceil = self.n - np.floor(self.n)
        # dn_ceil = 1 / CONST.electron_charge * (self.I1(np.ceil(self.n)) + self.I2(np.ceil(self.n)))

        # p_floor =  1 - self.n + np.floor(self.n)
        # dn_floor = 1 / CONST.electron_charge * (self.I1(np.floor(self.n)) + self.I2(np.floor(self.n)))

        p_ceil2 = (self.n - np.floor(self.n)) * 0.3
        dn_ceil2 = 1 / CONST.electron_charge * (self.I1(np.ceil(self.n + 1)) + self.I2(np.ceil(self.n + 1)))

        p_ceil = (self.n - np.floor(self.n)) * 0.7
        dn_ceil = 1 / CONST.electron_charge * (self.I1(np.ceil(self.n)) + self.I2(np.ceil(self.n)))

        p_floor =  (1 - self.n + np.floor(self.n)) * 0.7
        dn_floor = 1 / CONST.electron_charge * (self.I1(np.floor(self.n)) + self.I2(np.floor(self.n)))

        p_floor2 =  (1 - self.n + np.floor(self.n)) * 0.3
        dn_floor2 = 1 / CONST.electron_charge * (self.I1(np.floor(self.n - 1)) + self.I2(np.floor(self.n - 1)))

        return p_ceil2 * dn_ceil2 +  p_ceil * dn_ceil + p_floor * dn_floor + p_floor2 * dn_floor2
    
    def calc_dn2(self):

        # p_ceil = self.n - np.floor(self.n)
        # dn2_ceil = self.L(np.ceil(self.n))

        # p_floor =  1 - self.n + np.floor(self.n)
        # dn2_floor = self.L(np.floor(self.n))

        p_ceil2 = (self.n - np.floor(self.n)) * 0.3
        dn2_ceil2 = self.L(np.ceil(self.n + 1))

        p_ceil = (self.n - np.floor(self.n)) * 0.7
        dn2_ceil = self.L(np.ceil(self.n))

        p_floor =  (1 - self.n + np.floor(self.n)) * 0.7
        dn2_floor = self.L(np.floor(self.n))

        p_floor2 =  (1 - self.n + np.floor(self.n)) * 0.3
        dn2_floor2 = self.L(np.floor(self.n - 1))

        return p_ceil2 * dn2_ceil2 + p_ceil * dn2_ceil + p_floor * dn2_floor + p_floor2 * dn2_floor2
    
    def calc_dn_alt(self):
        
        samples = self.get_samples()
        dn = np.mean(self.I1(samples) + self.I2(samples))

        return dn
    
    def calc_dn2_alt(self):

        samples = self.get_samples()
        dn2 = np.mean(self.L(samples))

        return dn2
    
    def evolve(self, N = 10, dt = 0.1):
        for i in range(N):
            dn = self.calc_dn_alt() * dt
            dn2 = self.calc_dn2_alt() * dt

            self.n += dn
            self.n2 += dn2

            self.n2 = np.where(self.n2 < self.n**2, self.n**2 + 0.1, self.n2)

    def convergence_metric(self):
        return np.abs(self.calc_dn_alt()) + np.abs(self.calc_dn2_alt())