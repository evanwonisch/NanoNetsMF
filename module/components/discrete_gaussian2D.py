import numpy as np
from module.components.lawrence_dist import LawrenceDist
from module.components.discrete_gaussian1D import DiscreteGaussian1D

class DiscreteGaussian2D:
    """
    To calculate discrete gaussian probabilities in two dimensions with covariance.
    """

    def __init__(self, phase_space_bounds_n = (-20, 20), phase_space_bounds_m = (-20, 20)):
        """
        Instatiates a discrete Gaussian distribution generator.

        Parameters:
            phase_space_min: integer of lowest accounted state
            phase_space_max: integer of highest accounted state
        """
        self.phase_space_bounds_n = phase_space_bounds_n
        self.phase_space_bounds_m = phase_space_bounds_m

        self.phase_space = np.stack(np.meshgrid(np.arange(phase_space_bounds_n[0], phase_space_bounds_n[1]),
                                                np.arange(phase_space_bounds_m[0], phase_space_bounds_m[1])), axis = -1)

    def calc_2d_lawrence(self, mn, mm , cov):
        probs = np.zeros((self.phase_space.shape[0], self.phase_space.shape[1]))

        d1 = mn - np.floor(mn)
        d2 = mm - np.floor(mm)

        mincov = -d1*d2 + max(0, d1 + d2 - 1)
        maxcov = -d1*d2 + min(d1 , d2)

        cov = np.clip(cov, mincov, maxcov)

        pcc = cov + d1*d2
        pfc = d2 - pcc
        pcf = d1 - pcc
        pff = 1 - d2 - d1 + pcc

        low_n = np.where(self.phase_space[:,:,0] == np.floor(mn))[1][0]
        low_m = np.where(self.phase_space[:,:,1] == np.floor(mm))[0][0]
        probs[low_m, low_n] = pff
        probs[low_m + 1, low_n] = pfc
        probs[low_m, low_n + 1] = pcf
        probs[low_m + 1, low_n + 1] = pcc

        return probs
    
    def calc_lawrence_n(self, mean_n, mean_m, var_m):
        l = LawrenceDist(self.phase_space_bounds_n[0], self.phase_space_bounds_n[1])
        g = DiscreteGaussian1D(self.phase_space_bounds_m[0], self.phase_space_bounds_m[1])

        p_n = l.calc_prob(mean_n)
        p_m = g.calc_prob(mean_m, var_m)

        return p_n * np.expand_dims(p_m, axis = -1)
    
    def calc_lawrence_m(self, mean_n, var_n, mean_m):
        l = LawrenceDist(self.phase_space_bounds_m[0], self.phase_space_bounds_m[1])
        g = DiscreteGaussian1D(self.phase_space_bounds_n[0], self.phase_space_bounds_n[1])

        p_m = l.calc_prob(mean_m)
        p_n = g.calc_prob(mean_n, var_n)

        return p_n * np.expand_dims(p_m, axis = -1)

    def calc_prob_internal(self, mean_n, mean_m, var_n, var_m, cov):
        """
        Internally calculates the probabilities parametrised by means, variances, and covariances.

        Those parameters have to be fitted.
        """

        thresh = 0.1
        if var_n < thresh and var_m < thresh:
            #print("double-lawrence")
            return self.calc_2d_lawrence(mean_n, mean_m, cov)
        if var_n < thresh:
            #print("law in n")
            return self.calc_lawrence_n(mean_n, mean_m, var_m)
        if var_m < thresh:
            #print("law in m")
            return self.calc_lawrence_m(mean_n, var_n, mean_m)

        if np.abs(cov) >= np.sqrt(var_n * var_m) * 0.95:
            cov = np.sign(cov) * np.sqrt(var_n * var_m) * 0.95


        cov_matrix = np.array([[var_n, cov],[cov, var_m]])
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        x = self.phase_space - np.array([mean_n, mean_m])
        q = np.squeeze(inv_cov_matrix @ np.expand_dims(x, axis = -1))
        exp = -0.5 * np.sum(x * q, axis = -1)
        probs = np.exp(exp)
        Z = np.sum(probs)
        return probs / Z

    def get_param(self, target_mean_n, target_mean_m , target_var_n, target_var_m, target_cov):
        """
        Fits the distribution parameters such that the resulting distribution has
        the requested moments.
        """

        #initial parameters
        p_mean_n = target_mean_n
        p_mean_m = target_mean_m
        p_var_n = target_var_n
        p_var_m = target_var_m
        p_cov = target_cov

        dt = 0.5
        probs = None
        for i in range(40):
            #current values
            probs = self.calc_prob_internal(p_mean_n, p_mean_m, p_var_n, p_var_m, p_cov)
            mean_n = np.sum(self.phase_space[:,:,0] * probs)
            mean_m = np.sum(self.phase_space[:,:,1] * probs)
            var_n = np.sum(self.phase_space[:,:,0] ** 2 * probs) - mean_n **2
            var_m = np.sum(self.phase_space[:,:,1] **2 * probs) - mean_m **2
            cov =  np.sum(self.phase_space[:,:,0] * self.phase_space[:,:,1] * probs) - mean_n * mean_m

            #deltas
            d_mean_n = target_mean_n - mean_n
            d_mean_m = target_mean_m - mean_m
            d_var_n = target_var_n - var_n
            d_var_m = target_var_m - var_m
            d_cov = target_cov - cov

            #apply
            p_mean_n += dt * d_mean_n
            p_mean_m += dt * d_mean_m
            p_var_n += dt * d_var_n
            p_var_m += dt * d_var_m
            p_cov += dt * d_cov

            #restrict
            if p_var_n < 0:
                p_var_n = 0

            if p_var_m < 0:
                p_var_m = 0


        return p_mean_n, p_mean_m, p_var_n, p_var_m, p_cov
    
    def calc_prob(self, mean_n, mean_m, var_n, var_m, cov):
        """
        Calculates a probability vector of shape = phase_space.shape for given mean and variance. It maximises entropy.

        Throws an error if distribution is too close to phase space borders.
        """

        if var_n < 0:
            raise ValueError("Variance in n has to be positive")
        
        if var_m < 0:
            raise ValueError("Variance in m has to be positive")

        if mean_n < self.phase_space_bounds_n[0] + np.sqrt(var_n):
            raise ValueError("mean n is too close to phase space border")
        
        if mean_n > self.phase_space_bounds_n[1] - np.sqrt(var_n):
            raise ValueError("mean n is too close to phase space border")
        
        if mean_m < self.phase_space_bounds_m[0] + np.sqrt(var_m):
            raise ValueError("mean m is too close to phase space border")
        
        if mean_m > self.phase_space_bounds_m[1] - np.sqrt(var_m):
            raise ValueError("mean m is too close to phase space border")
        
        d1 = mean_n - np.floor(mean_n)
        d2 = mean_m - np.floor(mean_m)

        # return double-lawrence dist for too law variance in both
        if var_n < d1 * (1 - d1) and var_m < d2 * (1 - d2):
            #print("direct double-lawrence")
            return self.calc_2d_lawrence(mean_n, mean_m, cov)
        # return uncorrelated dist for too low var in one component
        if var_n < d1 * (1 - d1):
           #print("direct lawrence in n")
           return self.calc_lawrence_n(mean_n, mean_m, var_m)
        # return uncorrelated dist for too low var in one component
        if var_m < d2 * (1 - d2):
           #print("direct lawrence in m")
           return self.calc_lawrence_m(mean_n, var_n, mean_m)
        
        # be careful with high pearson coefficients
        if np.abs(cov) > np.sqrt(var_n * var_m) * 0.95:
            cov = np.sign(cov) * np.sqrt(var_n * var_m) * 0.95

        p_mean_n, p_mean_m, p_var_n, p_var_m, p_cov = self.get_param(mean_n, mean_m, var_n, var_m, cov)
        probs = self.calc_prob_internal(p_mean_n, p_mean_m, p_var_n, p_var_m, p_cov)
        return probs