import numpy as np

class RestrictedGaussian1D:
    """
    To calculate discrete gaussian probabilities restricted to 4 values in one dimension. 
    """

    def __init__(self, phase_space_min = -20, phase_space_max = 20):
        """
        Instatiates a discrete Gaussian distribution generator.

        Parameters:
            phase_space_min: integer of lowest accounted state
            phase_space_max: integer of highest accounted state
        """
        self.phase_space_min = phase_space_min
        self.phase_space_max = phase_space_max

        self.phase_space = np.arange(phase_space_min, phase_space_max)

    def calc_lawrence_1d(self, mean):
        decimals = mean - np.floor(mean)
        low = np.where(self.phase_space == np.floor(mean))[0][0]

        pre = np.zeros(self.phase_space.shape)
        pre[low] = 1 - decimals
        pre[low + 1] = decimals

        return pre

    def calc_prob_internal(self, my, alpha):
        """
        Internally calculates the probabilities for distribution parameters my and alpha.
        They do *not* correspond to the mean and variance.

        For too low alpha, Lawrence' distribution is resorted to.
        """
        alpha = np.abs(alpha)

        thresh = 0.3
        if alpha <= thresh: # outputs lawrence dist for small alpha parameter
           return self.calc_lawrence_1d(my)
        
        pre = np.exp(-(self.phase_space - my)**2 / alpha)

        index_0 = np.where(self.phase_space == np.floor(my))[0][0]

        post = np.zeros(self.phase_space.shape)
        post[index_0 - 1] = pre[index_0 - 1]
        post[index_0] = pre[index_0]
        post[index_0 + 1] = pre[index_0 + 1]
        post[index_0 + 2] = pre[index_0 + 2]
        
        Z = np.sum(post, axis = 0)
        return post / Z
    
    def get_param(self, target_mean, target_var):
        """
        Fits the distribution parameters my and alpha such that the resulting distribution has
        mean target_mean and variance target_var.
        """
        target_mean = np.array(target_mean, dtype="float")
        target_var = np.array(target_var, dtype="float")

        # initial conditions
        dt = 0.9
        my = np.copy(target_mean)
        alpha = np.copy(target_var)

        for i in range(20):
            probs = self.calc_prob_internal(my, alpha)
            mean = np.sum(self.phase_space * probs)
            var = np.sum(self.phase_space**2 * probs) - mean**2

            delta_mean = target_mean - mean
            delta_var = target_var - var

            my += dt * delta_mean
            alpha += dt * delta_var

            if alpha < 0: # restricts alpha to senseful intervall
                alpha = 0

        return my, alpha
    
    def calc_prob(self, mean, var):
        """
        Calculates a probability vector of shape = phase_space.shape for given mean and variance. It maximises entropy.

        Throws an error if distribution is too close to phase space borders.
        """
        if var < 0:
            raise ValueError("variance must be positive")

        if mean < self.phase_space_min + np.sqrt(var):
            raise ValueError("mean is too close to phase space border")
        
        if mean > self.phase_space_max - np.sqrt(var):
            raise ValueError("mean is too close to phase space border")
        
        # directly returns a lawrence dist if infeasable
        decimals = mean - np.floor(mean)
        if var < decimals * (1 - decimals):
            return self.calc_lawrence_1d(mean)

        my, alpha = self.get_param(mean, var)
        probs = self.calc_prob_internal(my, alpha)
        return probs