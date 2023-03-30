import numpy as np

class DiscreteGaussian1D:
    """
    To calculate discrete gaussian probabilities in one dimension. 
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

    def calc_prob_internal(self, my, alpha):
        """
        Internally calculates the probabilities for distribution parameters my and alpha.
        They do *not* correspond to the mean and variance.

        For too low alpha, Lawrence' distribution is resorted to.
        """
        alpha = np.abs(alpha)
        decimals = my - np.floor(my)

        if alpha <= 0.3: # lawrence dist for too small variances
            low = np.where(self.phase_space == np.floor(my))[0][0]
            pre = np.zeros(self.phase_space.shape)
            pre[low] = 1 - decimals
            pre[low + 1] = decimals

            return pre
        
        pre = np.exp(-(self.phase_space - my)**2 / alpha)
        
        Z = np.sum(pre, axis = 0)
        return pre / Z
    
    def get_param(self, target_mean, target_var):
        """
        Fits the distribution parameters my and alpha such that the resulting distribution has
        mean target_mean and variance target_var.
        """
        target_mean = np.array(target_mean, dtype="float")
        target_var = np.array(target_var, dtype="float")
        decimals = target_mean - np.floor(target_mean)

        # initial conditions
        dt = 1
        my = np.copy(target_mean)
        alpha = np.copy(target_var)

        opt_alpha = target_var > decimals * (1 - decimals)
        if not opt_alpha:
            alpha = 0

        for i in range(10):
            probs = self.calc_prob_internal(my, alpha)
            mean = np.sum(self.phase_space * probs)
            var = np.sum(self.phase_space**2 * probs) - mean**2

            delta_mean = target_mean - mean
            delta_var = target_var - var

            my += dt * delta_mean
            if opt_alpha:
                alpha += dt * delta_var

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

        my, alpha = self.get_param(mean, var)
        probs = self.calc_prob_internal(my, alpha)
        return probs