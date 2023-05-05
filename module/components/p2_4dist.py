import numpy as np

class p2_4dist:
    """
    To calculate the discrete Lawrence dist, assigning probability only to the two values
    adjacent to the mean.
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

    
    def calc_prob(self, mean, var):
        """
        Calculates a probability vector of shape = phase_space.shape for given mean.

        Throws an error if distribution is too close to phase space borders.
        """

        if np.floor(mean) < self.phase_space_min :
            raise ValueError("mean is too close to phase space border")
        
        if np.ceil(mean) > self.phase_space_max - 1:
            raise ValueError("mean is too close to phase space border")
        

        d = mean - np.floor(mean)

        var = np.clip(var, d * (1 - d), (2 - d) * (1 + d) - 0.2)
        alpha = -(d + 1) ** 2 + 5 * (d + 1) - 6 - var
        beta = 2 - d
        p0_opt = 1/40 * (2 - 10 * alpha - 8 * beta)
        a = 0.5 * (var - d * (1 - d))

        p0 = np.clip(p0_opt, 0, a * (2-d)/3)
        p1 = -0.5 * (alpha + 6 * p0)
        p2 = alpha + beta + 3*p0
        p3 = 1 - p0 - p1 - p2


        probs = np.zeros(self.phase_space.shape)
        low = np.where(self.phase_space == np.floor(mean))[0][0]

        probs[low - 1] = p0
        probs[low] = p1
        probs[low + 1] = p2
        probs[low + 2] = p3

        return probs