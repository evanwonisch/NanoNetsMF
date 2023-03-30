import numpy as np

class LawrenceDist:
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

    
    def calc_prob(self, mean):
        """
        Calculates a probability vector of shape = phase_space.shape for given mean.

        Throws an error if distribution is too close to phase space borders.
        """

        if np.floor(mean) < self.phase_space_min :
            raise ValueError("mean is too close to phase space border")
        
        if np.ceil(mean) > self.phase_space_max - 1:
            raise ValueError("mean is too close to phase space border")

        probs = np.zeros(self.phase_space.shape)
        decimals = mean - np.floor(mean)
        low = np.where(self.phase_space == np.floor(mean))[0][0]
        probs[low] = 1 - decimals

        if low + 1 < self.phase_space.shape[0]:
            probs[low + 1] = decimals

        return probs