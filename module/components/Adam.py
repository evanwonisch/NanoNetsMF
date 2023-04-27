import numpy as np

class Adam:
    """
    Implements the adaptive moment estimation strategy for better convergence
    """
    def __init__(self, props, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        """
        Parameters:
            props       : list of numpy arrays to perform gradient steps on
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.props = props
        self.N_props = len(props)

        self.V = [np.zeros(prop.shape) for prop in props] # momentum
        self.S = [np.ones(prop.shape) for prop in props] # RMSProp

        self.steps = [np.zeros(prop.shape) for prop in props]


    def calc_step(self, grad_props, learning_rate = 0.1):
        """
        Calculates the current step to props for given gradients.

        Parameters:
            grap_props  : list of numpy arrays
        """

        if not isinstance(learning_rate, list):
            learning_rate = np.ones(self.N_props) * learning_rate

        for i in range(self.N_props):
            self.V[i] = self.beta1 * self.V[i] + (1 - self.beta1) * grad_props[i]
            self.S[i] = self.beta2 * self.S[i] + (1 - self.beta2) * grad_props[i] ** 2

            self.steps[i] = learning_rate[i] * self.V[i] / (np.sqrt(self.S[i]) + self.eps)


        return self.steps


