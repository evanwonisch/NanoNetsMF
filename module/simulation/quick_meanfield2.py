import numpy as np

from module.base.network import Network

class QuickMeanField2:
    """
    This class implements a second order meanfield approximation accounting for means and variances.
    It utilises a generalised lawrence-distribution allowing not just for values adjacent to the mean, but also one step further.
    """
    def __init__(self, network : Network):
        """
        Creates a meanfield solver for a given Network instance.

        Parameters:
            network     :   instance of Network class containing network topology
        """

        # save network
        self.network = network

        # indices of islands to which current is flowing
        self.island_indices = np.arange(self.network.N_particles)

        # indices of current-supplying neighbours for each target_index
        self.neighbour_indices = self.network.get_nearest_neighbours(self.island_indices)

        # repeat island_indices to have the same shape as neighbour_indices. shape = (N_particles, 6)
        self.island_indices = np.expand_dims(self.island_indices, axis = 1)
        self.island_indices = np.repeat(self.island_indices, 6, axis = 1)

        # neighbour mask to later remove tunnel rates from non-existing nearest_neighbours
        self.neighbour_mask = np.where(self.neighbour_indices != -1, 1, 0)

        # means
        self.means = np.zeros(self.network.N_particles)

        # variances
        self.vars = np.ones(self.network.N_particles)

        # stores the calculated derivatives
        self.dmeans = np.zeros(self.network.N_particles)
        self.dvars = np.zeros(self.network.N_particles)
    
    def numeric_integration_solve(self, dt = 0.1, N = 100, verbose = False):
        """
        Integrates the currents over time to find a steady solution.

        Parameters:
            dt          : integration step
            N           : number of iterations
        """
        pass

    def convergence_metric(self, macrostate):
        pass

    def calc_derivatives(self):
        """
        Calculates the time derivatives of the means and variances.
        """

        # (N, 6, 4, 4, N)
        effective_states = np.expand_dims(self.means, axis = [0, 1, 2, 3])
        effective_states = np.repeat(effective_states, self.network.N_particles, axis = 0)
        effective_states = np.repeat(effective_states, 6, axis = 1)
        effective_states = np.repeat(effective_states, 4, axis = 2)
        effective_states = np.repeat(effective_states, 4, axis = 3)

        # (N, 6, 4, 1)
        probs_islands = np.expand_dims(self.calc_probability(self.island_indices), axis = -1)

        # (N, 6, 1, 4)
        probs_neighbours = np.expand_dims(self.calc_probability(self.neighbour_indices) * np.expand_dims(self.neighbour_mask, axis = -1), axis = -2)

        # (N, 6, 4, 4)
        probs = probs_islands * probs_neighbours

        # repeated island indices (N, 6, 4, 4)
        r_island_indices = np.expand_dims(np.expand_dims(self.island_indices, axis = [-1]), axis = -1)
        r_island_indices = np.repeat(r_island_indices, 4, axis = -1)
        r_island_indices = np.repeat(r_island_indices, 4, axis = -2)

        # repeated neighbour indices (N, 6, 4, 4)
        r_neighbour_indices = np.expand_dims(np.expand_dims(self.neighbour_indices, axis = [-1]), axis = -1)
        r_neighbour_indices = np.repeat(r_neighbour_indices, 4, axis = -1)
        r_neighbour_indices = np.repeat(r_neighbour_indices, 4, axis = -2)


        states = self.effective_operator(np.copy(effective_states), r_island_indices, )
        states = self.effective_operator(states, self.neighbour_indices, -1)



        currents = self.antisymmetric_tunnel_rate_island(states, self.neighbour_indices, self.island_indices)
        n_currents = None # there is work to be done
        currents_dag = self.symmetric_tunnel_rate_island(states, self.neighbour_indices, self.island_indices)





    def effective_operator(self, states, island_indices, effective_states):
        """
        Based on a given assembly of states, occupation numbers of islands at island_indices are replaced by effective states indicated by effective_states.
        effective states are integers interpreted as the distance from the floor value of the mean.

        Prameters:
            states              : shape = (..., N_particles)
            island_indices      : shape = (...)
            effective_states    : shape = (...)

        Returns:
            the effective states
        """

        assert island_indices.shape == effective_states.shape, "island indices and effective_states must have the same shape"
        assert island_indices.shape + (self.network.N_particles,) == states.shape, "no valid states are given"

        orig_shape = island_indices.shape

        f_states = states.reshape((-1, self.network.N_particles))
        f_indices = island_indices.reshape((-1,))
        f_effective_states =  effective_states.reshape((-1,))
        N = f_states.shape[0]

        print(f_states[np.arange(N, dtype="int"), f_indices])
        f_states[np.arange(N, dtype="int"), f_indices] = np.floor(f_states[np.arange(N, dtype="int"), f_indices]) + f_effective_states

        return f_states.reshape(orig_shape + (self.network.N_particles,))
    
    
    def calc_probability(self, island_indices):
        """
        Calculates the four probabilities of an island being in effective states -1,0,1,2 respectively.

        Parameters:
            island_indices      : indices of islands of which to calculate probabilities; they are calculated for floor(mean) - 1 tp floor(mean) + 2 in order.

        Returns:
            probabilities       : shape = (island_indices.shape, 4) hence four probabilities for each island
        """
        orig_shape = island_indices.shape
        f_indices = island_indices.flatten()

        mean = self.means[f_indices]
        var = self.vars[f_indices]
        d = mean - np.floor(mean)

        alpha = 0.5 * (var - d * (1 - d))
        alpha = np.clip(alpha, 0, 1)

        p0 = alpha * (2 - d) / 3        # for floor(mean) - 1
        p1 = (1 - alpha) * (1 - d)      # for floor(mean)
        p2 = (1- alpha) * d             # for floor(mean) + 1
        p3 = alpha * (1 -  (2 - d) / 3) # for floor(mean) + 2


        p0 = p0.reshape(orig_shape)
        p1 = p1.reshape(orig_shape)
        p2 = p2.reshape(orig_shape)
        p3 = p3.reshape(orig_shape)

        return np.stack((p0,p1,p2,p3), axis = -1)

    

    def antisymmetric_tunnel_rate_islands(self, occupation_numbers, alpha, beta):
        """
        For given occupation numbers or states, the antisymmetric or bidirectional tunnel rate from island alpha to island beta is calculated.

        This can be done in parallel by prepending additional dimensions to
        occupation_numbers, alpha and beta:

        shape(occupation_numbers)   = (..., N_particles)
        shape(alpha)                = (...)
        shape(beta)                 = (...)
        """

        assert occupation_numbers.shape[-1] == self.network.N_particles, "no valid system state given"


        return self.network.calc_rate_island(occupation_numbers, alpha, beta) - self.network.calc_rate_island(occupation_numbers, beta, alpha)
    
    def symmetric_tunnel_rate_islands(self, occupation_numbers, alpha, beta):
        """
        For given occupation numbers or states, the symmetric tunnel rate involving islands alpha and beta is calculated.

        This can be done in parallel by prepending additional dimensions to
        occupation_numbers, alpha and beta:

        shape(occupation_numbers)   = (..., N_particles)
        shape(alpha)                = (...)
        shape(beta)                 = (...)
        """

        assert occupation_numbers.shape[-1] == self.network.N_particles, "no valid system state given"


        return self.network.calc_rate_island(occupation_numbers, alpha, beta) + self.network.calc_rate_island(occupation_numbers, beta, alpha)
    
    def antisymmetric_tunnel_rate_electrode(self, occupation_numbers, electrode_index):
        """
        For given occupation numbers or states, this calculates the bidirectional tunnel rate towards the attached nanoparticle at electrode_index.

        This can be done in parallel by prepending additional dimensions to
        occupation_numbers, electrode_index

        shape(occupation_numbers)   = (..., N_particles)
        shape(electrode_index)      = (...)
        """
        
        assert occupation_numbers.shape[-1] == self.network.N_particles, "no valid system state given"

        return self.network.calc_rate_from_electrode(occupation_numbers, electrode_index) - self.network.calc_rate_to_electrode(occupation_numbers, electrode_index)
    
    def symmetric_tunnel_rate_electrode(self, occupation_numbers, electrode_index):
        """
        For given occupation numbers or states, this calculates the symmetric tunnel rate involving the island and electrode at electrode_index

        This can be done in parallel by prepending additional dimensions to
        occupation_numbers, electrode_index

        shape(occupation_numbers)   = (..., N_particles)
        shape(electrode_index)      = (...)
        """
        
        assert occupation_numbers.shape[-1] == self.network.N_particles, "no valid system state given"

        return self.network.calc_rate_from_electrode(occupation_numbers, electrode_index) + self.network.calc_rate_to_electrode(occupation_numbers, electrode_index)