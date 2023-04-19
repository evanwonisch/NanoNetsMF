import numpy as np

from module.base.network import Network
import module.components.CONST as CONST
from module.components.Adam import Adam

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

        # repeated island indices (N, 6, 4, 4)
        self.r_island_indices = np.expand_dims(np.expand_dims(self.island_indices, axis = -1), axis = -1)
        self.r_island_indices = np.repeat(self.r_island_indices, 4, axis = -1)
        self.r_island_indices = np.repeat(self.r_island_indices, 4, axis = -2)

        # repeated neighbour indices (N, 6, 4, 4)
        self.r_neighbour_indices = np.expand_dims(np.expand_dims(self.neighbour_indices, axis = -1), axis = -1)
        self.r_neighbour_indices = np.repeat(self.r_neighbour_indices, 4, axis = -1)
        self.r_neighbour_indices = np.repeat(self.r_neighbour_indices, 4, axis = -2)

        # repeat microstates for islands (N, 6, 4, 4)
        self.effective_states_islands = np.repeat(np.expand_dims(np.array([-1,0,1,2]), axis = -1), 4, axis = -1)
        self.effective_states_islands = np.expand_dims(self.effective_states_islands, axis = [0, 1])
        self.effective_states_islands = np.repeat(self.effective_states_islands, 6, axis = 1)
        self.effective_states_islands = np.repeat(self.effective_states_islands, self.network.N_particles, axis = 0)

        # repeat microstates for neighbours (N, 6, 4, 4)
        self.effective_states_neighbours = np.repeat(np.expand_dims(np.array([-1,0,1,2]), axis = 0), 4, axis = 0)
        self.effective_states_neighbours = np.expand_dims(self.effective_states_neighbours, axis = [0, 1])
        self.effective_states_neighbours = np.repeat(self.effective_states_neighbours, 6, axis = 1)
        self.effective_states_neighbours = np.repeat(self.effective_states_neighbours, self.network.N_particles, axis = 0)

        # means
        self.means = np.zeros(self.network.N_particles)

        # variances
        self.vars = np.zeros(self.network.N_particles)

        # Adam optimizer
        self.opt = None
    
    def numeric_integration_solve(self, dt = 0.05, N = 100, verbose = False, reset = True):
        """
        Integrates the currents over time to find a steady solution.

        Parameters:
            dt          : integration step
            N           : number of iterations
        """

        if reset:
            self.means = np.zeros(self.network.N_particles)
            self.vars = np.ones(self.network.N_particles)

        for i in range(N):
            dmeans, dvars = self.calc_derivatives()
            self.means += dt * dmeans
            self.vars += dt * dvars

            self.clip_vars()

        if verbose:
            print("convergence:", self.convergence_metric())

    def ADAM_solve(self, N = 60, learning_rate = 0.1, reset = True, verbose = False):
        """
        Perfroms 60 ADAM steps on derivatives.
        """
        if reset:
            self.means = np.zeros(self.network.N_particles)
            self.vars = np.zeros(self.network.N_particles)
            self.opt = Adam([self.means, self.vars])

        for i in range(N):
            dmean, dvar = self.opt.calc_step(self.calc_derivatives(), learning_rate)
            self.means += dmean
            self.vars += dvar
            self.clip_vars()

        if verbose:
            print("ADAM convergence:", self.ADAM_convergence_metric())

    def ADAM_convergence_metric(self):
        """
        Calculates the mean absolute value of the momentum terms stored in ADAM algortihmn for the means and variances.
        """

        if self.opt is None:
            raise Exception("ADAM optimizier was not yet initialised. Call ADAM_solve().")
        
        return (np.max(np.abs(self.opt.V[0])), np.max(np.abs(self.opt.V[1])))

    def convergence_metric(self):
        """
        Calculates a 2D convergence metric. The first component is the maximum of the time derivative of the mean.
        The second one the respective maximum vor the variances.
        """
        dmeans, dvars = self.calc_derivatives()
        
        return (np.abs(dmeans).max(), np.abs(dvars).max())
    
    def clip_vars(self):
        """
        Clips variances back to senseful intervall for current means.
        """
        d = self.means - np.floor(self.means) # digits
        self.vars = np.clip(self.vars, d * (1 - d), (2 - d) * (1 + d))

    def calc_derivatives(self):
        """
        Calculates the time derivatives of the mean and the variances

        Returns:
            dmeans
            dvars
        """

        dmeans = np.zeros(self.network.N_particles)
        dvars = np.zeros(self.network.N_particles)

        #islands
        currents_island, currents_dag_island, n_currents_island = self.calc_expectation_islands()
        dmeans += np.sum(currents_island, axis = -1)
        dvars += np.sum(2 * n_currents_island + currents_dag_island, axis = -1)

        #electrodes
        for electrode_index in range(len(self.network.electrode_pos)):
            island_index = self.network.get_linear_indices(self.network.electrode_pos[electrode_index])
            current, current_dag, n_current = self.calc_expectation_electrodes(electrode_index)
            dmeans[island_index] += current
            dvars[island_index] += 2 * n_current + current_dag


        dvars -= 2 * self.means * dmeans

        # if variances tend outside of valid interval, set derivative to zero
        d = self.means - np.floor(self.means) # digits
        cond1 = np.logical_and(self.vars <= d*(1-d) + 1e-3, np.sign(dvars) < 0)
        cond2 = np.logical_and(self.vars >= (2-d)*(1+d) - 1e-3, np.sign(dvars) > 0)
        cond = np.logical_or(cond1, cond2)
        dvars = np.where(cond, 0, dvars)

        return dmeans, dvars

    def calc_expectation_islands(self):
        """
        Calculates the expectation values needed for dynamics concerning only other islands. The calculation is broadcasted to all islands and neighbours at once.

        Returns:
            exp_currents        : the currents flowing towards islands in axis 0 by neighbours in axis 1
            exp_currents_dag    : the currents with opposite internal sign
            exp_n_currents      : the expectation of the current from island i to j times the occupation number of i
        """

        # (N, 6, 4, 4, N)
        effective_states = np.expand_dims(self.means, axis = [0, 1, 2, 3])
        effective_states = np.repeat(effective_states, self.network.N_particles, axis = 0)
        effective_states = np.repeat(effective_states, 6, axis = 1)
        effective_states = np.repeat(effective_states, 4, axis = 2)
        effective_states = np.repeat(effective_states, 4, axis = 3)
        assert effective_states.shape == (self.network.N_particles, 6, 4, 4, self.network.N_particles)

        # (N, 6, 4, 1)
        probs_islands = np.expand_dims(self.calc_probability(self.island_indices), axis = -1)

        # (N, 6, 1, 4)
        probs_neighbours = np.expand_dims(self.calc_probability(self.neighbour_indices) * np.expand_dims(self.neighbour_mask, axis = -1), axis = -2)

        # (N, 6, 4, 4)
        probs = probs_islands * probs_neighbours


        effective_states, occ_island = self.effective_operator(effective_states, self.r_island_indices, self.effective_states_islands) 
        effective_states, occ_neighbour = self.effective_operator(effective_states, self.r_neighbour_indices, self.effective_states_neighbours)

        currents = self.antisymmetric_tunnel_rate_islands(effective_states, self.r_neighbour_indices, self.r_island_indices)
        currents_dag = self.symmetric_tunnel_rate_islands(effective_states, self.r_neighbour_indices, self.r_island_indices)
        n_currents = currents * occ_island

        exp_currents = np.sum(np.sum(currents * probs, axis = -1), axis = -1)
        exp_currents_dag = np.sum(np.sum(currents_dag * probs, axis = -1), axis = -1)
        exp_n_currents = np.sum(np.sum(n_currents * probs, axis = -1), axis = -1)

        return exp_currents, exp_currents_dag, exp_n_currents
    
    def calc_expected_electrode_current(self, electrode_index):
        """
        Calculates the current flowing from an electrode into the system.
        """
        return self.calc_expectation_electrodes(electrode_index)[0] * CONST.electron_charge

    def calc_expectation_electrodes(self, electrode_index):
        """
        Calculates the expectation values of the current, the dagger-current and the n-current for an electrode and its attached particle.

        Returns:
            current
            current_dag
            n_current
        """
        island_index = self.network.get_linear_indices(self.network.electrode_pos[electrode_index])

        states = np.expand_dims(self.means, axis = 0)
        states = np.repeat(states, 4, axis = 0)

        effective_states, occ_num = self.effective_operator(states, np.repeat(island_index, 4), np.array([-1,0,1,2]))
        probs = self.calc_probability(np.array(island_index))

        current = self.antisymmetric_tunnel_rate_electrode(effective_states, electrode_index)
        current_dag = self.symmetric_tunnel_rate_electrode(effective_states, electrode_index)
        n_current = current * occ_num

        exp_current = np.sum(current * probs)
        exp_current_dag = np.sum(current_dag * probs)
        exp_n_current = np.sum(n_current * probs)

        return exp_current, exp_current_dag, exp_n_current

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
            the occupation numbers of the concerned islands put into their effective state
        """

        assert island_indices.shape == effective_states.shape, "island indices and effective_states must have the same shape"
        assert island_indices.shape + (self.network.N_particles,) == states.shape, "no valid states are given"

        orig_shape = island_indices.shape

        f_states = states.reshape((-1, self.network.N_particles))
        f_indices = island_indices.reshape((-1,))
        f_effective_states =  effective_states.reshape((-1,))
        N = f_states.shape[0]

        occupation_numbers = np.floor(f_states[np.arange(N, dtype="int"), f_indices]) + f_effective_states
        f_states[np.arange(N, dtype="int"), f_indices] = occupation_numbers

        return f_states.reshape(orig_shape + (self.network.N_particles,)), occupation_numbers.reshape(orig_shape)
    
    
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

        var = np.clip(var, d * (1 - d), (2 - d) * (1 + d) - 0.2)
        alpha = -(d + 1) ** 2 + 5 * (d + 1) - 6 - var
        beta = 2 - d
        p0_opt = 1/40 * (2 - 10 * alpha - 8 * beta)
        a = 0.5 * (var - d * (1 - d))

        p0 = np.clip(p0_opt, 0, a * (2-d)/3)
        p1 = -0.5 * (alpha + 6 * p0)
        p2 = alpha + beta + 3*p0
        p3 = 1 - p0 - p1 - p2

        # expanded Lawrence distribution
        # alpha = 0.5 * (var - d * (1 - d))
        # alpha = np.clip(alpha, 0, 1)

        # p0 = alpha * (2 - d) / 3        # for floor(mean) - 1
        # p1 = (1 - alpha) * (1 - d)      # for floor(mean)
        # p2 = (1- alpha) * d             # for floor(mean) + 1
        # p3 = alpha * (1 -  (2 - d) / 3) # for floor(mean) + 2


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