import numpy as np

from module.base.network import Network

class MeanField:
    """
    This class implements the meanfield solution algorithmn for charge configurations in gold-nanoparticle networks.
    """
    def __init__(self, network : Network, macrostate = None):
        """
        Creates a meanfield solver for a given Network instance.

        Parameters:
            network     :   instance of Network class containing network topology
            macrostate  :   initial average occupation numbers of electrons on each island
        """

        # save network
        self.network = network

        if macrostate is None:
            macrostate = np.zeros((self.network.N_particles,))
        
        assert len(macrostate.shape) == 1, "no valid macrostate was given"
        assert macrostate.shape[0] == self.network.N_particles, "wrong number of particles in macrostate"

        # initialise macrostate
        self.macrostate = macrostate

        # indices of islands to wich current is flowing
        self.island_indices = np.arange(self.network.N_particles)

        # indices of current-supplying neighbours for each target_index
        self.neighbour_indices = self.network.get_nearest_neighbours(self.island_indices)
        self.neighbour_indices = np.expand_dims(self.neighbour_indices, axis = 2)
        self.neighbour_indices = np.repeat(self.neighbour_indices, 4, axis = 2)

        # repeat island_indices to have the same shape as neighbour_indices. shape = (N_particles, 6)
        self.island_indices = np.expand_dims(self.island_indices, axis = [1,2])
        self.island_indices = np.repeat(self.island_indices, 6, axis = 1)
        self.island_indices = np.repeat(self.island_indices, 4, axis = 2)

        # neighbour mask to later remove tunnel rates from non-existing nearest_neighbours
        self.neighbour_mask = np.where(self.neighbour_indices != -1, 1, 0)

        # boolean masks to account for 4 different microstate combinations
        self.boolean_mask_A = np.array([False, False, True, True])
        self.boolean_mask_A = np.expand_dims(self.boolean_mask_A, axis = [0,1])
        self.boolean_mask_A = np.repeat(self.boolean_mask_A, self.network.N_particles, axis = 0)
        self.boolean_mask_A = np.repeat(self.boolean_mask_A, 6, axis = 1)

        self.boolean_mask_B = np.array([False, True, False, True])
        self.boolean_mask_B = np.expand_dims(self.boolean_mask_B, axis = [0,1])
        self.boolean_mask_B = np.repeat(self.boolean_mask_B, self.network.N_particles, axis = 0)
        self.boolean_mask_B = np.repeat(self.boolean_mask_B, 6, axis = 1)

        # simulation parameters
        self.dt = 1e-2 #nanoseconds
        self.t = 0

    def evolve(self, steps = 100):
        """
        Evolves the macrostate according to meanfield method by a given number of steps.
        """

        for _ in range(steps):

            # electrodes
            for i, pos in enumerate(self.network.electrode_pos):
                self.macrostate[self.network.get_linear_indices(pos)] += self.calc_expected_electrode_current(i) * self.dt

            # neighbours
            self.macrostate += self.calc_expected_island_currents() * self.dt

            # time
            self.t += self.dt
    

    def calc_expected_island_currents(self):
        """
        For the current macrostate, the currents contributing to charge movement are calculated for all islands.
        This does not calculate the currents from electrodes.
        """

        # axis = 0  :   listing all nanoparticles
        # axis = 1  :   listing all nearest_neighbours
        # axis = 2  :   different microstate configurations
        expanded_state = np.expand_dims(self.macrostate, axis = [0,1,2])
        expanded_state = np.repeat(expanded_state, self.network.N_particles, axis = 0)
        expanded_state = np.repeat(expanded_state, 6, axis = 1)
        expanded_state = np.repeat(expanded_state, 4, axis = 2)

        effective_states    = self.effective_operator(expanded_state, self.island_indices, self.boolean_mask_A)
        effective_states    = self.effective_operator(effective_states, self.neighbour_indices, self.boolean_mask_B)

        p1                  = self.calc_probability(expanded_state, self.island_indices, self.boolean_mask_A)
        p2                  = self.calc_probability(expanded_state, self.neighbour_indices, self.boolean_mask_B)


        rates = self.antisymmetric_tunnel_rate_islands(effective_states, self.neighbour_indices, self.island_indices) * p1 * p2 *  self.neighbour_mask

        return np.sum(rates, (1,2))
    
    def calc_expected_electrode_current(self, electrode_index):
        """
        For a given electrode index, the expected current to the corresponding island is calculated in terms of meanfield.
        """

        assert electrode_index in range(len(self.network.electrode_pos)), "no valid electrode index given"

        island_index = self.network.get_linear_indices(self.network.electrode_pos[electrode_index])

        effective_state_ground =    self.effective_operator(self.macrostate, island_index, np.array(True))
        p_ground               =    self.calc_probability(self.macrostate, island_index, np.array(True))

        effective_state_ceiling =    self.effective_operator(self.macrostate, island_index, np.array(False))
        p_ceiling               =    self.calc_probability(self.macrostate, island_index, np.array(False))

        rate1 = p_ground * self.antisymmetric_tunnel_rate_electrode(effective_state_ground, electrode_index)
        rate2 = p_ceiling * self.antisymmetric_tunnel_rate_electrode(effective_state_ceiling, electrode_index)

        return rate1 + rate2

    def effective_operator(self, macrostates, island_indices, microstates):
        """
        Implements the effective operator and thus calculates the effective state.

        For given macrostates with arbitrary leading shape the state at island_indices is modified to either be the floor or the ceiling state.

        Parameters:
            macrostates     :   numpy array of macrostates
            island_indices  :   numpy array defining wich island's occupation number to modify
            microstates     :   array containing either True to define the ground state or False to define the ceiling state

        Here, all arguments can have an arbitrary shape, except for macrostates, wich in the last dimension carry occupation numbers for each island, thus:
        shape(macrostates)      =   (..., N_particles)
        shape(island_indices)   =   (...)
        shape(microstates)      =   (...) 

        Returns:
            The effective state.
        """

        assert macrostates.shape[-1] == self.network.N_particles

        flat_macro = np.copy(macrostates.reshape((-1, self.network.N_particles))) # copy array
        flat_indices = island_indices.flatten()
        flat_microstates = microstates.flatten()
        n = flat_indices.shape[0]

        ground = np.floor(flat_macro[np.arange(n), flat_indices])
        ceil = np.ceil(flat_macro[np.arange(n), flat_indices])

        flat_macro[np.arange(n), flat_indices] = np.where(flat_microstates, ground, ceil)

        return flat_macro.reshape(macrostates.shape)
    
    
    def calc_probability(self, macrostates, island_indices, microstates):
        """
        Calculates the probability for an effective state to occur.

        For given macrostates with arbitrary leading shape, the probability of the by island_indices and microstates corresponding effective state
        is calculated. This is just the difference between the floor or ceiling state and the average state at the selected island.

        Parameters:
            macrostates     :   numpy array of macrostates
            island_indices  :   numpy array defining wich island's occupation number to account for
            microstates     :   array containing either True to define the ground state or False to define the ceiling state

        Here, all arguments can have an arbitrary shape, except for macrostates, wich in the last dimension carry occupation numbers for each island, thus:
        shape(macrostates)       =   (..., N_particles)
        shape(island_indices)    =   (...)
        shape(microstates)       =   (...) 

        Returns:
            The probabilties of the effective states occuring.
        """

        assert macrostates.shape[-1] == self.network.N_particles

        flat_macro = macrostates.reshape((-1, self.network.N_particles)) # copy array
        flat_indices = island_indices.flatten()
        flat_microstates = microstates.flatten()
        n = flat_indices.shape[0]

        ground = np.floor(flat_macro[np.arange(n), flat_indices])

        ps = flat_macro[np.arange(n), flat_indices] - ground

        prob = np.where(flat_microstates, 1 - ps, ps)

        return prob.reshape(island_indices.shape)
    

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
    
    def antisymmetric_tunnel_rate_electrode(self, occupation_numbers, electrode_index):
        """
        For given occupation numbers or states, this calculates the bidirectional tunnel rate towards the attached nanoparticle at electrode_index.

        This can be done in parallel by prepending additional dimensions to
        occupation_numbers, electrode_index

        shape(occupation_numbers)   = (..., N_particles)
        shape(electrode_index)                = (...)
        """
        
        assert occupation_numbers.shape[-1] == self.network.N_particles, "no valid system state given"

        return self.network.calc_rate_from_electrode(occupation_numbers, electrode_index) - self.network.calc_rate_to_electrode(occupation_numbers, electrode_index)