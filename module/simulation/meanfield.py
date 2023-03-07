import numpy as np

from module.base.network import Network

class MeanField:
    """
    This class implements the meanfield solution algorithmn for charge configurations in gold-nanoparticle networks.
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

    def confidence_based_solve(self, macrostate = None, eps = 1e-2, N_max = 100):
        """
        Solves for the equilibrium state in the current network configuration by a confidence based algoritmn. The state will be searched until the total current for each
        island is less than eps.

        Raises an exception if no appropriate solution is found within N_max steps.
        """

        assert eps > 0, "eps must be greater than zero"

        gamma = 1.2 # confidence step increase
        lambd = 0.5 # confidence reduction
        
        # initial condition
        if macrostate is None:
            macrostate = np.zeros(self.network.N_particles)
        currents = self.calc_total_currents(macrostate)
        
        # initial step
        step = 1/(np.sqrt(np.sum(currents**2)) + eps) * currents

        for i in range(N_max):
            if np.all(np.abs(currents) < eps): return macrostate

            # forward step
            a = np.sign(currents) * 1e-8 + np.clip(step * gamma, a_min = -1, a_max = 1)

            # change direction
            b = -lambd * step

            step = np.where(currents * step >= 0, a, b)

            # update
            macrostate = macrostate + step
            currents = self.calc_total_currents(macrostate)

        raise RuntimeError("confidence-based-solve was not able to find a soution in the given amount of steps")
    
    def numeric_integration_solve(self, macrostate = None, dt = 0.1, N = 50):
        """
        Integrates the currents over time to find a steady solution.
        """

        if macrostate is None:
            macrostate = np.zeros(self.network.N_particles)

        for i in range(N):
            macrostate += self.calc_total_currents(macrostate) * dt

        return macrostate


    def calc_total_currents(self, macrostate):
        """
        For a given macrostate of shape (N_particles,), this calculates all the currents flowing to the nanoparticles. This sums over all nearest neighbours and includes electrodes.
        The current is given in electron charges per ns.
        """

        currents = np.sum(self.calc_expected_island_currents(macrostate), axis = 1)

        for i, pos in enumerate(self.network.electrode_pos):
            particle_index = self.network.get_linear_indices(pos)
            currents[particle_index] += self.calc_expected_electrode_current(macrostate, electrode_index = i)

        return currents
        

    def calc_expected_island_currents(self, macrostate):
        """
        For the given macrostate the expected currents for all nanoparticles originating from their neighbours are calculated.
        Thereby, an expectation over all combinations of effective states, where either the neighbours or the target islands
        lie in ground or ceiling state are concerned.
        """
        
        final_currents = np.zeros((self.network.N_particles, 6))

        # both ground state
        effective_states, p = self.calc_effective_states(macrostate, True, True)
        final_currents += self.antisymmetric_tunnel_rate_islands(effective_states, self.neighbour_indices, self.island_indices) * p

        # both ceiling state
        effective_states, p = self.calc_effective_states(macrostate, False, False)
        final_currents += self.antisymmetric_tunnel_rate_islands(effective_states, self.neighbour_indices, self.island_indices) * p


        # mixed state
        effective_states, p = self.calc_effective_states(macrostate, True, False)
        final_currents += self.antisymmetric_tunnel_rate_islands(effective_states, self.neighbour_indices, self.island_indices) * p

        # other mixed state
        effective_states, p = self.calc_effective_states(macrostate, False, True)
        final_currents += self.antisymmetric_tunnel_rate_islands(effective_states, self.neighbour_indices, self.island_indices) * p


        return final_currents
    
    def calc_expected_electrode_current(self, macrostate, electrode_index):
        """
        Calculates the expected current flowing to the nanoparticle connected to electrode at electrode_index.
        """

        assert len(macrostate.shape) == 1, "only one macrosate accepted"
        assert macrostate.shape[0] == self.network.N_particles, "wrong number of particles for macrostate"
        assert electrode_index in range(len(self.network.electrode_pos)), "no valid electrode index"

        particle_index = self.network.get_linear_indices(self.network.electrode_pos[electrode_index])

        # ground state
        ground_state = self.effective_operator(macrostate, particle_index, np.array(True))
        p_ground = self.calc_probability(macrostate, particle_index, np.array(True))
        current_ground = self.antisymmetric_tunnel_rate_electrode(ground_state, electrode_index)

        # ceiling state
        ceiling_state = self.effective_operator(macrostate, particle_index, np.array(False))
        p_ceiling = self.calc_probability(macrostate, particle_index, np.array(False))
        current_ceiling = self.antisymmetric_tunnel_rate_electrode(ceiling_state, electrode_index)

        return current_ground * p_ground + current_ceiling * p_ceiling

    
    def calc_effective_states(self, macrostate, neighbour_in_ground_state = True, island_in_ground_state = True):
        """
        Turns the current macrostate into a grid of effective states, each corresponding to the tunnel event they are associated with. Each row in the (N_particles, 6) grid
        corresponds to the particle current is flowing to. The 6 possible entries per row describe all 6 possible next neighbours providing current. Currents of nearest neighbours
        who don't exist are set to zero.

        Returns:
            (N_particle, 6, N_particle) grid containing all effective states
            (N_particle, 6) grid containing the probabilities of the effective states to occur 
        """

        assert len(macrostate.shape) == 1, "only one macrosate accepted"
        assert macrostate.shape[0] == self.network.N_particles, "wrong number of particles for macrostate"

        assert isinstance(neighbour_in_ground_state, bool), "expected a boolean value"
        assert isinstance(island_in_ground_state, bool), "expected a boolean value"

        neighbour_in_ground_state = np.array(neighbour_in_ground_state)
        island_in_ground_state = np.array(island_in_ground_state)


        # axis = 0  :   listing all nanoparticles
        # axis = 1  :   listing all nearest_neighbours
        expanded_state = np.expand_dims(macrostate, axis = [0,1])
        expanded_state = np.repeat(expanded_state, self.network.N_particles, axis = 0)
        expanded_state = np.repeat(expanded_state, 6, axis = 1)


        # apply effective operator two times:

        # neighbour state and probability
        effective_states    = self.effective_operator(expanded_state, self.neighbour_indices, neighbour_in_ground_state)
        p_neighbour = self.calc_probability(expanded_state, self.neighbour_indices, neighbour_in_ground_state)

        # effective island state and probability
        effective_states    = self.effective_operator(effective_states, self.island_indices, island_in_ground_state)
        p_island = self.calc_probability(expanded_state, self.island_indices, island_in_ground_state)


        effective_states = effective_states * np.expand_dims(self.neighbour_mask, axis = -1)
        p = p_neighbour * p_island * self.neighbour_mask

        return effective_states, p



    def effective_operator(self, macrostates, island_indices, groundstate):
        """
        Implements the effective operator and thus calculates the effective state.

        For given macrostates with arbitrary leading shape the state at island_indices is modified to either be the floor or the ceiling state.

        Parameters:
            macrostates     :   numpy array of macrostates
            island_indices  :   numpy array defining wich island's occupation number to modify
            groundstate     :   array containing either True to define the ground state or False to define the ceiling state

        Here, all arguments can have an arbitrary shape, except for macrostates, wich in the last dimension carry occupation numbers for each island, thus:
        shape(macrostates)      =   (..., N_particles)
        shape(island_indices)   =   (...)
        shape(groundstate)      =   (...) 

        Returns:
            The effective state.
        """

        assert macrostates.shape[-1] == self.network.N_particles

        flat_macro = np.copy(macrostates.reshape((-1, self.network.N_particles))) # copy array
        flat_indices = island_indices.flatten()
        flat_groundstate = groundstate.flatten()
        n = flat_indices.shape[0]

        ground = np.floor(flat_macro[np.arange(n), flat_indices])
        ceil = np.ceil(flat_macro[np.arange(n), flat_indices])

        flat_macro[np.arange(n), flat_indices] = np.where(flat_groundstate, ground, ceil)

        return flat_macro.reshape(macrostates.shape)
    
    
    def calc_probability(self, macrostates, island_indices, groundstate):
        """
        Calculates the probability for an effective state to occur.

        For given macrostates with arbitrary leading shape, the probability of the by island_indices and groundstate corresponding effective state
        is calculated. This is just the difference between the floor or ceiling state and the average state at the selected island.

        Parameters:
            macrostates     :   numpy array of macrostates
            island_indices  :   numpy array defining wich island's occupation number to account for
            groundstate     :   array containing either True to define the ground state or False to define the ceiling state

        Here, all arguments can have an arbitrary shape, except for macrostates, wich in the last dimension carry occupation numbers for each island, thus:
        shape(macrostates)       =   (..., N_particles)
        shape(island_indices)    =   (...)
        shape(groundstate)       =   (...) 

        Returns:
            The probabilties of the effective states occuring.
        """

        assert macrostates.shape[-1] == self.network.N_particles

        flat_macro = macrostates.reshape((-1, self.network.N_particles)) # copy array
        flat_indices = island_indices.flatten()
        flat_groundstate = groundstate.flatten()
        n = flat_indices.shape[0]

        ground = np.floor(flat_macro[np.arange(n), flat_indices])

        ps = flat_macro[np.arange(n), flat_indices] - ground

        prob = np.where(flat_groundstate, 1 - ps, ps)

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
        shape(electrode_index)      = (...)
        """
        
        assert occupation_numbers.shape[-1] == self.network.N_particles, "no valid system state given"

        return self.network.calc_rate_from_electrode(occupation_numbers, electrode_index) - self.network.calc_rate_to_electrode(occupation_numbers, electrode_index)