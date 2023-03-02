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

        self.network = network

        if not macrostate:
            macrostate = np.zeros((self.network.N_particles,))
        
        assert len(macrostate.shape) == 1, "no valid macrostate was given"
        assert macrostate.shape[0] == self.network.N_particles, "wrong number of particles in macrostate"

        self.macrostate = macrostate

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
    
    
    def calc_porbability(self, macrostates, island_indices, microstates):
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