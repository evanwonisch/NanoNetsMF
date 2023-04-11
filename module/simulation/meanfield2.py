import numpy as np
from module.base.network import Network
from module.components.discrete_gaussian1D import DiscreteGaussian1D
from module.components.discrete_gaussian2D import DiscreteGaussian2D
import module.components.CONST as CONST

class MeanField2:
    """
    Implements a full second-order meanfield solution algorithmn using a 2D-discrete gaussian with full covariance.
    """
    def __init__(self, network, include_covs = True):
        """
        Parameters:
            network:        : the network instance to operate with
            include_covs    : whether to include covariances or not
        """
        self.net = network

        self.g2 = DiscreteGaussian2D(phase_space_bounds_n=(-15,15), phase_space_bounds_m=(-15,15))
        self.g1 = DiscreteGaussian1D(phase_space_min=-15, phase_space_max=15)

        self.neighbour_table = self.net.get_nearest_neighbours(np.arange(0, self.net.N_particles))
        self.island_indices = self.net.get_linear_indices(self.net.electrode_pos)

        # include covariances
        self.include_covs = include_covs

        # means
        self.means = np.zeros(self.net.N_particles)
        self.dmeans = np.zeros(self.net.N_particles)

        # variances
        self.vars = np.ones(self.net.N_particles)
        self.dvars = np.zeros(self.net.N_particles)

        # covariances
        self.covs = np.zeros((self.net.N_particles, 6))
        self.dcovs = np.zeros((self.net.N_particles, 6))

    def get_cov(self, i, j):
        """
        Manages covariance entries. Return correpsonding covariance to islands i and j.
        """
        table_index = np.where(self.neighbour_table[i] == j)[0]
        if table_index.shape[0] == 0:
            return 0

        return self.covs[i, table_index[0]]

    def set_cov(self, i, j, value):
        """
        Sets the covariance corresponding to islands i and j.
        """
        table_index = np.where(self.neighbour_table[i] == j)[0]
        if table_index.shape[0] == 1:
            self.covs[i, table_index[0]] = value

        table_index = np.where(self.neighbour_table[j] == i)[0]
        if table_index.shape[0] == 1:
            self.covs[j, table_index[0]] = value


    def get_dcov(self, i, j):
        """
        Manages the change in covariances. Returns the change in covariance for islands i and j.
        """
        table_index = np.where(self.neighbour_table[i] == j)[0]
        if table_index.shape[0] == 0:
            return 0

        return self.dcovs[i, table_index[0]]

    def set_dcov(self, i, j, value):
        """
        Sets the change in covariance for islands i and j.
        """
        table_index = np.where(self.neighbour_table[i] == j)[0]
        if table_index.shape[0] == 1:
            self.dcovs[i, table_index[0]] = value

        table_index = np.where(self.neighbour_table[j] == i)[0]
        if table_index.shape[0] == 1:
            self.dcovs[j, table_index[0]] = value

    def calc_effective_states_2D(self, i, j):
        """
        Returns the mean state, broadcasted to the 2D-phase space dimensions and replaces occupation numbers of islands i and j
        with n and m phase-space values for the 2D-gaussian distribution respectively.
        """
        phase_space = self.g2.phase_space
        states = np.repeat(np.expand_dims(self.means, axis = [0, 1]), phase_space.shape[0], axis = 0)
        states = np.repeat(states, phase_space.shape[1], axis = 1)

        states[:,:,i] = phase_space[:,:,0]
        states[:,:,j] = phase_space[:,:,1]
        
        return states
    
    def calc_effective_states_1D(self, i):
        """
        Reeturns the mean state, broadcasted to 1D-phase space dimensons and replaces occupation numbers of island i with the
        one dimensional phase space.
        """
        phase_space = self.g1.phase_space
        states = np.expand_dims(self.means, axis = 0)
        states = np.repeat(states, phase_space.shape[0], axis = 0)
        states[:, i] = phase_space

        return states
    
    #############################################################################################
    ########################## Phase Space Functions ############################################
    #############################################################################################

    ######## Currents between islands.
    ######## I_{ij} over 2D-phase-space
    ########

    def calc_R_island(self, i, j):
        """
        Calculates the rates from island i to island j for effective states suitable for 2D-gaussian phase-space.
        Thus i corresponding to the first, and j corresponding to the second component of the distribution.
        """
        states = self.calc_effective_states_2D(i, j)
        rates = self.net.calc_rate_island(states, i, j)
        return rates

    def calc_R_island_inv(self, i, j):
        """
        Calculates the rates from island j to island i for effective states suitable for 2D-gaussian phase-space.
        Thus i corresponding to the first, and j corresponding to the second component of the distribution.
        """
        states = self.calc_effective_states_2D(i, j)
        rates = self.net.calc_rate_island(states, j, i)
        return rates
    
    ######## Currents between particles and electrodes.
    ######## I_{ei} over 1D-phase-space
    ########

    def calc_R_from_electrode(self, electrode_index):
        """
        For a given electrode index, calculates the rates to the attached particle over 1D-phase space.
        """
        island_index = self.island_indices[electrode_index]
        states = self.calc_effective_states_1D(island_index)
        rates = self.net.calc_rate_from_electrode(states, electrode_index)
        return rates

    def calc_R_to_electrode(self, electrode_index):
        """
        For a given electrode index, calculates the rates toward the attached electrode over 1D-phase space.
        """
        island_index = self.island_indices[electrode_index]
        states = self.calc_effective_states_1D(island_index)
        rates = self.net.calc_rate_to_electrode(states, electrode_index)
        return rates
    
    ######## The occupation number times the current from the attached electrode.
    ######## n_i I_{ei} over 1D-phase-space
    ########

    def calc_nR_to_electrode(self, electrode_index):
        """
        Calculates the 1D-phase-space times the electrode rates towards the attached electrode, specified by electrode_index
        """
        phase_space = self.g1.phase_space
        rates = self.calc_R_to_electrode(electrode_index)
        values = rates * phase_space 
        return values

    def calc_nR_from_electrode(self, electrode_index):
        """
        Calculates the 1D-phase-space times the electrode rates towards the attached nanoparticle, specified by electrode_index.
        """
        phase_space = self.g1.phase_space
        rates = self.calc_R_from_electrode(electrode_index)
        values = rates * phase_space 
        return values
    
    ######## The occupation numbers of the first 2D-distribution components times the electrode rates from or towards 
    ######## the island connected to an electrode, specified by electrode index.
    ######## n_i I_{ej} over 2D-phase-space
    ######## electrode index is corresponding to island j.

    def calc_nR_from_electrode_2(self, i, electrode_index):
        """
        Over 2D-phase-space, calculates the state of the first component i times the rates from an electrode
        to an attached nanoparticle, specified by electrode_index, with occupation numbers of the second distribution component.
        """
        phase_space = self.g2.phase_space
        island_index = self.island_indices[electrode_index]

        states = self.calc_effective_states_2D(i, island_index)
        rates = self.net.calc_rate_from_electrode(states, electrode_index)

        return rates * phase_space[:,:,0]

    def calc_nR_to_electrode_2(self, i, electrode_index):
        """
        Over 2D-phase-space, calculates the state of the first component i times the rates towards an electrode
        of an attached nanoparticle, specified by electrode_index, with occupation numbers of the second distribution component.
        """
        phase_space = self.g2.phase_space
        island_index = self.island_indices[electrode_index]

        states = self.calc_effective_states_2D(i, island_index)
        rates = self.net.calc_rate_to_electrode(states, electrode_index)

        return rates * phase_space[:,:,0]
    
    ######## Over 2D-phase-space, calculates the currents from island i to j times the occupation numbers
    ######## of j. island i's occupation numbers are the first component of the 2D-distribution.
    ######## n_j I_{ij}

    def calc_nR_island(self, i, j):
        """
        Over 2D-phase space, calculates the rate from i to j times the occupation numbers of j.
        """
        phase_space = self.g2.phase_space
        rates = self.calc_R_island(i, j)
        values = rates * phase_space[:,:,1] 
        return values

    def calc_nR_island_inv(self, i, j):
        """
        Over 2D-phase-space, calculates the rate from j to i times the occupation numbers of j.
        """
        phase_space = self.g2.phase_space
        rates = self.calc_R_island_inv(i, j)
        values = rates * phase_space[:,:,1] 
        return values
            
    ######## Over 2D-phase space, calculates the currents from island i to j times the occupation numbers
    ######## of i. i and j are the first and second component of the 2D-distribution, repsectively.
    ######## n_i I_{ij}
 
    def calc_nR_island_alt(self, i, j):
        """
        On 2D-phase-space, calculates the rates from particle i to j times the occupation numbers of islands i.
        i and j are the first and second component of the 2D-distribution.
        """
        phase_space = self.g2.phase_space
        rates = self.calc_R_island(i, j)
        values = rates * phase_space[:,:,0] 
        return values

    def calc_nR_island_inv_alt(self, i, j):
        """
        On 2D-phase_space, calculates the rates from particle j to i times the occupation number of island i.
        i and j correspond to the first and second component of the distribution.
        """
        phase_space = self.g2.phase_space
        rates = self.calc_R_island_inv(i, j)
        values = rates * phase_space[:,:,0] 
        return values
    
    ###############################################################################
    ###################### Expectation values #####################################
    ###############################################################################
    
    def calc_deltas(self):
        """
        For the current means, variances, and covariances, calculates their time derivatives under approximations.
        They are stored in class members dmeans, dvars, dcovs.
        """

        # reset deltas
        self.dmeans = np.zeros(self.net.N_particles)
        self.dvars = np.zeros(self.net.N_particles)
        self.dcovs = np.zeros((self.net.N_particles, 6))

        # islands
        l_R = np.zeros(self.net.N_particles)
        r_R = np.zeros(self.net.N_particles)
        l_nR = np.zeros(self.net.N_particles)
        r_nR = np.zeros(self.net.N_particles)

        for i in range(self.net.N_particles):
            for j in self.neighbour_table[i]:
                if not j == -1: # all neighbour relations
                    probs = self.g2.calc_prob(self.means[j], self.means[i], self.vars[j], self.vars[i], self.get_cov(i, j))
                    l_R[i] += np.sum(probs * self.calc_R_island(j, i))
                    r_R[i] += np.sum(probs * self.calc_R_island_inv(j, i))

                    l_nR[i] += np.sum(probs * self.calc_nR_island(j, i))
                    r_nR[i] += np.sum(probs * self.calc_nR_island_inv(j, i))

        # island results
        I_islands = l_R - r_R
        I_dag_islands = l_R + r_R
        nI_islands = l_nR - r_nR

        # electrodes
        l_R_electrodes = np.zeros(self.net.N_particles)
        r_R_electrodes = np.zeros(self.net.N_particles)
        l_nR_electrodes = np.zeros(self.net.N_particles)
        r_nR_electrodes = np.zeros(self.net.N_particles)

        for electrode_index, i in enumerate(self.island_indices):
            probs = self.g1.calc_prob(self.means[i], self.vars[i])
            l_R_electrodes[i] += np.sum(probs * self.calc_R_from_electrode(electrode_index)) 
            l_nR_electrodes[i] += np.sum(probs * self.calc_nR_from_electrode(electrode_index))

            r_R_electrodes[i] += np.sum(probs * self.calc_R_to_electrode(electrode_index)) 
            r_nR_electrodes[i] += np.sum(probs * self.calc_nR_to_electrode(electrode_index))

        # electrode results
        I_electrodes = l_R_electrodes - r_R_electrodes
        I_dag_electrodes = l_R_electrodes + r_R_electrodes
        nI_electrodes = l_nR_electrodes - r_nR_electrodes

        # total
        I = I_islands + I_electrodes
        I_dag = I_dag_islands + I_dag_electrodes
        nI = nI_islands + nI_electrodes
        
        self.dmeans = I
        self.dvars = (2 * nI + I_dag) - 2 * self.means * I

        # covariances
        if not self.include_covs:
            return
        
        for i in range(self.net.N_particles):
            for j in self.neighbour_table[i]:
                if not j == -1:
                    probs = self.g2.calc_prob(self.means[i], self.means[j], self.vars[i], self.vars[j], self.get_cov(i, j))
                    probs2 = self.g2.calc_prob(self.means[j], self.means[i], self.vars[j], self.vars[i], self.get_cov(i, j))


                    # < ni Ij >
                    dcov = self.means[i] * I_islands[j]
                    dcov -= self.means[i] * np.sum(probs * (self.calc_R_island(i, j) - self.calc_R_island_inv(i, j)))
                    dcov += np.sum(probs * (self.calc_nR_island_alt(i, j) - self.calc_nR_island_inv_alt(i, j)))

                    electrode_index = np.where(self.island_indices == j)[0]
                    if electrode_index.shape[0] == 1:
                        dcov += np.sum(probs * (self.calc_nR_from_electrode_2(i, electrode_index[0]) - self.calc_nR_to_electrode_2(i, electrode_index[0])))


                    # < nj Ii >
                    dcov += self.means[j] * I_islands[i]
                    dcov -= self.means[j] * np.sum(probs2 * (self.calc_R_island(j, i) - self.calc_R_island_inv(j, i)))
                    dcov += np.sum(probs2 * (self.calc_nR_island_alt(j, i) - self.calc_nR_island_inv_alt(j, i)))

                    electrode_index = np.where(self.island_indices == i)[0]
                    if electrode_index.shape[0] == 1:
                        dcov += np.sum(probs2 * (self.calc_nR_from_electrode_2(j, electrode_index[0]) - self.calc_nR_to_electrode_2(j, electrode_index[0])))

                    # < I^dag_ij >
                    dcov -= np.sum(probs * (self.calc_R_island(i, j) + self.calc_R_island_inv(i, j)))

                    self.set_dcov(i, j, dcov - self.dmeans[i] * self.means[j] - self.means[i] * self.dmeans[j])

    def calc_expected_electrode_current(self, electrode_index):
        """
        For the current stored moments, calculates the expected output current towards the electrode, specified by electrode index.
        """
        i = self.island_indices[electrode_index]
        probs = self.g1.calc_prob(self.means[i], self.vars[i])
        return -np.sum(probs * (self.calc_R_from_electrode(electrode_index) - self.calc_R_to_electrode(electrode_index))) * CONST.electron_charge

    def solve(self, dt = 0.05, N = 60, verbose = False, reset = False):
        """
        Integrates the moments in time.

        Parameters:
            dt      : integration step
            N       : number of iterations
            verbose : whether to print convergence at the end
            rest    : whether to reset all moments before at the beginning, for example after change of voltages.
        """
        if reset:
            self.means = np.zeros(self.net.N_particles)
            self.vars = np.ones(self.net.N_particles)
            self.covs = np.zeros((self.net.N_particles, 6))


        for i in range(N):
            self.calc_deltas()

            self.means += dt * self.dmeans
            self.vars += dt * self.dvars
            self.covs += dt * self.dcovs

            decimals = self.means - np.floor(self.means)
            self.vars = np.where(self.vars < decimals * (1 - decimals), decimals * (1 - decimals), self.vars)

        if verbose:
            self.calc_deltas()
            print("convergence mean:", np.abs(self.dmeans).max())
            print("convergence variances:", np.abs(self.dvars).max())
            print("convergence covariances:", np.abs(self.dcovs).max())

    def calc_moments(self):
        """
        Recalculates and returns all moments, based on the current moments. This can reduce error, since
        the moment stored as class members are not directly calculated with the 1D or 2D gaussian distributions.
        """
        means_ = np.zeros(self.net.N_particles)
        vars_ = np.zeros(self.net.N_particles)
        covs_ = np.zeros((self.net.N_particles, 6))

        for i in range(self.net.N_particles):
            probs = self.g1.calc_prob(self.means[i], self.vars[i])
            means_[i] = np.sum(self.g1.phase_space * probs)
            vars_[i] = np.sum(self.g1.phase_space ** 2 * probs) - means_[i] ** 2


            if self.include_covs:
                for index, j in enumerate(self.neighbour_table[i]):
                    if not j == -1:
                        probs = self.g2.calc_prob(self.means[i], self.means[j], self.vars[i], self.vars[j], self.get_cov(i, j))
                        moment = np.sum(self.g2.phase_space[:,:,0] * self.g2.phase_space[:,:,1] * probs) - self.means[i] * self.means[j]
                        covs_[i, index] = moment

        return means_, vars_, covs_
