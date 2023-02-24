import numpy as np

import module.base.capacitance
import module.components.utils as utils
import module.components.CONST as CONST


class Network:
    def __init__(self, Nx, Ny, Nz, electrode_pos):
        """
        Creates a network topology.

        Nx, Ny, Nz : int
            dimensions of the network

        electrode_pos : list
            index positions as a list of 3-tuples representing the index coordinates
        """

        # Defines Network size
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz

        self.N_particles = Nx * Ny * Nz

        # Specifies electrode attachment
        self.electrode_pos = electrode_pos

        # Calculates the inverse capacity matrix
        self.capacities = module.base.capacitance.build_network(Nx, Ny, Nz, electrode_pos)
        self.inv_cap_mat = np.linalg.inv(self.capacities["cap_mat"])

        # Sets all applied voltages to zero and thus calculates dq
        self.dq = None                  # represents induced charges by voltage of electrodes
        self.electrode_voltages = None
        self.gate_voltage = None
        self.set_voltage_config()


    def get_index(self, index_x, index_y, index_z):
        """
        Transforms nanoparitcle-3D-index-coordinates to linear indices
        """
        return index_x + self.Nx * index_y + self.Nx * self.Ny * index_z


    def set_voltage_config(self, electrode_voltages = None, gate_voltage = None):
        """
        Sets all voltages applied to the network. Sets them to zero if nothing is given.
        electrode_voltages      :   list of voltages for each electrode electrode
        gate_voltage        :   voltage of the gate
        """

    	#for no input: set all voltages to zero
        if electrode_voltages == None:
            self.electrode_voltages = np.zeros(len(self.electrode_pos))
            self.gate_voltage = 0
            self.calc_induced_charges()
            return

        #for inputs: check dimensions and apply
        assert electrode_voltages != None, "Please specify electrode voltages"
        assert gate_voltage != None, "Please specify gate voltage"

        assert len(electrode_voltages) == len(self.electrode_pos), "Wrong number of electrode voltages"

        self.electrode_voltages = electrode_voltages
        self.gate_voltage = gate_voltage

        self.calc_induced_charges()

    def calc_induced_charges(self):
        """
        This calculates and stores what needs to be added to the elementary system-charges
        to calculate the charge vector. The electrode charges will be added only to
        particles whose positions are specified for the electrodes.
        The gate voltage will be apllied to all particles concerning the self-capacitance.

        This method will run automatically upon modification of the volatge configuration
        """

        dq = np.zeros(shape = self.N_particles)

        c_electrode = self.capacities['lead']    #Kapazität zu den Elektroden
        c_gate = self.capacities["self"]         #Kapazität zum Gate

        ## electrodes
        for i, (x,y,z) in enumerate(self.electrode_pos):
            list_index = self.get_index(x,y,z)
            dq[list_index] += self.electrode_voltages[i] * c_electrode

        ## gate
        dq += c_gate * self.gate_voltage

        self.dq = dq

    def calc_free_energy(self, occupation_numbers):
        """
        Calculates the total free energy
        occupation_unmbers   :   number of electrons on each island

        Thereby opccupation number can have an arbitrary leading shape
        as long as the last dimenion is of size N_particles, thus
        shape = (..., N_particle)

        The free energies are returned in the same format expect for the missing last dimension
        """
        assert occupation_numbers.shape[-1] == self.N_particles, "Wrong number of particles"

        # calculate free energy
        q = module.components.CONST.electron_charge * occupation_numbers + self.dq

        q_ = np.matmul(self.inv_cap_mat, np.expand_dims(q, axis = -1))
        q_ = np.squeeze(q_, axis = -1)


        F = 0.5 * np.sum(q * q_, axis = -1)

        return F
    
    def calc_potentials(self, occupation_numbers):
        """
        Calculates the potentials for each island
        occupation_numbers   :   number of electrons on each island

        Thereby opccupation number can have an arbitrary leading shape
        as long as the last dimenion is of size N_particles, thus
        shape = (..., N_particle)

        The potentials are returned in the same format
        """
        assert occupation_numbers.shape[-1] == self.N_particles, "Wrong number of particles"

        # calculate free energy
        q = module.components.CONST.electron_charge * occupation_numbers + self.dq

        q_ = np.matmul(self.inv_cap_mat, np.expand_dims(q, axis = -1))
        q_ = np.squeeze(q_, axis = -1)

        return q_
    
    def calc_rate_internal(self, dF):
        """
        Calculates the tunnel-rate for a given difference in free energy dF where dF = F_final - F_initial

        Returns: tunnel rate
        """

        eps = 10e-16             # dF = 0 excluded
        dF = np.where(dF == 0, eps, dF)

        # clipping for safety
        exponential = np.clip(dF/ CONST.kb / CONST.temperature, a_min = None, a_max = 200)

        rate = -dF / CONST.electron_charge ** 2 / CONST.tunnel_resistance / (1 - np.exp(exponential))

        return rate
    
    def calc_rate_island(self, occupation_numbers, alpha, beta):
        """
        Calculates the tunnel rates for jumps between islands.
        
        Here a jump from island with index alpha to island with index beta is considered, given the current
        occupation numbers of the system. This can be done in parallel by prepending additional dimensions to
        occupation_numbers, alpha and beta:

        shape(occupation_numbers)   = (..., N_particles)
        shape(alpha)                = (...)
        shape(beta)                 = (...)
        """

        # expand alpha and beta to categoricals which can be used to modify the charge vector
        exp_a = utils.to_categorical(alpha, self.N_particles)
        exp_b = utils.to_categorical(beta, self.N_particles)

        # calculate the difference in free energy and then the tunnel rates
        F1 = self.calc_free_energy(occupation_numbers)
        F2 = self.calc_free_energy(occupation_numbers - exp_a + exp_b)
        dF = F2 - F1

        return self.calc_rate(dF)