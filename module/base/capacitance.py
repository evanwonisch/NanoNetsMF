# This module resulted from the C++ implementation of KMC methods for nano-particle networks 
# and builds the interface between the mean-field method and Jonas Mensings code.
# For this to work, it is required to compile and install a C++ library and convert it to python.
# This gives rise to the package "np_network".

##########################################################################

import np_network as np_net

import numpy as np
import pandas as pd
from typing import Tuple, List

from module.components import CONST

##########################################################################
################# Added by Evan ##########################################
##########################################################################

def build_network(N_x, N_y, N_z, electrode_pos):
    """
    Calculates the capacities of the network.
    
    Parameters:

    Nx, Ny, Nz : int
        dimensions of the network

    electrode_pos : list
        index positions as a list of 3-tuples representing the index coordinates

    Returns: dictionary
        "node" : node capacity
        "lead" : lead capacity between nodes and electrodes
        "self" : self capacity wrt. the Si-substrate

    """

    # Tests for dimensions

    assert N_x > 0, "invalid network dimensions"
    assert N_y > 0, "invalid network dimensions"
    assert N_z > 0, "invalid network dimensions"

    # Tests for electrode positions
    for pos in electrode_pos:
        assert len(pos) == 3, "no valid electrode position"

        assert pos[0] >= 0, "no valid electrode position"
        assert pos[0] < N_x, "no valid electrode position"

        assert pos[1] >= 0, "no valid electrode position"
        assert pos[1] < N_y, "no valid electrode position"

        assert pos[2] >= 0, "no valid electrode position"
        assert pos[2] < N_z, "no valid electrode position"

    # Number of inputs, Outputs, Controls
    N_i, N_o, N_c   = len(electrode_pos), 0, 0          # removed different types of electrodes

    # Permittivity Values
    eps_r = CONST.eps_r
    eps_s = CONST.eps_s

    # Nanoparticle radius and spacing in between
    r_NP = CONST.r_NP
    d_NP = CONST.d_NP


    # Init dictonaries
    capacitance_values                  = init_capacitance_values(eps_r=eps_r, eps_s=eps_s, radius=r_NP, distance=d_NP)
    network_info, electrode_positions   = init_cubic_packed_dicts(N_x=N_x, N_y=N_y, N_z=N_z, N_inputs=N_i, N_outputs=N_o, N_controls=N_c,
                                            input_positions=electrode_pos, output_positions=[], control_positions=[])

    # Init network topology class and electrostatic class
    _, electrostatics       = electrostatics_cubic_network(network_info=network_info, electrode_positions=electrode_positions,
                                            capacitance_values=capacitance_values)
    
    cap_mat = electrostatics.capacitance_matrix

    capacitance_values.update({"cap_mat": cap_mat})

    return capacitance_values


##########################################################################
#################### Added by Jonas ######################################
##########################################################################



def init_cubic_packed_dicts(N_x : int, N_y : int, N_z : int, N_inputs : int, N_outputs : int, N_controls : int,
    input_positions : List[List[int]], output_positions : List[List[int]], control_positions : List[List[int]]) -> Tuple[dict,dict]:
    """
    Function returns Network Info and Electrode Position Dictonaries
    N_x         :   Number of Nanoparticles in x-direction
    N_y         :   Number of Nanoparticles in y-direction
    N_z         :   Number of Nanoparticles in z-direction
    N_inputs    :   Number of Input Electrodes
    N_outputs   :   Number of Output Electrodes
    N_controls  :   Number of Control Electrodes
    """

    if (len(input_positions) != N_inputs):
        raise ValueError("Length of input_positions list does not match N_inputs!")
    if (len(output_positions) != N_outputs):
        raise ValueError("Length of output_positions list does not match N_outputs!")
    if (len(control_positions) != N_controls): 
        raise ValueError("Length of control_positions list does not match N_controls!")

    network_info = {
        "N_x"           : N_x,
        "N_y"           : N_y,
        "N_z"           : N_z,
        "N_particles"   : N_x*N_y*N_z,
        "N_inputs"      : N_inputs,
        "N_outputs"     : N_outputs,
        "N_controls"    : N_controls
    }

    electrode_positions = {
        "input_positions"   : input_positions,
        "output_positions"  : output_positions,
        "control_positions" : control_positions
    }

    return network_info, electrode_positions

def init_capacitance_values(eps_r : float, eps_s : float, radius  : float, distance  : float) -> dict:
    """
    Function returns Capacitance Dictonary
    eps_s       :   Permittivity of oxide layer - self capacitance calculation
    eps_r       :   Permittivity of organic molecule - mutual capacitance calculation
    radius      :   Nanoparticle radius
    distance    :   Distance between two Nanoparticle / Thickness of organic Molecule
    """

    capacitance_values = {
        "node" : np_net.mutal_capacitance_adjacent_spheres_approx(eps_r, radius, distance),
        "lead" : np_net.mutal_capacitance_adjacent_spheres_approx(eps_r, radius, distance),
        "self" : np_net.self_capacitance_sphere(eps_s, radius)
    }

    return capacitance_values

def init_const_resistances(R : float, network_info : dict) -> dict:
    """
    Function returns dictonary of constant resistances
    R               :   Resistance Value
    network_info    :   Network info dictonary 
    """

    resistances = {
        "np"        : [R]*network_info["N_particles"],
        "input"     : [R]*network_info["N_inputs"],
        "output"    : [R]*network_info["N_outputs"],
        "control"   : [R]*network_info["N_controls"]
    }

    return resistances

def electrostatics_cubic_network(network_info : dict, electrode_positions : dict, capacitance_values : dict, N_states=1, thread_number=0):
    """
    Function returns Topology and Electrostatic Class for a cubic packed Network 
    network_info            :   Network Topology Dictonary
    electrode_positions     :   Electrode Position Dictonary
    capacitance_values      :   Capacitance Dictonary
    N_states                :   Number of possible States per Input
    thread_number           :   For Multithreading -> current thread number
    """

    # Helper List with boolean 0 or 1, if i-th NP is affected by Gate Voltage
    gate_nps = [1]*network_info["N_particles"]

    # Initialize a cubic packed Network of Nanoparticles
    network_cubic = np_net.network_topology(thread_number)
    network_cubic.build_cubic_network(network_info, electrode_positions)

    # Initialize electrostatic properties
    electrostatics = np_net.initial_electrostatic(thread_number)
    electrostatics.init_electrodes(network_info, N_states, network_info["N_particles"])
    electrostatics.get_capacitance_matrix(network_cubic.net_topology, capacitance_values, gate_nps)
    electrostatics.get_transition_array_w_currs(network_cubic.net_topology)

    return network_cubic, electrostatics

def run_simulation(network_class : object, electrostatics : object, voltage_values : np.array, network_info : dict,
    resistances : dict, simulation_info : dict, capacitance_values : dict, T_val : float, folder : str, save_state=0)->None:
    """
    network_class           :   Topology class
    electrostatics          :   Electrostatics class
    voltage_values          :   Array containing voltage values for Inputs, Gate and Controls (in this order!)
    network_info            :   Network Topology Dictonary
    resistances             :   Dictonary of resistances for each NP
    simulation_info         :   Dictonary containing information about the simulation process
    capacitance_values      :   Capacitance dictonary
    T_val                   :   Operating Temperature
    folder                  :   save folder path
    save_state              :   If save_state != 0 save charge vector on each save_state step
    """
    
    # Number of electrodes and voltage values
    N_voltages      = len(voltage_values)
    N_inputs        = network_info["N_inputs"]
    N_controls      = network_info["N_controls"]
    input_values    = voltage_values[:,0:N_inputs]
    control_values  = voltage_values[:,N_inputs:(N_inputs+N_controls)]
    output_value    = voltage_values[:,(N_inputs+N_controls)]
    gate_value      = voltage_values[:,-1]
    gate_nps        = [1]*network_info["N_particles"]

    # if N_controls != 0:
    #     control_values  = voltage_values[:,N_inputs+1:]
    # Init KMC class and list for output values
    kmc_model   = np_net.monte_carlo(simulation_info["thread_num"])
    outputs     = []

    for i in range(N_voltages):

        # Assign Voltage Values and set state vector
        electrostatics.assign_input(input_values[i])
        electrostatics.gate_voltages = [gate_value[i]]*network_info["N_particles"]
        if N_controls != 0:
            electrostatics.control_voltages = control_values[i]
        electrostatics.output_voltages = [output_value[i]]
        electrostatics.get_charge_vector(network_class.net_topology, capacitance_values, gate_nps)

        # Set KMC Class based on Electrostatics
        kmc_model.init_model(electrostatics.input_voltages, electrostatics.control_voltages, electrostatics.output_voltages, electrostatics.gate_voltages,
            np.array(electrostatics.capacitance_matrix), np.array(electrostatics.transition_array), np.array(electrostatics.charge_vector))
        kmc_model.init_kbt(T_val)
        kmc_model.init_resistance_array(resistances)

        # Init electric current and time
        kmc_model.current[0]    = 0
        mc_time                 = 0.0

        # Run KMC and return current and standard deviation
        mc_event, total_eq_jumps    = kmc_model.reach_equilibrium_by_mean(simulation_info["p_eq"], simulation_info["max_pot_dev"])
        mc_time, jumps              = kmc_model.perform_algorithm(simulation_info["max_rel_error"], mc_time, mc_event, simulation_info["max_jumps"],
                                        save_state, folder+f'state_Nx={network_info["N_x"]}_Ny={network_info["N_y"]}_Nz={network_info["N_z"]}_Ni={network_info["N_inputs"]}_No={network_info["N_outputs"]}_Nc={network_info["N_controls"]}')
        output1, output2            = kmc_model.return_output_values()

        # Append data to output list
        output_list = []
        if N_inputs != 0:
            for input_val in input_values[i]:
                output_list.append(input_val)
        output_list.append(gate_value[i])

        if N_controls != 0:
            for control_val in control_values[i]:
                output_list.append(control_val)
        output_list.append(output_value[i])
        output_list.append(total_eq_jumps)
        output_list.append(jumps)
        output_list.append(output1*(10**(-6)))
        output_list.append(output2*(10**(-6)))
        outputs.append(output_list)
    
    # Transform data to pandas DataFrame and save
    df          = pd.DataFrame(np.array(outputs))
    col_names   = [f'I{i}' for i in range(1,N_inputs+1)]\
                + ['G']\
                + [f'C{i}' for i in range(1,N_controls+1)]\
                + ['O']\
                + ['Jumps_eq', 'Jumps'] + ['Current', 'Error']
    df.columns  = col_names
    df.to_csv(folder + f'Nx={network_info["N_x"]}_Ny={network_info["N_y"]}_Nz={network_info["N_z"]}_Ni={network_info["N_inputs"]}_No={network_info["N_outputs"]}_Nc={network_info["N_controls"]}.csv')