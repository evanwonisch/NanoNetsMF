from module.base.network import Network

import numpy as np

# Creating network object

net = Network(2,2,1,[[0,0,0]])

print(net.dq)
print(net.inv_cap_mat)
print(net.electrode_voltages)
print(net.gate_voltage)
print(net.calc_free_energy(np.ones(net.N_particles)))

net.set_voltage_config([1], 0)
print(net.dq)
print(net.calc_free_energy(np.ones(net.N_particles)))

