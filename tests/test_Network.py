from module.base.network import Network

import numpy as np

# Crearing network object

net = Network(2,2,1,[[0,0,0]],[],[])

print(net.dq)
print(net.inv_cap_mat)
print(net.voltage_config)
print(net.calc_free_energy(np.ones(net.N_particles)))

net.set_voltage_config([1],[],[], 0)
print(net.dq)
print(net.calc_free_energy(np.ones(net.N_particles)))
