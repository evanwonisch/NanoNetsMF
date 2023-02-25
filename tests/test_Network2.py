from module.base.network import Network

import numpy as np

# Crearing network object

net = Network(2,2,1,[[0,0,0], [1,1,0]])
net.set_voltage_config([1,0], 0)

#  hopping vers output
n = np.array([1,0,0,0])
m = np.array([0,1,0,0])

F1 = net.calc_free_energy(n)
F2 = net.calc_free_energy(m)

print("dF:", F2 - F1)
print("rate:", net.calc_rate_internal(F2 - F1))

