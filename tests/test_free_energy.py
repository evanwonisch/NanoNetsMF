import numpy as np

from module.base.network import Network

net = Network(5,5,1,[[0,0,0],[4,4,0]])
net.set_voltage_config([0.1,-0.1],0)

n = np.random.randint(-10,10, size = (10,25))
F = net.calc_free_energy(n)

np.savetxt("./out/n.csv", n)
np.savetxt("./out/F.csv", F)

print(F)