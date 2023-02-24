import numpy as np
from module.base.network import Network

N = 4
net = Network(2,2,1,[],[],[])

# with broadcasting
occupations = np.array([[[1,0,0,0],[0,0,0,0]],[[0,0,0,0],[1,1,1,1]]])
print(net.calc_free_energy(occupations))

# without broadcasting
occupations = np.array([1,0,0,0])
print(net.calc_free_energy(occupations))

occupations = np.array([1,1,1,1])
print(net.calc_free_energy(occupations))


