import numpy as np

from module.simulation.meanfield import MeanField as MeanField
from module.base.network import Network as Network

net = Network(2,2,1,[[0,0,0],[1,1,0]])
mf = MeanField(net)
island_index = np.array([[0,1],[2,3]])
microstates = np.array([[False,True],[False,True]])
occ = np.array([[[16.17410522, 18.49276232, 23.45403068, 11.1422785 ],
  [10.16361653, 28.2023101,   5.33504647, 25.85101559]],

 [[ 9.14831688, 19.33463189, 21.3374962,  12.9502747 ],
  [17.87378285, 14.48118671,  6.5975802,  15.22911196]]])
prob = mf.calc_probability(occ, island_index, microstates)
print(occ)
print(prob)