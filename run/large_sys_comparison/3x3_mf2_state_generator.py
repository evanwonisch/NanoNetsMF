import numpy as np
from matplotlib import pyplot as plt

from module.base.network import Network
from module.simulation.meanfield2 import MeanField2
import module.components.CONST as CONST

## Comparison of states and currents for larger systems
net = Network(3,3,1,[[0,0,0],[2,0,0],[0,2,0],[2,2,0]])
mf2 = MeanField2(net)
voltage_configs = np.loadtxt("data/large_sys/sorted/voltage_configs.csv")

means_array = np.zeros((100, 9))
vars_array = np.zeros((100, 9))
covs_array = np.zeros((100, 9, 6))
currents_array = np.zeros(100)

for j in range(0, 100): # voltage configs
    print(str(j) + "%")
    voltages = voltage_configs[j]
    net.set_voltage_config([voltages[0], voltages[1], voltages[2], voltages[3]], voltages[4])
    
    mf2.solve(dt = 0.05, N = 40, verbose=True, reset=True)

    means_array[j] = mf2.means
    vars_array[j] = mf2.vars
    covs_array[j] = mf2.covs
    currents_array[j] = mf2.calc_expected_electrode_current(3)
    
    np.savetxt("data/large_sys/sorted/mf2/mf2_means3x3.csv", means_array)
    np.savetxt("data/large_sys/sorted/mf2/mf2_vars3x3.csv", vars_array)
    np.savetxt("data/large_sys/sorted/mf2/mf2_covs3x3.csv", covs_array.reshape((100, -1)))
    np.savetxt("data/large_sys/sorted/mf2/mf2_currents3x3.csv", currents_array)