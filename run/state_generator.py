import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join

from module.base.network import Network
from module.simulation.meanfield import MeanField
import module.components.CONST as CONST

## Comparison of states and currents for larger systems
net_sizes = np.arange(2, 11)
nets = [Network(size, size, 1, [[0,0,0], [size - 1, 0, 0],[0, size - 1, 0],[size - 1, size - 1, 0]]) for size in net_sizes]
mfs = [MeanField(net) for net in nets]
voltage_configs = np.loadtxt("data/large_sys/voltage_configs.csv")


for i in range(len(net_sizes)): # net sizes

    print("doing size", net_sizes[i])

    state_array = np.zeros((100, nets[i].N_particles))
    current_array = np.zeros(100)

    for j in range(100): # voltage configs
        print(str(j) + "%")
        voltages = voltage_configs[j]
        nets[i].set_voltage_config([voltages[0], voltages[1], voltages[2], voltages[3]], voltages[4])

        eps = 1
        state = np.zeros(nets[i].N_particles)
        dt = 0.07
        n = 0
        while eps > 1e-4 and n < 1000:
            state = mfs[i].numeric_integration_solve(state, N = 5, dt = dt)
            eps = mfs[i].convergence_metric(state)
            n += 1
            if n % 10 == 0:
                dt = dt * 0.95
                if dt < 0.005:
                    dt = 0.005
                print(n, eps)

        
        state_array[j] = nets[i].calc_charge_vector(state)
        current_array[j] = -mfs[i].calc_expected_electrode_current(state, 3) * CONST.electron_charge

    print("done size:", net_sizes[i])
    np.savetxt("data/large_sys/mf_1e-4/states"+str(net_sizes[i])+"x"+str(net_sizes[i])+".csv", state_array)
    np.savetxt("data/large_sys/mf_1e-4/currents"+str(net_sizes[i])+"x"+str(net_sizes[i])+".csv", current_array)


