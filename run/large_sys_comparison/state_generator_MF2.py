import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join

from module.base.network import Network
from module.simulation.meanfield import MeanField
from module.simulation.quick_meanfield2 import QuickMeanField2
from module.simulation.meanfield2 import MeanField2
import module.components.CONST as CONST

## Comparison of states and currents for larger systems
net_sizes = np.arange(2, 11)
nets = [Network(size, size, 1, [[0,0,0], [size - 1, 0, 0],[0, size - 1, 0],[size - 1, size - 1, 0]]) for size in net_sizes]
mfs = [MeanField2(net, include_covs = False) for net in nets]
for mf in mfs:
    mf.ADAM_solve(N = 0)
voltage_configs = np.loadtxt("data/large_sys/voltage_configs.csv")


for i in range(3, 9): # net sizes

    print("doing size", net_sizes[i])

    means_array = np.zeros((100, nets[i].N_particles))
    vars_array = np.zeros((100, nets[i].N_particles))
    #means_array = np.loadtxt("data/large_sys/mf2/entropy_means"+str(net_sizes[i])+"x"+str(net_sizes[i])+".csv")
    #vars_array = np.loadtxt("data/large_sys/mf2/entropy_vars"+str(net_sizes[i])+"x"+str(net_sizes[i])+".csv")

    conv_means_array = np.zeros(100)
    conv_vars_array = np.zeros(100)
    current_array = np.zeros(100)

    for j in range(100): # voltage configs
        print(str(j) + "%")
        voltages = voltage_configs[j]
        nets[i].set_voltage_config([voltages[0], voltages[1], voltages[2], voltages[3]], voltages[4])

        # mfs[i].ADAM_solve(N = 180, learning_rate = 0.1, reset=True)
        # mfs[i].ADAM_solve(N = 50, learning_rate = 0.005, reset=False, verbose = True)

        mfs[i].means = means_array[j]
        mfs[i].vars = vars_array[j]
        #mfs[i].ADAM_solve(verbose = True, N = 0, reset = False)
        mfs[i].ADAM_solve(verbose = True, N = 100, reset = False)

        conv_mean, conv_var, _ = mfs[i].ADAM_convergence_metric()

        means_array[j] = mfs[i].means
        vars_array[j] = mfs[i].vars
        conv_means_array[j] = conv_mean
        conv_vars_array[j] = conv_var
        current_array[j] = -mfs[i].calc_expected_electrode_current(3)


    print("done size:", net_sizes[i])
    np.savetxt("data/large_sys/full_MF2/means"+str(net_sizes[i])+"x"+str(net_sizes[i])+".csv", means_array)
    np.savetxt("data/large_sys/full_MF2/vars"+str(net_sizes[i])+"x"+str(net_sizes[i])+".csv", vars_array)
    np.savetxt("data/large_sys/full_MF2/conv_means"+str(net_sizes[i])+"x"+str(net_sizes[i])+".csv", conv_means_array)
    np.savetxt("data/large_sys/full_MF2/conv_vars"+str(net_sizes[i])+"x"+str(net_sizes[i])+".csv", conv_vars_array)
    np.savetxt("data/large_sys/full_MF2/currents"+str(net_sizes[i])+"x"+str(net_sizes[i])+".csv", current_array)
