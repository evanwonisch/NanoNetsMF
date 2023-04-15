import numpy as np
from module.base.network import Network
from module.simulation.quick_meanfield2 import QuickMeanField2
from module.simulation.meanfield import MeanField
from module.components.Adam import Adam

voltage_configs = np.loadtxt("data/cont_comparison/voltage_configs.csv")
net = Network(4, 4, 1, [[0,0,0],[3,0,0],[0,3,0],[3,3,0]])

mf_means = np.loadtxt("data/cont_comparison/mf/mf_means.csv")

for i in range(200):
    print(i/200 * 100, "%")

    net.set_voltage_config(voltage_configs[i], 0)
    mf = MeanField(net)

    mf_means[i] = mf.numeric_integration_solve(mf_means[i], verbose = True, N = 500, dt = 0.005)

    np.savetxt("data/cont_comparison/mf/mf_means.csv", mf_means)
