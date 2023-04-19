import numpy as np
from module.base.network import Network
from module.simulation.quick_meanfield2 import QuickMeanField2
from module.simulation.meanfield import MeanField
from module.components.Adam import Adam

voltage_configs = np.loadtxt("data/cont_comparison/voltage_configs.csv")
net = Network(4,4,1,[[0,0,0],[3,0,0],[0,3,0],[3,3,0]])

mf_means = np.loadtxt("data/cont_comparison/mf2/entropy_mf2_means.csv")
mf_vars = np.loadtxt("data/cont_comparison/mf2/entropy_mf2_vars.csv")


for i in range(0, 200):
    print(i / 200 * 100, "%")

    net.set_voltage_config(voltage_configs[i], 0)
    mf = QuickMeanField2(net)
    mf.ADAM_solve(N = 0)

    mf.means = mf_means[i]
    mf.vars = mf_vars[i]

    # for _ in range(15):
    #     mf.ADAM_solve(N = 10, learning_rate = 0.005, reset = False, verbose = True)
    mf.numeric_integration_solve(N = 40, verbose = True, reset = False, dt = 0.001)

    mf_means[i] = np.copy(mf.means)
    mf_vars[i] = np.copy(mf.vars)

    np.savetxt("data/cont_comparison/mf2/entropy_mf2_means.csv", mf_means)