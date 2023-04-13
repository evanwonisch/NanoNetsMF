import numpy as np
from module.base.network import Network
from module.simulation.quick_meanfield2 import QuickMeanField2

voltage_configs = np.loadtxt("data/cont_comparison/voltage_configs.csv")
print(voltage_configs.shape)
net = Network(4,4,1,[[0,0,0],[3,0,0],[0,3,0],[3,3,0]])

mf_means = np.zeros((200, 16))
mf_vars = np.zeros((200, 16))
for i in range(200):
    print(i/200 * 100, "%")

    net.set_voltage_config(voltage_configs[i,0:4], 0)
    mf = QuickMeanField2(net)

    mf.ADAM_solve(N = 100, learning_rate = 0.1, reset=True)
    mf.ADAM_solve(N = 100, learning_rate = 0.005, reset=False, verbose = True)

    conv_mean, conv_var = mf.ADAM_convergence_metric()

    mf_means[i] = mf.means
    mf_vars[i] = mf.vars

    np.savetxt("data/cont_comparison/mf2/mf_means.csv", mf_means)
    np.savetxt("data/cont_comparison/mf2/mf_vars.csv", mf_vars)