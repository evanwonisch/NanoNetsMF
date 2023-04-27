import numpy as np
from module.base.network import Network
from module.simulation.quick_meanfield2 import QuickMeanField2
from module.simulation.meanfield import MeanField
from module.components.Adam import Adam

voltage_configs = np.loadtxt("data/cont_comparison/voltage_configs.csv")
net = Network(4,4,1,[[0,0,0],[3,0,0],[0,3,0],[3,3,0]])

mf_means = np.loadtxt("data/cont_comparison/mf2/entropy_mf2_means.csv")
mf_vars = np.loadtxt("data/cont_comparison/mf2/entropy_mf2_vars.csv")

mf_dmeans = np.zeros((200, 16))
mf_dvars = np.zeros((200, 16))

mf_convs_mean = np.loadtxt("data/cont_comparison/mf2/entropy_mf2_conv_means.csv")
mf_convs_var = np.loadtxt("data/cont_comparison/mf2/entropy_mf2_conv_vars.csv")


for i in range(150, 200):
    print(i)

    net.set_voltage_config(voltage_configs[i], 0)
    mf = QuickMeanField2(net)
    mf.ADAM_solve(N = 0)

    # mf.means = np.copy(mf_means[i])
    # mf.vars = np.copy(mf_vars[i])

    for _ in range(6):
        # 0.01 for high values of <n>
        # 0.1 for low values
        # mf.ADAM_solve(N = 15, learning_rate = (0.01, 1.05), reset = False, verbose = True)
        mf.numeric_integration_solve(N = 20, verbose = True, reset = False, dt = 0.01)




    # save
    mf_means[i] = np.copy(mf.means)
    mf_vars[i] = np.copy(mf.vars)

    dmeans, dvars = mf.calc_derivatives()
    mf_dmeans[i] = dmeans
    mf_dvars[i] = dvars

    mf_convs_mean[i] = mf.convergence_metric()[0]
    mf_convs_var[i] = mf.convergence_metric()[1]

    np.savetxt("data/cont_comparison/mf2/entropy_mf2_means.csv", mf_means)
    np.savetxt("data/cont_comparison/mf2/entropy_mf2_vars.csv", mf_vars)

    np.savetxt("data/cont_comparison/mf2/entropy_mf2_dmeans.csv", mf_dmeans)
    np.savetxt("data/cont_comparison/mf2/entropy_mf2_dvars.csv", mf_dvars)

    np.savetxt("data/cont_comparison/mf2/entropy_mf2_conv_means.csv", mf_convs_mean)
    np.savetxt("data/cont_comparison/mf2/entropy_mf2_conv_vars.csv", mf_convs_var)