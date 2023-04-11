import numpy as np
from matplotlib import pyplot as plt

from module.base.network import Network
from module.simulation.meanfield2 import MeanField2
from module.simulation.meanfield import MeanField
import module.components.CONST as CONST


net = Network(4,4,1,[[0,0,0],[3,0,0],[0,3,0],[3,3,0]])
mf = MeanField(net)
mf2 = MeanField2(net, include_covs=False)

N = 200
voltages_0 = np.linspace(-0.05,0.05, num = N)
voltages_1 = np.linspace(0.06,-0.02, num = N)
voltages_2 = np.linspace(-0.04, -0.02, num = N)
voltages_3 = np.linspace(0.06, 0.03, num= N)
voltage_config = np.stack((voltages_0, voltages_1, voltages_2, voltages_3), axis = 1)
#np.savetxt("data/second_order/4x4/voltage_configs.csv", voltage_config)

means = np.zeros((N, 16))
vars = np.zeros((N, 16))
covs = np.zeros((N, 16, 6))
currents = np.zeros(N)

conv_means = np.zeros(N)
conv_vars = np.zeros(N)
conv_covs = np.zeros(N)


for i in range(N):
    print("Doing config", i + 1, "of", N)
    net.set_voltage_config([voltages_0[i], voltages_1[i], voltages_2[i], voltages_3[i]], 0)

    print("first order convergence:")
    mf2.means = mf.numeric_integration_solve(verbose = True)
    mf2.vars = np.ones(16)
    mf2.covs = np.zeros((16, 6))

    print("second order convergence:")
    mf2.solve(dt = 0.05, N = 60, verbose = True, reset = False)
    mean, var, cov = mf2.calc_moments()
    means[i] = np.copy(mean)
    vars[i] = np.copy(var)
    covs[i] = np.copy(cov)
    currents[i] = mf2.calc_expected_electrode_current(3)

    conv_means[i] = np.abs(mf2.dmeans).max()
    conv_vars[i] = np.abs(mf2.dvars).max()
    conv_covs[i] = np.abs(mf2.dcovs).max()

    np.savetxt("data/second_order/4x4/mf2_without_cov/mf2_means4x4.csv", means)
    np.savetxt("data/second_order/4x4/mf2_without_cov/mf2_vars4x4.csv", vars)
    np.savetxt("data/second_order/4x4/mf2_without_cov/mf2_covs4x4.csv", covs.reshape((N, -1)))
    np.savetxt("data/second_order/4x4/mf2_without_cov/mf2_currents4x4.csv", currents)

    np.savetxt("data/second_order/4x4/mf2_without_cov/mf2_conv_means4x4.csv", conv_means)
    np.savetxt("data/second_order/4x4/mf2_without_cov/mf2_conv_vars4x4.csv", conv_vars)
    np.savetxt("data/second_order/4x4/mf2_without_cov/mf2_conv_covs4x4.csv", conv_covs)


