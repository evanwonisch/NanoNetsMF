import sys
import numpy as np
from matplotlib import pyplot as plt
from module.base.network import Network
from module.simulation.meanfield import MeanField
from module.simulation.quick_meanfield2 import QuickMeanField2
import module.components.CONST as CONST

net = Network(3,3,1,[[0,0,0],[2,0,0],[0,2,0],[2,2,0]])

ws = (np.arange(0, 80) / 2)[1:]

for w in ws:
    N_oscil = 15   #oscillations
    N_point = 250  # points per oscillation

    t_max = N_oscil / w * 2 * np.pi  
    dt = 1/w * 2 * np.pi / N_point   

    N = int(t_max / dt)
    ts = np.linspace(0, t_max, N)    # nanoseconds
    U_in = np.sin(w * ts) * 0.09

    name_str = "_w="+str(w)

    print("Doing:")
    print("w =", w)
    print("N =", N)
    print("dt =", dt)

    np.savetxt("data/time_dep/oscil/ts"+name_str+".csv", ts)
    np.savetxt("data/time_dep/oscil/Us"+name_str+".csv", U_in)


    mf = MeanField(net)
    mf_means = np.zeros((N, 9))
    mf_currents = np.zeros((N, 4))
    means = np.zeros((9))
    for i in range(N):
        net.set_voltage_config([U_in[i], 0, 0, 0], 0)
        means = mf.numeric_integration_solve(macrostate = means, N = 1, dt = dt)
        mf_means[i] = np.copy(means)

        mf_currents[i, 0] = mf.calc_expected_electrode_rates(means, 0) * CONST.electron_charge
        mf_currents[i, 1] = mf.calc_expected_electrode_rates(means, 1) * CONST.electron_charge
        mf_currents[i, 2] = mf.calc_expected_electrode_rates(means, 2) * CONST.electron_charge
        mf_currents[i, 3] = mf.calc_expected_electrode_rates(means, 3) * CONST.electron_charge

    np.savetxt("data/time_dep/oscil/mf_means"+name_str+".csv", mf_means)
    np.savetxt("data/time_dep/oscil/mf_currents"+name_str+".csv", mf_currents)


    qmf2 = QuickMeanField2(net)
    qmf2_means = np.zeros((N, 9))
    qmf2_vars = np.zeros((N, 9))
    qmf2_currents = np.zeros((N, 4))
    for i in range(N):
        net.set_voltage_config([U_in[i], 0, 0, 0], 0)
        qmf2.numeric_integration_solve(N = 1, dt = dt, reset = False)
        qmf2_means[i] = np.copy(qmf2.means)
        qmf2_vars[i] = np.copy(qmf2.vars)

        qmf2_currents[i, 0] = qmf2.calc_expected_electrode_current(0)
        qmf2_currents[i, 1] = qmf2.calc_expected_electrode_current(1)
        qmf2_currents[i, 2] = qmf2.calc_expected_electrode_current(2)
        qmf2_currents[i, 3] = qmf2.calc_expected_electrode_current(3)

    np.savetxt("data/time_dep/oscil/qmf2_means"+name_str+".csv", qmf2_means)
    np.savetxt("data/time_dep/oscil/qmf2_vars"+name_str+".csv", qmf2_vars)
    np.savetxt("data/time_dep/oscil/qmf2_currents"+name_str+".csv", qmf2_currents)