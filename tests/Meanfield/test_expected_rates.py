import numpy as np

from module.simulation.meanfield import MeanField as MeanField
from module.base.network import Network as Network

net = Network(2,2,1,[[0,0,0],[1,1,0]])
net.set_voltage_config([0.1,-0.1],0)

mf = MeanField(net)

mf.macrostate = np.random.randn(4) * 0

print(mf.calc_expected_island_currents().shape)

print(mf.calc_expected_electrode_current(0))
print(mf.calc_expected_electrode_current(1))

mf.evolve(steps = 0)
