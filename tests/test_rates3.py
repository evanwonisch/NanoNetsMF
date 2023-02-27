import numpy as np
from matplotlib import pyplot as plt

from module.base.network import Network
import module.components.CONST as CONST

net = Network(2,2,1,[[0,0,0]])
net.set_voltage_config([-0.1], 0)

n = np.array([0,0,0,0])
electrode_index = 0

rates_to_electrode = net.calc_rate_to_electrode(n, electrode_index)
rates_from_electrode = net.calc_rate_from_electrode(n , electrode_index)

print("to electrode:", rates_to_electrode)
print("from electrode:", rates_from_electrode)