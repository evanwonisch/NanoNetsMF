import numpy as np
from matplotlib import pyplot as plt

from module.base.network import Network
import module.components.CONST as CONST

net = Network(2,2,1,[])
net.set_voltage_config([], 0)

n = np.array([[1,0,0,0],[0,1,0,0]])
alpha = np.array([0,1])
beta = np.array([1,3])

rates = net.calc_rate_island(n, alpha, beta)

print(rates)
print("rates should be equal")