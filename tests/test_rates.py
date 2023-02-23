import numpy as np
from matplotlib import pyplot as plt

from module.base.network import Network
import module.components.CONST as CONST

net = Network(2,2,1,[[0,0,0]],[],[])
net.set_voltage_config([1],[],[], 0)

print("Thermal Energy:", CONST.kb * CONST.temperature, "aJ")
print("Rate per Energy (approx):", 1/CONST.electron_charge**2/CONST.tunnel_resistance)
print("Rate for -1aJ:", net.calc_rate(-1))

print("Rate for 0aJ:", net.calc_rate(0))
print("Rate for 0aJ (exact)", CONST.kb * CONST.temperature / CONST.electron_charge**2 / CONST.tunnel_resistance)


print("END")

# plot tunnel rates
dF = np.linspace(-0.10,0.10,100)
rates = net.calc_rate(dF)
np.savetxt("out/rates.csv",np.stack((dF, rates), axis = 1))

plt.figure()
dF = np.linspace(-0.11,0.11,100)
T = [0.26,10,100,400]

for i in range(len(T)):
    CONST.temperature = T[i]
    rates = net.calc_rate(dF)
    plt.plot(dF, rates, label = "T = " +  "{:10.4f}".format(CONST.temperature) + "K")

dF = np.linspace(-0.11,0,100)
plt.plot(dF, -dF/CONST.electron_charge**2/CONST.tunnel_resistance)

plt.xlabel("dF in aJ")
plt.ylabel("rate in 1/as")
plt.title("a = atto")
plt.legend()
plt.savefig("out/rates.png")