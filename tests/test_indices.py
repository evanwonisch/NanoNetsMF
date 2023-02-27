import numpy as np

from module.base.network import Network

net = Network(3,3,3,[])


# convert to cartesian
linear_indices = np.arange(29)
cartesian_indices = net.get_cartesian_indices(linear_indices)
print(cartesian_indices)

# convert to linear indices
linear_indices = net.get_linear_indices(cartesian_indices)
print(linear_indices)
print("should be 0 1 2 3 ... 26 -1 -1")