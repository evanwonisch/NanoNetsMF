import numpy as np

from module.base.network import Network

net = Network(3,3,3,[])
print("Testing index-schlacht:\n\n")


# convert to cartesian
linear_indices = np.arange(29)
cartesian_indices = net.get_cartesian_indices(linear_indices)
print(cartesian_indices)
print("should bei in standard order ascending\n")

# convert to linear indices
linear_indices = net.get_linear_indices(cartesian_indices)
print(linear_indices)
print("should be 0 1 2 3 ... 26 -1 -1\n")

# next neighbours
linear_indices = np.arange(29)
print("calculating neighbours for indices", linear_indices)
neighbours = net.get_nearest_neighbours(linear_indices)
print("Neighbours")
print(neighbours)