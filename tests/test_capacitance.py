import module.base.capacitance as capacitance

capacities = capacitance.build_network(2,2,1,[[0,0,0],[1,1,0]],[],[])

print("Capacities:")
print("node", capacities["node"])
print("lead", capacities["lead"])
print("self", capacities["self"])
print("cap_mat:")
print( capacities["cap_mat"])
print("END")