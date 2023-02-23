import module.base.capacitance as capacitance

capacities = capacitance.build_network(2,2,1,[[0,0,0],[1,1,0]],[],[])

print(capacities)