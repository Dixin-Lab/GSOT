"""
A MindSpore implementation of GSOT
https://github.com/Dixin-Lab/GSOT
"""
from GSOT import get_transport_plan,get_edges_from_plan
from GSOT import evaluate_net_profit,plot_network
from utils import *

num_src=10    #M
num_dst= 10     #N
num_pairs=10    #K

#systhetic data
# generate training and testing data

simulator = SystheticData(num_src=num_src, num_dst=num_dst,var = 0.3)
supply, demand=simulator.generate_pairs(num_pairs=num_pairs,seed=12)
train_demand, test_demand= simulator.generate_data(num_pairs=num_pairs,balan=False)

'''
#real-world data
simulator = RealData(num_pairs=num_pairs)
supply=simulator.supply[:,:int(1/2*num_pairs)]
train_demand,test_demand=simulator.generate_data()
'''
# get tranport_plan
plan = get_transport_plan(price=simulator.cost, supply=supply,
                          demand=train_demand,alpha=1,rho=2.5, balance=False)
# get designed edges from tranport_plan
edges = get_edges_from_plan(trans_plan=plan,max_num_edges=15)

# evaluate the profit achieved by the designed netwotk
values,net_profit = evaluate_net_profit(sorted_edges=edges, price=simulator.cost,
                                        supply=supply, demand=test_demand)
print("GSOT profit:",net_profit)
#show the network structure
plot_network("network.pdf",num_src,num_dst,edges)


#full-flexibility
cost = simulator.cost.asnumpy()
full_flex = np.unravel_index(np.argsort(cost, axis=None)[::-1], cost.shape)
full_edges = [[full_flex[0][k], full_flex[1][k] + num_src] for k in range(num_src * num_dst)]
full_edges = np.array(full_edges)
_,max_profit = evaluate_net_profit(Tensor(full_edges),simulator.cost, supply, test_demand)

print("full-profit:",max_profit)

