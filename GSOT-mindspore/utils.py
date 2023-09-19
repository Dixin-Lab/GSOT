import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib
from typing import List,Tuple
import mindspore.numpy as mnp
from mindspore import Tensor, dtype,set_seed,ops



matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 8})
class RealData:
    def __init__(self, num_pairs):
        """
        Initialize data simulator
        :param num_src: the number of "supply" sources
        :param num_dst: the number of "demand" destinations
        :param var: the variance of the demands' dynamics in the range [0, 1]
        """
        '''
        self.num_src = 10
        self.num_dst = 10
        self.num_pairs = num_pairs
        self.cost = np.array([110, 99, 80, 90, 123, 173, 133, 73, 93, 148])/173
        self.cost = np.tile(self.cost, (10, 1))
        self.dmean = np.array([1017, 1042, 1358, 2525, 1100, 2150, 1113, 4017, 3296, 2383])
        self.dvar = np.array([194, 323, 248, 340, 381, 404, 524, 556, 1047, 697])
        self.vec_supply = self.dmean.reshape((10, 1))
        self.supply = self.vec_supply / self.vec_supply.sum()
        self.supply = np.tile(self.supply, (1, num_pairs))
        '''
        self.num_src = 10
        self.num_dst = 10
        self.num_pairs = num_pairs
        self.dvar = [194, 323, 248, 340, 381, 404, 524, 556, 1047, 697]
        self.cost = Tensor([110, 99, 80, 90, 123, 173, 133, 73, 93, 148], dtype.float64)
        self.cost = self.cost / 173
        self.cost = self.cost.broadcast_to((10, 10))
        self.dmean = [1017, 1042, 1358, 2525, 1100, 2150, 1113, 4017, 3296, 2383]
        self.vec_supply = Tensor([self.dmean] * num_pairs, dtype.float64)
        self.supply = self.vec_supply / self.vec_supply.sum(axis=0)

    def generate_data(self):
        sample = mnp.zeros((self.num_dst, self.num_pairs), dtype=int)
        # 对每个需求分布dst，采样pair个数据
        for i in range(self.num_dst):
            seed = self.dmean[i]
            var = self.dvar[i]
            sample[i, :] = ops.normal(shape=(self.num_pairs,), mean=seed,  stddev=var)


        sample_normed = sample /self.vec_supply.sum()
        return sample_normed[:,:int(1/2*self.num_pairs)],sample_normed[:,int(1/2*self.num_pairs):]

class SystheticData:
    def __init__(self, num_src: int, num_dst: int = None, var: float = 0.0):
        """
        Initialize data simulator
        :param num_src: the number of "supply" sources
        :param num_dst: the number of "demand" destinations
        :param var: the variance of the demands' dynamics in the range [0, 1]
        """
        self.num_src = num_src
        if num_dst is None:
            self.num_dst = num_src
        else:
            self.num_dst = num_dst
        self.var = var
        # set source and destination points and the distance between them
        '''
        self.pts_src = (np.array(list(range(self.num_src))).reshape(self.num_src, 1) + 0.5) / self.num_src
        self.pts_src = np.concatenate((np.zeros_like(self.pts_src), self.pts_src), axis=1)
        self.pts_dst = (np.array(list(range(self.num_dst))).reshape(self.num_dst, 1) + 0.5) / self.num_dst
        self.pts_dst = np.concatenate((np.ones_like(self.pts_dst), self.pts_dst), axis=1)
        self.cost = euclidean_distances(self.pts_src, self.pts_dst)
        '''

        self.pts_src = (mnp.arange(num_src).reshape((num_src, 1)) + 0.5) / self.num_src
        self.pts_src = mnp.concatenate((mnp.zeros_like(self.pts_src), self.pts_src), axis=1)
        self.pts_dst = (mnp.arange(num_dst).reshape((num_dst, 1)) + 0.5) / num_dst
        self.pts_dst = mnp.concatenate((mnp.ones_like(self.pts_dst), self.pts_dst), axis=1)
        self.cost = ops.cdist(self.pts_src, self.pts_dst, p=2.0)
        #self.cost = euclidean_distances(self.pts_src, self.pts_dst)
        #diff = self.pts_src.unsqueeze(2) - self.pts_dst.unsqueeze(1)
        #squared_distance = P.ReduceSum()(diff ** 2, axis=2)
        #self.cost = P.Sqrt()(squared_distance)




    def generate_pairs(self,
                       num_pairs: int = 100,
                       seed: int = 42,
                       integer: bool = False) -> Tuple[Tensor, Tensor]:
        """
        Generate supply-demand pairs
        :param num_pairs: the number of pairs
        :param seed: the random seed controlling the perturbation
        :param integer: require integer or not
        :return:
        """
        set_seed(seed)
        if integer:
            supply = self.num_dst * mnp.ones((self.num_src, num_pairs))
            demand_ideal = self.num_src * mnp.ones((self.num_dst, num_pairs))
            perturbation = mnp.randn(self.num_dst, num_pairs)
            #perturbation = mnp.random.RandomState(seed).rand(self.num_dst, num_pairs)
            perturbation = mnp.round(self.var * ((perturbation - 0.5) * 2) * min([self.num_src, self.num_dst]))
        else:
            supply = mnp.ones((self.num_src, num_pairs)) / self.num_src
            demand_ideal = mnp.ones((self.num_dst, num_pairs)) * self.num_src / self.num_dst
            perturbation = mnp.randn(self.num_dst, num_pairs)
            #perturbation = np.random.RandomState(seed).rand(self.num_dst, num_pairs)
            perturbation = self.var * ((perturbation - 0.5) * 2) * (self.num_src / self.num_dst)
        demand = perturbation + demand_ideal
        return supply[:,:int(1/2*num_pairs)], demand


    def generate_data(self, num_pairs,balan:bool=True):
        sample = mnp.zeros((self.num_dst, num_pairs))
        # 对每个需求分布dst，采样pair个数据
        for i in range(self.num_dst):
            seed = mnp.randint(5, 8)
            sample[i, :] = ops.normal(shape=(num_pairs,), mean=seed,  stddev=0.5)
        # scaling column-wise
        if(balan):
            sample_normed = sample / sample.sum(axis=0)
        else:
            sample_normed = sample
        return sample_normed[:,:int(1/2*num_pairs)],sample_normed[:,int(1/2*num_pairs):]

    def plot_supply_chain(self, path_name: str, chain: Tensor):
        """
        Plot the supply chain
        :param path_name: the path with image name
        :param chain: the proposed chain, a matrix with size (num_src, num_dst)
        :return:
        """
        plt.figure(figsize=(6, 5))
        plt.scatter(self.pts_src[:, 0], self.pts_src[:, 1], marker='o', s=20, c='blue')
        plt.scatter(self.pts_dst[:, 0], self.pts_dst[:, 1], marker='o', s=20, c='red')
        for i in range(self.num_src):
            for j in range(self.num_dst):
                if chain[i, j] > 1e-8:
                    plt.plot([self.pts_src[i, 0], self.pts_dst[j, 0]],
                             [self.pts_src[i, 1], self.pts_dst[j, 1]],
                             'k-', alpha=chain[i, j])
        plt.title('#Chains={}'.format(mnp.sum(chain > 1e-8)))
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(path_name)
        plt.close()

    def plot_edge(self, path_name: str, edge: list):
        """
        Plot the supply chain
        :param path_name: the path with image name
        :param chain: the proposed chain, a matrix with size (num_src, num_dst)
        :return:
        """

        plt.figure(figsize=(6, 5))
        plt.scatter(self.pts_src[:, 0], self.pts_src[:, 1], marker='o', s=20, c='blue')
        plt.scatter(self.pts_dst[:, 0], self.pts_dst[:, 1], marker='o', s=20, c='red')
        for i, j in edge:
            plt.plot([self.pts_src[i, 0], self.pts_dst[j - self.num_src, 0]],
                     [self.pts_src[i, 1], self.pts_dst[j - self.num_src, 1]],
                     'k-')

        plt.title('#Chains={}'.format(len(edge)))
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(path_name)
        plt.close()

def edge2mat(edges: Tensor, num_nodes: int):
    """
    :param edges: a matrix with size (K, 2) for K directed edges, each row (v, u) indicates an edge v->u
    :param num_nodes: the number of nodes in a graph
    :return:
        a matrix with size (#nodes, #edges)
    """
    edge_cost = mnp.zeros((num_nodes, edges.shape[0]))
    for n in range(num_nodes):
        edge_cost[n, edges[:, 0] == n] = 1
        edge_cost[n, edges[:, 1] == n] = 1
    return edge_cost


def max_flow(price: Tensor,
             edges: Tensor,
             supply: Tensor,
             demand: Tensor,
             integer: bool = False) -> Tuple[float, Tensor, Tensor]:
    """
    Minimum flow algorithm given a bipartite graph
    :param price: the cost matrix with size (ns, nd)
    :param edges: an array with size (K, 2), each row represents an edge u->v
    :param supply: (ns, ) supply histogram
    :param demand: (nt, ) demand histogram
    :param integer: require integer variable or not
    :return:
        result: the optimum objective
        weights: the weights on the edges, with size (K, )
        flow: the flows on the edges, with size (K, )
    """
    price = price.asnumpy()
    supply = supply.asnumpy()
    demand = demand.asnumpy()

    num_src = supply.shape[0]
    weights = np.array([price[edges[i, 0], edges[i, 1] - num_src] for i in range(edges.shape[0])])
    edge_topo = edge2mat(edges, num_nodes=supply.shape[0] + demand.shape[0])
    edge_topo = edge_topo.asnumpy()
    b = np.concatenate((supply, demand), axis=0)
    if integer:
        x = cp.Variable(edges.shape[0], nonneg=True, integer=True)
    else:
        x = cp.Variable(edges.shape[0], nonneg=True)
    objective = cp.Maximize(cp.sum(cp.multiply(weights, x)))

    constraints = [edge_topo @ x <= b, x >= np.zeros((edges.shape[0],))]
    prob = cp.Problem(objective, constraints)
    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    # The optimal value for x is stored in `x.value`.
    return Tensor(result), Tensor(weights), Tensor(x.value)

