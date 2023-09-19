import matplotlib.pyplot as plt
import ot
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 8})
from smooth_ot import *
from mindspore import ops

class ADMM:


    def __init__(self,n,m,k,P,u_c,u_d,alpha=1,rho=1,gamma=1):
        self.m = m
        self.n = n
        self.k = k
        self.alpha = alpha
        self.rho = rho
        self.gamma = gamma
        self.P = P
        self.C = -P
        self.u_c = u_c
        self.u_d = u_d
        self.T = mnp.zeros((n,m,k), dtype=mnp.float64)
        self.U = mnp.zeros((n,m,k), dtype=mnp.float64)
        self.Z = mnp.zeros((n,m,k), dtype=mnp.float64)

        for i in range(k):
            self.T[:,:,i] = mnp.outer(self.u_c[:,i],self.u_d[:,i])
            self.Z[:,:,i] = self.T[:,:,i]

    def Normalization(self):
        self.u_d /= mnp.sum(self.u_c, axis=0)
        self.u_c /= mnp.sum(self.u_c, axis=0)
    def Update_BalanceT(self):
        '''
        the primal-dual algorithm in (Blondel et al., 2017)
        '''

        #self.T = UpdateBT(self.T,self.Z,self.U,self.P,self.rho,self.u_c,self.u_d)
        regul = SquaredT(gamma=1.0)
        # update K independent optimal transport problems, parallel computation is possible
        for k in range(self.T.shape[2]):
            C = -self.P - self.rho * (self.Z[:, :, k] - self.U[:, :, k])
            alpha, beta = solve_dual( self.u_c[:, k], self.u_d[:, k], C,regul, max_iter=1000)
            self.T[:, :, k] = get_plan_from_dual(alpha, beta, C, regul)
            #X = alpha.reshape(-1, 1) + beta - C
            #self.T[:, :, k] = Tensor(regul.delta_Omega(X)[1],dtype.float64)


    def Update_UnbalanceT_semi(self):
        '''
               the semi-dual algorithm in (Blondel et al., 2017)
        '''
        regul = SquaredT(gamma=1.0)
        for k in range(self.T.shape[2]):
            C = -self.P - self.rho * (self.Z[:, :, k] - self.U[:, :, k])
            alpha = solve_semi_dual(self.u_c[:, k], self.u_d[:, k],C,  regul, max_iter=1000)
            self.T[:, :, k] = get_plan_from_semi_dual(alpha,self.u_d[:, k] , C, regul)

    def Update_UnbalanceT_CG(self):
        '''
        the scaling algorithm in (Frogner et al., 2015)
        '''

        lamb = 2 / self.rho
        for k in range(self.T.shape[2]):
            C = self.rho * self.U[:, :, k] - self.P - 0.5 * self.rho * mnp.log(self.Z[:, :, k] + 1e-9)
            uc = self.u_c[:, k]
            ud = self.u_d[:, k]
            T = ot.sinkhorn(uc.astype('float64').asnumpy(), ud.astype('float64').asnumpy(), C.astype('float64').asnumpy(), lamb)
            self.T[:, :, k]  = Tensor(T)

    def UpdateZ(self):
        '''
        Applying soft-thresholding method
        z_ij = max{ 1- alpha/(rho*||r_ij||) , 0}
        r_ij = t_ij + u_ij
        '''
        for i in range(self.Z.shape[0]):
            for j in range(self.Z.shape[1]):
                r_ij = self.T[i, j, :] + self.U[i, j, :]
                Z_ij = self.soft_thresholding (r_ij)+1e-8
                self.Z[i, j, :] = Z_ij

    def soft_thresholding (self,r_ij):
        '''
        z_ij = max{ 1- alpha/(rho*||r_ij||) , 0}
        '''
        norm2 = ops.sqrt(mnp.sum(r_ij ** 2))
        if norm2 > 0:
            z = max(0, 1 - self.alpha / (self.rho * norm2)) * r_ij
        else:
            z = mnp.zeros_like(r_ij)

        return z

    def UpdateU(self):

        self.U = self.U+(self.T-self.Z)





