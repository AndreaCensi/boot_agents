from .bdse_estimator import BDSEEstimator
from astatsa.expectation_weighted import ExpectationWeighted
from astatsa.utils import check_all_finite
from boot_agents.utils import MeanCovariance, outer
from contracts import contract
import numpy as np


__all__ = ['BDSEEstimatorRobust']

class BDSEEstimatorRobust(BDSEEstimator):

    def __init__(self, **other):
        BDSEEstimator.__init__(self, **other)
        
        self.T = ExpectationWeighted()
        self.U = ExpectationWeighted()
        self.y_mean = ExpectationWeighted()
        self.y_stats = MeanCovariance()  # TODO: make robust
        self.u_stats = MeanCovariance()
        self.once = False
        
    def merge(self, other):
        assert isinstance(other, BDSEEstimatorRobust)
        self.T.merge(other.T)
        self.U.merge(other.U)
        self.y_stats.merge(other.y_stats)
        self.y_mean.merge(other.y_mean)
        self.u_stats.merge(other.u_stats)

  
    @contract(u='array[K],K>0,finite',
              y='array[N],N>0,finite',
              y_dot='array[N],finite',
              w='array[N]')
    def update(self, y, u, y_dot, w):
        self.once = True
        check_all_finite(y)
        check_all_finite(u)
        check_all_finite(y_dot)
        check_all_finite(w)
        
        self.n = y.size
        self.k = u.size   

        self.y_stats.update(y)  # TODO: make robust 
        self.u_stats.update(u)
        
        # remove mean
        u_n = u - self.u_stats.get_mean()
        self.y_mean.update(y, w)  # TODO: make robust
        y_n = y - self.y_mean.get_value(fill_value=0.5)
        
        # weights
        y_n_w = w
        y_dot_w = w
        u_n_w = np.ones(u.shape)
        
        T_k = outer(outer(y_n, y_dot), u_n)
        T_k_w = outer(outer(y_n_w, y_dot_w), u_n_w)
        
        U_k = outer(y_dot, u_n)
        U_k_w = outer(y_dot_w, u_n_w) 

        assert T_k.shape == (self.n, self.n, self.k)
        assert U_k.shape == (self.n, self.k)
        
        # update tensor
        self.T.update(T_k, T_k_w)
        self.U.update(U_k, U_k_w)

