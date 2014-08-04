from .bdse_estimator import BDSEEstimator
from astatsa.expectation_weighted import ExpectationWeighted
from astatsa.mean_covariance import MeanCovariance
from astatsa.utils import check_matrix_finite, outer
from contracts import contract
import numpy as np

__all__ = [
    'BDSEEstimatorRobust',
]


class BDSEEstimatorRobust(BDSEEstimator):

    def __init__(self, **other):
        BDSEEstimator.__init__(self, **other)
        
        self.T = ExpectationWeighted()
        self.U = ExpectationWeighted()
        self.y_mean = ExpectationWeighted()  # XXX: not necessary, y_stats.get_mean()
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
        check_matrix_finite('y', y)
        check_matrix_finite('y_dot', y_dot)
        check_matrix_finite('u', u)
        check_matrix_finite('w', w)
        
        self.once = True
        self.nsamples += 1
        
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

    def publish_learned_tensors(self, sub):
        BDSEEstimator.publish_learned_tensors(self, sub)
        with sub.subsection('weights') as s:
            if s:
                Tw = self.T.get_mass()
                Uw = self.U.get_mass()
                TTw = Tw * self.get_T()
                from boot_agents.misc_utils import pub_tensor3_slice2
                from boot_agents.misc_utils import pub_tensor2_comp1
                pub_tensor3_slice2(s, 'Tw', Tw)
                pub_tensor3_slice2(s, 'T * Tw', TTw)
                pub_tensor2_comp1(s, 'Uw', Uw)
            
            
