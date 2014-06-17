from contracts import contract

from astatsa.expectation_weighted import ExpectationWeighted
from astatsa.utils import check_all_finite
from boot_agents.bgds.utils import outer_first_dim
from boot_agents.utils import generalized_gradient
from boot_agents.utils import outer
import numpy as np

from .bgds_estimator import BGDSEstimator


__all__ = ['BGDSEstimator1DRobust']


class BGDSEstimator1DRobust(BGDSEstimator):
    '''
     
        Dimensions of G:
        - for 1D signals:  (K x 1 x N )
        Dimensions of P (covariance of gradient):  
            (1 x 1 x H x W )    
        Dimensions of Q:
            (K x K)
         
        Dimensions of C:
        - for 1D signals:  (K x N )
     
    '''
    def __init__(self):
        self.Q = ExpectationWeighted()  # XXX bug
        self.P = ExpectationWeighted()
        self.G = ExpectationWeighted()
        self.B = ExpectationWeighted()
 
        self.C = None
        self.C_needs_update = True
        self.R = None
        self.R_needs_update = True
        self.H = None
        self.H_needs_update = True 
 
        self.once = False
        
    @contract(y='(array[M]|array[MxN]),shape(x)',
            y_dot='shape(x)', u='array[K]', w='array[M]')
    def update(self, y, y_dot, u, w):
        self.once = True
        
        M = y.shape[0]
        check_all_finite(y)
        check_all_finite(y_dot)
        check_all_finite(u)
        # TODO: check shape is conserved
        self.is1D = y.ndim == 1
        self.is2D = y.ndim == 2
 
        gy = generalized_gradient(y)
  
        y_dot_w = w
        u_w = np.ones(u.shape)         
        gy_w = w.reshape((1, M)) 
        assert gy.shape == gy_w.shape
        
        Qi = outer(u, u)
        Qi_w = outer(u_w, u_w)
        self.Q.update(Qi, Qi_w)
        
        Pi = outer_first_dim(gy)
        Pi_w = outer_first_dim(gy_w)
         
        self.P.update(Pi, Pi_w)
        self.R_needs_update = True
 
        Gi = outer(u, gy * y_dot)
        Gi_w = outer(u_w, gy_w * y_dot_w)
        
        self.G.update(Gi, Gi_w)
        self.H_needs_update = True
 
        Bk = outer(u, y_dot)
        Bk_w = outer(u_w, y_dot_w)
        self.B.update(Bk, Bk_w)
        self.C_needs_update = True
 
        self.last_y = y
        self.last_gy = gy
        self.last_y_dot = y_dot
        self.last_u = u
        self.last_w = w
     
