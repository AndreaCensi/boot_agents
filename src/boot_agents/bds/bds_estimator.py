import numpy as np
from boot_agents.utils import MeanCovariance, Expectation, outer, DerivativeBox, Queue
from boot_agents.simple_stats import ExpSwitcher
from contracts import contract
from contracts import new_contract
import scipy.linalg

@new_contract
@contract(x='array')
def array_finite(x):
    return np.isfinite(x).all()
    
class BDSEstimator2:
    
    def __init__(self):
        self.T = Expectation()
        self.uu = Expectation()
        self.yy = Expectation()
        self.yy_inv = None
        self.yy_inv_needs_update = True
        self.M = None
        self.M_needs_update = True
    
    @contract(u='array[K],K>0,array_finite',
              y='array[N],N>0,array_finite',
              y_dot='array[N],array_finite', dt='>0')
    def update(self, u, y, y_dot, dt):
        self.num_commands = u.size
        self.num_sensels = y.size
        T = outer(u, outer(y, y_dot))         
        self.T.update(T , dt)
        self.yy.update(outer(y, y), dt)
        self.uu.update(outer(u, u), dt)
        self.yy_inv_needs_update = True
        self.M_needs_update = True
        
    def get_yy_inv(self, rcond=1e-5):
        if self.yy_inv_needs_update:
            self.yy_inv_needs_update = False
            yy = self.yy.get_value()
            self.yy_inv = np.linalg.pinv(yy, rcond) 
        return self.yy_inv 
    
    def get_M(self):
        
        if self.M_needs_update:
            self.M_needs_update = False
            T = self.get_T()
            if self.M is None:
                self.M = np.zeros(T.shape, T.dtype)
            for k in range(self.num_commands): 
                yy = self.get_yy()
                Tk = T[k, :, :]
                Mk = scipy.linalg.solve(yy, Tk)
                self.M[k, :, :] = Mk.T # note transpose
        return self.M
        
    def get_T(self):
        return self.T.get_value()

    def get_yy(self):
        return self.yy.get_value()
    
    def get_uu(self):
        return self.uu.get_value()
        
    def publish(self, pub):        
        params = dict(filter=pub.FILTER_POSNEG, filter_params={'skim':2})

        T = self.get_T()
        
        for i in range(self.num_commands):
            Ti = T[i, :, :]
            pub.array_as_image(('T', 'T%d' % i), Ti, **params)

        M = self.get_M()
        
        for i in range(self.num_commands):
            Mi = M[i, :, :]
            pub.array_as_image(('M', 'M%d' % i), Mi, **params)

        
        pub.array_as_image(('stats', 'yy'), -self.get_yy(), **params)
        pub.array_as_image(('stats', 'yy_inv'), self.get_yy_inv(), **params)
        pub.array_as_image(('stats', 'uu'), self.get_uu(), **params)
