from . import (compute_gradient_information_matrix, contract,
    generalized_gradient, outer_first_dim, np)
from ..utils import Expectation, outer
import itertools
from . import BGDS


__all__ = ['BGDSEstimator']


class BGDSEstimator:
    '''

Dimensions of G:
- for 1D signals:  (K x 1 x N )
- for 2D signals:  (K x 2 x H x W )
Dimensions of P (covariance of gradient):  
    (ndim x ndim x H x W )    
Dimensions of Q:
    (K x K)

'''

    def __init__(self):
        self.Q = Expectation()
        self.P = Expectation()
        self.G = Expectation()
        self.R = None
        self.R_needs_update = True
        self.H = None
        self.H_needs_update = True

    def get_model(self):
        return BGDS(self.get_H())

    @contract(y='(array[M]|array[MxN]),shape(x)', y_dot='shape(x)', u='array[K]',
              dt='>0')
    def update(self, y, y_dot, u, dt):
        if not np.isfinite(y).all(): # XXX: this is wasteful
            raise ValueError('Invalid values in y.')
        if not np.isfinite(y_dot).all():
            raise ValueError('Invalid values in y_dot.')
        if not np.isfinite(u).all():
            raise ValueError('Invalid values in u.')
        
        gy = generalized_gradient(y)
        self.Q.update(outer(u, u), dt) 
        self.P.update(outer_first_dim(gy), dt)
        self.R_needs_update = True

        Gi = outer(u, gy * y_dot)
        self.G.update(Gi, dt)
        self.H_needs_update = True
    
        self.last_y = y
        self.last_gy = gy
        self.last_y_dot = y_dot
        self.last_u = u    

    def get_R(self):
        ''' Returns the gradient information matrix. '''
        if self.R_needs_update:
            self.R = compute_gradient_information_matrix(self.P.get_value())
            self.R_needs_update = False
        return self.R
    
    def get_G(self): return self.G.get_value()
    def get_P(self): return self.P.get_value()
    def get_Q(self): return self.Q.get_value()
    
    def get_H(self):
        if self.H_needs_update:
            R = self.get_R()
            G = self.get_G()
            Q = self.get_Q()
            H = np.empty_like(G)
            Q = self.get_Q()
            Qinv = np.linalg.pinv(Q)
            # - for 2D signals:  (K x 2 x H x W )
            # multiply Qinv on the first
            # multiply R on the second
            for i, j in itertools.product(range(G.shape[2]), range(G.shape[3])):
                G_s = G[:, :, i, j].squeeze()
                R_s = R[:, :, i, j].squeeze() 
                H_s = np.dot(Qinv, np.dot(G_s, R_s)) 
                H[:, :, i, j] = H_s 
            self.H = H
            self.H_needs_update = False
        return self.H
    
    def publish(self, pub):
        # FIXME: add 2d case
        self.publish_2d(pub)
        
    def publish_2d(self, pub):
        K = self.last_u.size
        G = self.get_G()
        P = self.get_P()
        R = self.get_R()
        H = self.get_H()
        
        u_labels = ['cmd%s' % k for k in range(K)]
        grad_labels = ['h', 'v']

        display_4d_tensor(pub, 'G', G, xlabels=u_labels, ylabels=grad_labels)
        display_4d_tensor(pub, 'P', P, xlabels=grad_labels, ylabels=grad_labels)
        display_4d_tensor(pub, 'R', R, xlabels=grad_labels, ylabels=grad_labels)
        display_4d_tensor(pub, 'H', H, xlabels=u_labels, ylabels=grad_labels)

        data = pub.section('last_data')
        data.array_as_image('last_y', self.last_y, filter='scale')
        data.array_as_image('last_y_dot', self.last_y_dot, filter='scale')
        data.array_as_image('last_gy_h', self.last_gy[0, :, :])
        data.array_as_image('last_gy_v', self.last_gy[1, :, :])
        
@contract(G='array[AxBxHxW]', xlabels='list[A](str)', ylabels='list[B](str)')
def display_4d_tensor(pub, name, G, xlabels, ylabels):
    section = pub.section(name)
    A = G.shape[0]
    B = G.shape[1]
    for (a, b) in itertools.product(range(A), range(B)):
        value = G[a, b , :, :].squeeze() 
        label = '%s_%s_%s' % (name, xlabels[a], ylabels[b])
        section.array_as_image(label, value)
