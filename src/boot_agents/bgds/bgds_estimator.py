from . import (compute_gradient_information_matrix, contract,
    generalized_gradient, outer_first_dim, np, BGDS)
from ..utils import Expectation, outer
import itertools


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

    @contract(y='(array[M]|array[MxN]),shape(x)',
            y_dot='shape(x)', u='array[K]', dt='>0')
    def update(self, y, y_dot, u, dt):
        if not np.isfinite(y).all(): # XXX: this is wasteful
            raise ValueError('Invalid values in y.')
        if not np.isfinite(y_dot).all():
            raise ValueError('Invalid values in y_dot.')
        if not np.isfinite(u).all():
            raise ValueError('Invalid values in u.')
        # TODO: check shape is conserved
        self.is1D = y.ndim == 1
        self.is2D = y.ndim == 2
        
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
            P = self.P.get_value()
            if self.is2D:
                self.R = compute_gradient_information_matrix(P)
            else:
                self.R = 1.0 / P
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
            if self.is2D:
                for i, j in itertools.product(range(G.shape[2]), range(G.shape[3])):
                    G_s = G[:, :, i, j].squeeze()
                    R_s = R[:, :, i, j].squeeze() 
                    H_s = np.dot(Qinv, np.dot(G_s, R_s)) 
                    H[:, :, i, j] = H_s
            else:
                nS = G.shape[-1]
                for i in range(nS):
                    G_s = G[:, :, i].squeeze()
                    R_s = R[:, :, i].squeeze() 
                    H_s = np.dot(Qinv, np.dot(G_s, R_s))
                    H[:, 0, i] = H_s # XXX: not sure, not tested
            self.H = H
            self.H_needs_update = False
        return self.H
    
    def publish(self, pub):
        # FIXME: add 2d case
        if self.is2D:
            self.publish_2d(pub)
        else:
            self.publish_1d(pub)
        
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
    
    
    def publish_1d(self, pub):
#        K = self.last_u.size
        G = self.get_G()
        P = self.get_P()
        R = self.get_R()
        H = self.get_H()
        
        @contract(value='array[Kx1xN]')
        def display_1d_tensor(pub, name, value):
            with pub.plot(name) as pylab:
                for k in range(value.shape[0]):
                    x = value[k, :, :].squeeze()
                    pylab.plot(x, label='%s%s' % (name, k))
                pylab.legend()
                
        def display_1d_field(pub, name, value):
            with pub.plot(name) as pylab:
                pylab.plot(value)
                
        display_1d_tensor(pub, 'G', G)
        display_1d_tensor(pub, 'P', P)
        display_1d_tensor(pub, 'R', R)
        display_1d_tensor(pub, 'H', H)

        data = pub.section('last_data')
        display_1d_field(data, 'last_y', self.last_y)
        display_1d_field(data, 'last_y_dot', self.last_y_dot)
#        display_1d_field(data, 'last_gy', self.last_gy[0, :, :])
#        display_1d_field(data, 'last_gy_v', self.last_gy[1, :, :])

    def publish_compact(self, pub):
#        K = self.last_u.size
        G = self.get_G()
#        P = self.get_P()
#        R = self.get_R()
        H = self.get_H()
        inter = [0, 1]
        for k in inter:
            with pub.plot('G%s' % k) as pylab:
                pylab.plot(G[k, ...].squeeze())
                
        for k in inter:
            with pub.plot('H%s' % k) as pylab:
                pylab.plot(H[k, ...].squeeze())
        
        
@contract(G='array[AxBxHxW]', xlabels='list[A](str)', ylabels='list[B](str)')
def display_4d_tensor(pub, name, G, xlabels, ylabels):
    section = pub.section(name)
    A = G.shape[0]
    B = G.shape[1]
    for (a, b) in itertools.product(range(A), range(B)):
        value = G[a, b , :, :].squeeze() 
        label = '%s_%s_%s' % (name, xlabels[a], ylabels[b])
        section.array_as_image(label, value)
