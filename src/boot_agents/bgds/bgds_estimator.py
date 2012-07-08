from . import (compute_gradient_information_matrix, contract,
    generalized_gradient, outer_first_dim, np, BGDSmodel)
from ..misc_utils import (display_3d_tensor, display_4d_tensor, display_1d_tensor,
    display_1d_field, iterate_indices)
from ..utils import Expectation, outer


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
    
    Dimensions of C:
    - for 1D signals:  (K x N )
    - for 2D signals:  (K x H x W )


'''

    def __init__(self):
        self.Q = Expectation()
        self.P = Expectation()
        self.G = Expectation()
        self.B = Expectation()

        self.C = None
        self.C_needs_update = True

        self.R = None
        self.R_needs_update = True
        self.H = None
        self.H_needs_update = True

    def get_model(self):
        return BGDSmodel(self.get_H(), self.get_C())

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

        Bk = outer(u, y_dot)
        self.B.update(Bk, dt)
        self.C_needs_update = True

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

    def get_B(self):
        return self.B.get_value()

    def get_G(self):
        return self.G.get_value()

    def get_P(self):
        return self.P.get_value()

    def get_Q(self):
        return self.Q.get_value()

    def get_H(self):
        if self.H_needs_update:
            R = self.get_R()
            G = self.get_G()
            H = np.empty_like(G)
            Qinv = self.get_Q_inv()
            # - for 2D signals:  (K x 2 x H x W )
            # multiply Qinv on the first
            # multiply R on the second
            if self.is2D:
                h, w = G.shape[2], G.shape[3]
                for i, j in iterate_indices((h, w)):
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

    def get_C(self):
        if self.C_needs_update:
            B = self.get_B()
            Qinv = self.get_Q_inv()
            if self.C is None:
                self.C = np.empty_like(B)

            if self.is2D:
                h, w = B.shape[-2], B.shape[-1]
                for i, j in iterate_indices((h, w)):
                    B_s = B[:, i, j].squeeze()
                    C_s = np.dot(Qinv, B_s)
                    self.C[:, i, j] = C_s
            else: # 1D
                for i in range(B.shape[-1]):
                    B_s = B[:, i].squeeze()
                    C_s = np.dot(Qinv, B_s)
                    self.C[:, i] = C_s
        return self.C

    def get_Q_inv(self):
        # TODO: check convergence
        Q = self.get_Q()
        return np.linalg.pinv(Q)

    def publish(self, pub):
        if self.is2D:
            self.publish_2d(pub)
        else:
            self.publish_1d(pub)

    def publish_2d(self, pub):
        K = self.last_u.size
        G = self.get_G()
        P = self.get_P()
        R = self.get_R()
        B = self.get_B()

        # TODO: use names from boot spec
        u_labels = ['cmd%s' % k for k in range(K)]
        grad_labels = ['h', 'v']

        acc = pub.section('accumulators')
        display_4d_tensor(acc, 'G', G, xlabels=u_labels, ylabels=grad_labels)
        display_3d_tensor(acc, 'B', B, labels=u_labels)
        display_4d_tensor(acc, 'P', P, xlabels=grad_labels,
                          ylabels=grad_labels)
        display_4d_tensor(acc, 'R', R, xlabels=grad_labels,
                          ylabels=grad_labels)

        model = self.get_model()
        model.publish(pub.section('model'))

        data = pub.section('last_data')
        data.array_as_image('last_y', self.last_y, filter='scale')
        data.array_as_image('last_y_dot', self.last_y_dot, filter='scale')
        data.array_as_image('last_gy_h', self.last_gy[0, :, :])
        data.array_as_image('last_gy_v', self.last_gy[1, :, :])

    def publish_1d(self, pub):
        G = self.get_G()
        P = self.get_P()
        R = self.get_R()
        B = self.get_B()

        accum = pub.section('accumulators')
        display_1d_tensor(accum, 'G', G)
        display_1d_tensor(accum, 'B', B)
        display_1d_tensor(accum, 'P', P)
        display_1d_tensor(accum, 'R', R)

        model = self.get_model()
        model.publish(pub.section('model'))

        data = pub.section('last_data')
        display_1d_field(data, 'last_y', self.last_y)
        display_1d_field(data, 'last_y_dot', self.last_y_dot)
        display_1d_field(data, 'last_gy', self.last_gy[0, :])

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


