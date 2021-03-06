from boot_agents.misc_utils import display_1d_tensor, display_4d_tensor, display_3d_tensor
from boot_agents.utils import expect_shape, generalized_gradient
from contracts import contract
import numpy as np


class BGDSmodel(object):
    """
        
        Dimensions of G:
        - for 1D signals:  (K x 1 x N )
        - for 2D signals:  (K x 2 x H x W )
        

        Dimensions of B:
        - for 1D signals:  (K x N )
        - for 2D signals:  (K x H x W )

    """

    @contract(G='array[Kx1xN]|array[Kx2xHxW]',
              B='array[KxN]|array[KxHxW]')
    def __init__(self, G, B):
        # TODO: check finite
        self.G = G
        self.B = B
        self.ncmd = G.shape[0]
        if len(G.shape) == 3: 
            self.is2D = False
            self.y_shape = (G.shape[-1],)
        else:
            self.is2D = True
            self.y_shape = (G.shape[-2], G.shape[-1])
        
    # @contract(y='array[N]', u='array[K]', returns='array[N]')
    def get_y_dot(self, y, u, gy=None):
        self.check_valid_y(y)
        self.check_valid_u(u)
    
        if self.is2D:
            return self.get_y_dot_2d(y, u, gy)
        else:
            return self.get_y_dot_1d(y, u, gy)
    
    @contract(y='array[N]', u='array[K]', returns='array[N]')
    def get_y_dot_1d(self, y, u, gy=None):
        if gy is None:
            gy = generalized_gradient(y)
        
        uH = np.tensordot(u, self.G, axes=(0, 0))
        # XXX: add others
        y_dot = (uH * gy).sum(axis=0)
        return y_dot

    @contract(y='array[HxW]', u='array[K]', returns='array[HxW]')
    def get_y_dot_2d(self, y, u, gy=None):
        if gy is None:
            gy = generalized_gradient(y)
        
        uH = np.tensordot(u, self.G, axes=(0, 0))
        # XXX: add others
        y_dot = (uH * gy).sum(axis=0)
        return y_dot

    def estimate_u(self, y, y_dot, gy=None):
        self.check_valid_y(y)
        self.check_valid_y(y_dot)
    
        if self.is2D:
            return self.estimate_u_2d(y, y_dot, gy)
        else:
            return self.estimate_u_1d(y, y_dot, gy)
    
    
    @contract(y='array[N]', y_dot='array[N]')
    def estimate_u_1d(self, y, y_dot, gy=None):
        GyB = self.get_lin_op_1d(y, gy)
        u_est, _, _, _ = np.linalg.lstsq(GyB, y_dot)
        return u_est

    @contract(y='array[N]', gy='None|array[1xN]', returns='array[NxK]')
    def get_lin_op_1d(self, y, gy=None):
        """ Returns G (nabla)y + B, in the 1D case. """
        if gy is None:
            gy = generalized_gradient(y) 
        # gy: (1 x N)
        # self.G: array[Kx1xN]
        # self.B: array[KxN]
        
        # FIXME: I am not sure of this, should be tested
        Ggy = (self.G * gy).squeeze()  # K x N 
        L = Ggy + self.B  # K x N
        return L.T  # N x K
            
    @contract(y='array[AxB]', y_dot='array[AxB]')
    def estimate_u_2d(self, y, y_dot, gy=None):
        # FIXME: this must be fixed, still uses old conventions
        if gy is None:
            gy = generalized_gradient(y)

        A, B = y.shape
        n = A * B
        K = self.H.shape[0]

        # FIXME: G
        H = self.H
        # H = (K x 2 x H x W )
        # gy = (2 x H x W )
        Hgy = np.tensordot(H * gy, [1, 1], axes=(1, 0))

        # This was the slower implementation:
        #        a = np.zeros((n, K))
        #        b = np.zeros(n)
        #        s = 0
        #        for i, j in itertools.product(range(y.shape[0]), 
        # range(y.shape[1])):
        #            a[s, :] = Hgy[:, i, j] 
        #            b[s] = y_dot[i, j]
        #            s += 1
        #        assert s == n

        a = Hgy.reshape((K, n)).T
        b = y_dot.reshape(n)

        u_est, _, _, _ = np.linalg.lstsq(a, b)

        return u_est

    def publish(self, pub):
        # FIXME: add 2d case
        if self.is2D:
            self.publish_2d(pub)
        else:
            self.publish_1d(pub)

    def publish_2d(self, pub):
        u_labels = ['cmd%s' % k for k in range(self.ncmd)]
        grad_labels = ['h', 'v']
        display_4d_tensor(pub, 'G', self.G, xlabels=u_labels,
                                            ylabels=grad_labels)
        display_3d_tensor(pub, 'B', self.B, labels=u_labels)
 
    def publish_1d(self, pub):
        display_1d_tensor(pub, 'G', self.G)
        display_1d_tensor(pub, 'B', self.B)

    @contract(u='array[K]')
    def check_valid_u(self, u):
        expect_shape('u', u, (self.ncmd,))

    @contract(y='array[N]|array[MxN]')
    def check_valid_y(self, y):
        expect_shape('y', y, self.y_shape)

    @contract(y_dot='array[N]|array[MxN]')
    def check_valid_y_dot(self, y_dot):
        expect_shape('y_dot', y_dot, self.y_shape)
