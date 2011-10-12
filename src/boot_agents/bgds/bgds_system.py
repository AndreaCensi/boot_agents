from . import generalized_gradient, np, contract

class BGDS(object):
    """
    Dimensions of G:
    - for 1D signals:  (K x 1 x N )
    - for 2D signals:  (K x 2 x H x W )
    """
    def __init__(self, H):
        self.H = H
        
    @contract(y='array[HxW]', u='array[K]',
              returns='array[HxW]')
    def estimate_y_dot(self, y, u, gy=None):
        if gy is None:
            gy = generalized_gradient(y)
        H = self.H
        uH = np.tensordot(u, H, axes=(0, 0))
        y_dot = (uH * gy).sum(axis=0)
        return y_dot
    
    @contract(y='array[AxB]', y_dot='array[AxB]')
    def estimate_u(self, y, y_dot, gy=None):
        if gy is None:
            gy = generalized_gradient(y)

        A, B = y.shape
        n = A * B
        K = self.H.shape[0]
        
        H = self.H
        # H = (K x 2 x H x W )
        # gy = (2 x H x W )
        Hgy = np.tensordot(H * gy, [1, 1], axes=(1, 0))

        # This was the slow system
        #        a = np.zeros((n, K))
        #        b = np.zeros(n)
        #        s = 0
        #        for i, j in itertools.product(range(y.shape[0]), range(y.shape[1])):
        #            a[s, :] = Hgy[:, i, j] 
        #            b[s] = y_dot[i, j]
        #            s += 1
        #        assert s == n

        a = Hgy.reshape((K, n)).T
        b = y_dot.reshape(n)
        
        u_est, _, _, _ = np.linalg.lstsq(a, b)
        
        return u_est
