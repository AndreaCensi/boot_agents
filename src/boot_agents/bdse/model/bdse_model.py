from . import contract, np
from geometry.formatting import formatm


class BDSEmodel:
    """
        
        M^s_vi is (N) x (N x K)
        N^s_i  is (N) x (K)
    
    """

    @contract(M='array[NxNxK],K>=1,N>=1', N='array[NxK]')
    def __init__(self, M, N):
        # TODO: check finite
        self.M = M
        self.N = N
        self.n = M.shape[0]
        self.k = M.shape[2]

    def description(self):
        s = ""
        for i in range(self.k):
            s += formatm('M^s_v%d' % i, self.M[:, :, i])
            s += formatm('N^s_%d' % i, self.N[:, i])
        return s

    def get_y_shape(self):
        return self.n

    def get_u_shape(self):
        return self.k

    @contract(y='array[N]', u='array[K]', returns='array[N]')
    def get_y_dot(self, y, u):
        self.check_valid_u(u)
        self.check_valid_y(y)
        My = np.tensordot(self.M, y, axes=(1, 0))
        MyN = My + self.N
        y_dot = np.tensordot(MyN, u, axes=(1, 0))
        self.check_valid_y_dot(y_dot)
        return y_dot

    @contract(u='array[K]')
    def check_valid_u(self, u):
        self.expect_shape('u', u, (self.k,))

    @contract(y='array[N]')
    def check_valid_y(self, y):
        self.expect_shape('y', y, (self.n,))

    @contract(y_dot='array[N]')
    def check_valid_y_dot(self, y_dot):
        self.expect_shape('y_dot', y_dot, (self.n,))

    def expect_shape(self, name, vector, shape):
        if vector.shape != shape:
            msg = ('Expected shape %s for %r but found %s' %
                   (shape, vector, vector.shape))
            raise ValueError(msg)
