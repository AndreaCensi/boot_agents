from . import contract, np


class BDSEmodel:

    @contract(M='array[KxNxN],K>=1,N>=1', N='array[KxN]')
    def __init__(self, M, N):
        # TODO: check finite
        self.M = M
        self.k = M.shape[0]
        self.n = M.shape[1]

    @contract(y='array[N]', u='array[K]', returns='array[N]')
    def y_dot(self, y, u):
        self.check_valid_u(u)
        self.check_valid_y(y)
        y_dot = np.dot(u, np.dot(self.M, y) + self.N)
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
