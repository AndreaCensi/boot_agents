from . import contract, np
from geometry.formatting import formatm
from boot_agents.misc_utils.tensors_display import pub_tensor3_slice2, \
    pub_tensor2_comp1

__all__ = ['BDSEmodel']


class BDSEmodel:
    """
        Tensor sizes: ::
        
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

    @contract(y='array[N]', u='array[K]', returns='array[N]')
    def get_y_dot(self, y, u):
        self.check_valid_u(u)
        self.check_valid_y(y)
        MyN = self.get_MyN(y)
        y_dot = np.tensordot(MyN, u, axes=(1, 0))
        self.check_valid_y_dot(y_dot)
        return y_dot

    @contract(y='array[N]', returns='array[NxK]')
    def get_MyN(self, y):
        My = np.tensordot(self.M, y, axes=(1, 0))
        MyN = My + self.N
        return MyN

    @contract(y='array[N]', y_goal='array[N]', metric='None|array[NxN]')
    def get_servo_descent_direction(self, y, y_goal, metric=None): #@UnusedVariable
        # TODO: add arbitrary metric
        self.check_valid_y(y)
        self.check_valid_y(y_goal)
        MyN = self.get_MyN(y) # should I use average?
        e = y_goal - y
        direction = np.tensordot(MyN, e, axes=(0, 0))
        return direction
    
    def publish(self, pub):
        pub_tensor3_slice2(pub, 'M', self.M)
        pub_tensor2_comp1(pub, 'N', self.N)

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

    @contract(u='array[K]')
    def check_valid_u(self, u):
        expect_shape('u', u, (self.k,))

    @contract(y='array[N]')
    def check_valid_y(self, y):
        expect_shape('y', y, (self.n,))

    @contract(y_dot='array[N]')
    def check_valid_y_dot(self, y_dot):
        expect_shape('y_dot', y_dot, (self.n,))


# TODO: move away
def expect_shape(name, vector, shape):
    if vector.shape != shape:
        msg = ('Expected shape %s for %r but found %s' % 
               (shape, name, vector.shape))
        raise ValueError(msg)
