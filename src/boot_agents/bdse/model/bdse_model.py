from . import contract, np
from boot_agents.misc_utils import pub_tensor3_slice2, pub_tensor2_comp1
from boot_agents.utils import expect_shape
from geometry import formatm
from boot_agents.bdse.model.bdse_tensors import get_expected_T_from_M_P_Q

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
        """ Predicts y_dot given u and y. """
        self.check_valid_u(u)
        self.check_valid_y(y)
        MyN = self.get_MyN(y)
        y_dot = np.tensordot(MyN, u, axes=(1, 0))
        self.check_valid_y_dot(y_dot)
        return y_dot
    
    @contract(y='array[N]', y_dot='array[N]', returns='array[K]')
    def estimate_u(self, y, y_dot):
        """ Infers u given y and y_dot. """
        self.check_valid_y(y)
        self.check_valid_y(y_dot)
        
        # Size (N x K)
        MyN = self.get_MyN(y)
        
        # We have the system y_dot = MyN u
        # which is overdetermined (if we have more observations than commands)
        u, residuals, rank, st = np.linalg.lstsq(MyN, y_dot) #@UnusedVariable
        
        # TODO: not tested
        return u
    
    @contract(y='array[N]', returns='array[NxK]')
    def get_MyN(self, y):
        """ Returns (My + N) """
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
    
    # From now on, just visualization and other boring stuff
     
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


    @contract(P='array[NxN]', Q='array[KxK]', returns='array[NxNxK]')
    def get_expected_T(self, P, Q):
        """ Expected value of the T statistics. """
        return get_expected_T_from_M_P_Q(self.M, P, Q)
    
    @contract(y_mean='array[N]', Q='array[KxK]', returns='array[NxK]')
    def get_expected_U(self, y_mean, Q):
        """ 
            Expected value of the U statistics. 
        
                U^{sj} =  (M^s_vi ym^v + N^s_i) Q^ij 
        """
        MyN = self.get_MyN(y_mean)
        # MyN^s_i 
        return np.einsum('si,ij->sj', MyN, Q)
    
    
    @contract(A='array[NxN]')
    def conjugate(self, A):
        """ 
            Returns the dynamics of z = Ay. 
        
            Mz^{x}_{qi} = A^{x}_{s}  M^{s}_{vi} (A^{-1})^{v}_{q}
            
            N^{x}_{i} = A^{x}_{s} N^{s}_{i}
            
        """
        Ainv = np.linalg.inv(A)
        Mz = np.einsum('xs, svi, vq -> xqi', A, self.M, Ainv)
        Nz = np.einsum('xs, si  -> xi', A, self.N)
        return BDSEmodel(Mz, Nz)
        
        
        
    
