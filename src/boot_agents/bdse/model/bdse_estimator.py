from . import BDSEmodel, np, contract
from boot_agents.misc_utils.tensors_display import (pub_tensor2_cov,
    pub_tensor3_slice2, pub_tensor2_comp1)
from boot_agents.utils import Expectation, MeanCovariance, outer
from geometry import printm
from boot_agents.bdse.model.bdse_tensors import get_M_from_P_T_Q



__all__ = ['BDSEEstimator']


class BDSEEstimator:
    """
         M^s_vi   (N) x (N x K)
         N^s_i    (N) x (K)
         T^svi    (NxNxK)
         U^si     (NxK)
    
    """

    @contract(rcond='float,>0')
    def __init__(self, rcond=1e-10):
        """
            :param rcond: Threshold for computing pseudo-inverse of P.
        """
        self.T = Expectation()
        self.U = Expectation()
        self.y_stats = MeanCovariance()
        self.u_stats = MeanCovariance()
        self.rcond = rcond

    @contract(u='array[K],K>0,array_finite',
              y='array[N],N>0,array_finite',
              y_dot='array[N],array_finite', w='>0')
    def update(self, y, u, y_dot, w=1.0):
        self.n = y.size
        self.k = u.size # XXX: check

        self.y_stats.update(y, w)
        self.u_stats.update(u, w)
        # remove mean
        u_n = u - self.u_stats.get_mean()
        y_n = y - self.y_stats.get_mean()
        T_k = outer(outer(y_n, y_dot), u_n)
        assert T_k.shape == (self.n, self.n, self.k)
        U_k = outer(y_dot, u_n)
        assert U_k.shape == (self.n, self.k)
        # update tensor
        self.T.update(T_k, w)
        self.U.update(U_k, w)

    def get_P_inv_cond(self):
        P = self.y_stats.get_covariance()
        if False:
            P_inv = np.linalg.pinv(P, rcond=self.rcond)
        if True:
            P2 = P + np.eye(P.shape[0]) * self.rcond
            P_inv = np.linalg.inv(P2)
        return P_inv

    def get_model(self):
        T = self.T.get_value()
        U = self.U.get_value()
        P = self.y_stats.get_covariance()
        Q = self.u_stats.get_covariance()

        P_inv = self.get_P_inv_cond()
        Q_inv = np.linalg.pinv(Q)

        if False:
            printm('Tshape', T.shape)
            printm('T', T)
            printm('U', U)
            printm('P', P)
            printm('Q', Q)
            printm('Q_inv', Q_inv)
            printm('P_inv', P_inv)

        M = get_M_from_P_T_Q(P, T, Q)
        
        UQ_inv = np.tensordot(U, Q_inv, axes=(1, 0))
        # This works but badly conditioned
        Myav = np.tensordot(M, self.y_stats.get_mean(), axes=(1, 0))
        N = UQ_inv - Myav
        
        
        # Note: this does not work, don't know why
        if False:
            printm('MYav1', Myav)
            y2 = np.linalg.solve(P, self.y_stats.get_mean())
            Myav2 = np.tensordot(T, y2, axes=(0, 0))
            #Myav = np.tensordot(T, y2, axes=(1, 0))
            printm('MYav2', Myav2)

        if False:
            printm('U', U, 'Q_inv', Q_inv)
            printm('UQ_inv', UQ_inv, 'Myav', Myav, 'N', N)
            printm('u_mean', self.u_stats.get_mean())
            printm('u_std', np.sqrt(Q.diagonal()))
            printm('y_mean', self.y_stats.get_mean())
            
        self.Myav = Myav
        self.UQ_inv = UQ_inv
            
        return BDSEmodel(M, N)

    def publish(self, pub):
        pub.text('rcond', '%g' % self.rcond)
        model = self.get_model()
        model.publish(pub.section('model'))

        pub = pub.section('tensors')
        T = self.T.get_value()
        U = self.U.get_value()
        P = self.y_stats.get_covariance()
        Q = self.u_stats.get_covariance()
        P_inv = np.linalg.pinv(P)
        P_inv_cond = self.get_P_inv_cond()
        Q_inv = np.linalg.pinv(Q)
#
#        TP_inv2 = obtain_TP_inv_from_TP_2(T, P)  
#        M2 = np.tensordot(TP_inv2, Q_inv, axes=(2, 0))

        pub_tensor3_slice2(pub, 'T', T)
        pub_tensor2_comp1(pub, 'U', U)
        pub_tensor2_cov(pub, 'P', P, rcond=self.rcond)
        pub_tensor2_cov(pub, 'P_inv', P_inv)
        pub_tensor2_cov(pub, 'P_inv_cond', P_inv_cond)
        pub_tensor2_cov(pub, 'Q', Q)
        pub_tensor2_cov(pub, 'Q_inv', Q_inv)
        pub_tensor2_comp1(pub, 'Myav', self.Myav)
        pub_tensor2_comp1(pub, 'UQ_inv', self.UQ_inv)

