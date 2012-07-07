from . import BDSEmodel, np, contract
from boot_agents.misc_utils.tensors_display import (pub_tensor2_cov,
    pub_tensor3_slice2, pub_tensor2_comp1)
from boot_agents.utils import Expectation, MeanCovariance, outer
from geometry import printm
from numpy.linalg.linalg import LinAlgError


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

        if True:
            TP_inv = obtain_TP_inv_from_TP(T, P)            
        else:
            TP_inv = np.tensordot(T, P_inv, axes=(0, 0))
            TP_inv = np.transpose(TP_inv, (0, 2, 1))

        M = np.tensordot(TP_inv, Q_inv, axes=(2, 0))
        UQ_inv = np.tensordot(U, Q_inv, axes=(1, 0))

        # This works but badly conditioned
        # Myav = np.tensordot(M, self.y_stats.get_mean(), axes=(1, 0))
        # This works but perhaps worth checking indices formally
        y2 = np.linalg.solve(P, self.y_stats.get_mean())
        Myav = np.tensordot(T, y2, axes=(1, 0))

        N = UQ_inv - Myav
        
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
        pub_tensor3_slice2(pub, 'T', T)
        pub_tensor2_comp1(pub, 'U', U)
        pub_tensor2_cov(pub, 'P', P, rcond=self.rcond)
        pub_tensor2_cov(pub, 'P_inv', P_inv)
        pub_tensor2_cov(pub, 'P_inv_cond', P_inv_cond)
        pub_tensor2_cov(pub, 'Q', Q)
        pub_tensor2_cov(pub, 'Q_inv', Q_inv)
        pub_tensor2_comp1(pub, 'Myav', self.Myav)
        pub_tensor2_comp1(pub, 'UQ_inv', self.UQ_inv)


def obtain_TP_inv_from_TP(T, P):
    M = np.empty_like(T)
    for k in range(T.shape[2]):
        Tk = T[:, :, k]
        try:
            Mk = np.linalg.solve(P, Tk) #@UndefinedVariable
        except LinAlgError:
            raise
        M[:, :, k] = Mk.T # note transpose (to check)
    return M
