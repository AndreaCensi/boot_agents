from . import BDSEmodel, np, contract
from boot_agents.utils import Expectation, MeanCovariance, outer
from geometry import printm
from boot_agents.misc_utils.tensors_display import pub_tensor2_cov, \
    pub_tensor3_slice2, pub_tensor2, pub_tensor2_comp1


__all__ = ['BDSEEstimator']


class BDSEEstimator:
    """
         M^s_vi   (N) x (N x K)
         N^s_i    (N) x (K)
         T^svi    (NxNxK)
         U^si     (NxK)
    
    """

    def __init__(self, min_count_for_prediction=100):
        self.T = Expectation()
        self.U = Expectation()
        self.y_stats = MeanCovariance()
        self.u_stats = MeanCovariance()

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

    def get_model(self):
        T = self.T.get_value()
        U = self.U.get_value()
        P = self.y_stats.get_covariance()
        Q = self.u_stats.get_covariance()

        P_inv = np.linalg.pinv(P)
        Q_inv = np.linalg.pinv(Q)

        if False:
            printm('Tshape', T.shape)
            printm('T', T)
            printm('U', U)
            printm('P', P)
            printm('Q', Q)
            printm('Q_inv', Q_inv)
            printm('P_inv', P_inv)

        TP_inv = np.tensordot(T, P_inv, axes=(0, 0))
        TP_inv = np.transpose(TP_inv, (0, 2, 1))

        M = np.tensordot(TP_inv, Q_inv, axes=(2, 0))
        UQ_inv = np.tensordot(U, Q_inv, axes=(1, 0))

        Myav = np.tensordot(M, self.y_stats.get_mean(), axes=(1, 0))

        N = UQ_inv - Myav
        return BDSEmodel(M, N)

    def publish(self, pub):
        model = self.get_model()
        model.publish(pub.section('model'))

        pub = pub.section('tensors')
        T = self.T.get_value()
        U = self.U.get_value()
        P = self.y_stats.get_covariance()
        Q = self.u_stats.get_covariance()
        P_inv = np.linalg.pinv(P)
        Q_inv = np.linalg.pinv(Q)
        pub_tensor3_slice2(pub, 'T', T)
        pub_tensor2_comp1(pub, 'U', U)
        pub_tensor2_cov(pub, 'P', P)
        pub_tensor2_cov(pub, 'P_inv', P_inv)
        pub_tensor2_cov(pub, 'Q', Q)
        pub_tensor2_cov(pub, 'Q_inv', Q_inv)
