from .bdse_estimator_interface import BDSEEstimatorInterface
from .bdse_model import BDSEmodel
from .bdse_tensors import get_M_from_P_T_Q, get_M_from_Pinv_T_Q
from boot_agents.bdse.model.bdse_tensors import (get_M_from_P_T_Q_alt,
    get_M_from_P_T_Q_alt_scaling)
from boot_agents.misc_utils import (pub_tensor2_cov, pub_tensor3_slice2,
    pub_tensor2_comp1)
from boot_agents.utils import Expectation, MeanCovariance, outer
from conf_tools.utils import indent
from contracts import contract
from numpy.linalg.linalg import LinAlgError
import numpy as np
import traceback
import warnings


__all__ = ['BDSEEstimator']


class BDSEEstimator(BDSEEstimatorInterface):
    """
        Estimates a BDSE model.
        
        Tensors used: ::
        
            M^s_vi   (N) x (N x K)
            N^s_i    (N) x (K)
            T^svi    (NxNxK)
            U^si     (NxK)
    
    """

    @contract(rcond='float,>0')
    def __init__(self, rcond=1e-10, antisym_T=False, antisym_M=False, use_P_scaling=False):
        """
            :param rcond: Threshold for computing pseudo-inverse of P.
            :param antisym_T: If True, the estimate of T is antisymmetrized.
            :param antisym_M: If True, the estimate of M is antisymmetrized.
        """
        self.rcond = rcond
        self.antisym_M = antisym_M
        self.antisym_T = antisym_T
        self.use_P_scaling = use_P_scaling
        self.info('rcond: %f' % rcond)
        self.info('antisym_T: %s' % antisym_T)
        self.info('antisym_M: %s' % antisym_M)
        self.info('use_P_scaling: %s' % use_P_scaling)

        self.T = Expectation()
        self.U = Expectation()
        self.y_stats = MeanCovariance()
        self.u_stats = MeanCovariance()
        self.once = False
        
        
    def merge(self, other):
        assert isinstance(other, BDSEEstimator)
        self.T.merge(other.T)
        self.U.merge(other.U)
        self.y_stats.merge(other.y_stats)
        self.u_stats.merge(other.u_stats)


    @contract(u='array[K],K>0,finite',
              y='array[N],N>0,finite',
              y_dot='array[N],finite', w='>0')
    def update(self, y, u, y_dot, w=1.0):
        self.once = True
        
        self.n = y.size
        self.k = u.size  # XXX: check

        self.y_stats.update(y, w)
        self.u_stats.update(u, w)
        
        # remove mean
        u_n = u - self.u_stats.get_mean()
        y_n = y - self.y_stats.get_mean()
        
        # make products
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

    def get_T(self):
        T = self.T.get_value()
        if self.antisym_T:
            self.info('antisymmetrizing T')
            T = antisym(T)
        return T
    
    def get_model(self):
        T = self.get_T()
            
        U = self.U.get_value()
        P = self.y_stats.get_covariance()
        Q = self.u_stats.get_covariance()

        P_inv = self.get_P_inv_cond()
        Q_inv = np.linalg.pinv(Q)

        if False:
            M = get_M_from_P_T_Q(P, T, Q)
        else:
            if hasattr(self, 'use_P_scaling') and self.use_P_scaling:
                M = get_M_from_P_T_Q_alt_scaling(P, T, Q)
            else:
                warnings.warn('untested')
                try:
                    M = get_M_from_Pinv_T_Q(P_inv, T, Q)
                except LinAlgError as e:
                    msg = 'Could not get_M_from_Pinv_T_Q.\n'
                    msg += indent(traceback.format_exc(e), '> ')
                    raise BDSEEstimatorInterface.NotReady(msg)
        
        UQ_inv = np.tensordot(U, Q_inv, axes=(1, 0))
        # This works but badly conditioned
        Myav = np.tensordot(M, self.y_stats.get_mean(), axes=(1, 0))
        N = UQ_inv - Myav

        if self.antisym_M:
            self.info('antisymmetrizing M')
            M = antisym(M)
        
#         # Note: this does not work, don't know why
#         if False:
#             printm('MYav1', Myav)
#             y2 = np.linalg.solve(P, self.y_stats.get_mean())
#             Myav2 = np.tensordot(T, y2, axes=(0, 0))
#             # Myav = np.tensordot(T, y2, axes=(1, 0))
#             printm('MYav2', Myav2)
#         if False:
#             printm('U', U, 'Q_inv', Q_inv)
#             printm('UQ_inv', UQ_inv, 'Myav', Myav, 'N', N)
#             printm('u_mean', self.u_stats.get_mean())
#             printm('u_std', np.sqrt(Q.diagonal()))
#             printm('y_mean', self.y_stats.get_mean())
            
        self.Myav = Myav
        self.UQ_inv = UQ_inv
            
        return BDSEmodel(M, N)

    def publish(self, pub):
        if not self.once:
            pub.text('warning', 'not updated yet')
            return
        
        pub.text('rcond', '%g' % self.rcond)
        with pub.subsection('model') as sub:
            try:
                model = self.get_model()
                model.publish(sub)
            except BDSEEstimatorInterface.NotReady as e:
                pub.text('not-ready', str(e))

        with pub.subsection('tensors') as sub:
            T = self.get_T()
            U = self.U.get_value()
            P = self.y_stats.get_covariance()
            Q = self.u_stats.get_covariance()
            P_inv = np.linalg.pinv(P)
            P_inv_cond = self.get_P_inv_cond()
            Q_inv = np.linalg.pinv(Q)
    #
    #        TP_inv2 = obtain_TP_inv_from_TP_2(T, P)  
    #        M2 = np.tensordot(TP_inv2, Q_inv, axes=(2, 0))
        
            pub_tensor3_slice2(sub, 'T', T)
            pub_tensor2_comp1(sub, 'U', U)
            pub_tensor2_cov(sub, 'P', P, rcond=self.rcond)
            pub_tensor2_cov(sub, 'P_inv', P_inv)
            pub_tensor2_cov(sub, 'P_inv_cond', P_inv_cond)
            pub_tensor2_cov(sub, 'Q', Q)
            pub_tensor2_cov(sub, 'Q_inv', Q_inv)
            # Might not have been computed
            # pub_tensor2_comp1(sub, 'Myav', self.Myav)
            # pub_tensor2_comp1(sub, 'UQ_inv', self.UQ_inv)

        with pub.subsection('y_stats') as sub:
            self.y_stats.publish(sub)

        with pub.subsection('u_stats') as sub:
            self.u_stats.publish(sub)
          
        with pub.subsection('alternative', robust=True) as sub:
            sub.text('info', 'This is estimating without conditioning P')
            T = self.get_T() 
            P = self.y_stats.get_covariance()
            Q = self.u_stats.get_covariance()
            
            M1 = get_M_from_P_T_Q(P, T, Q)
            pub_tensor3_slice2(sub, 'get_M_from_P_T_Q', M1)
            
            M2 = get_M_from_P_T_Q_alt(P, T, Q)
            pub_tensor3_slice2(sub, 'get_M_from_P_T_Q_alt', M2)

            M3 = get_M_from_P_T_Q_alt_scaling(P, T, Q)
            pub_tensor3_slice2(sub, 'get_M_from_P_T_Q_alt2', M3)
            
            
@contract(T='array[NxNxK]', returns='array[NxNxK]')        
def antisym(T):
    """ Antisymmetrizes a tensor with respect to the first two dimensions. """
    T = T.copy()
    for k in range(T.shape[2]):
        T[:, :, k] = antisym2d(T[:, :, k])
    return T


@contract(M='array[NxN]', returns='array[NxN]')        
def antisym2d(M):
    """ Antisymmetrizes a matrix """
    return 0.5 * (M - M.T)


    
