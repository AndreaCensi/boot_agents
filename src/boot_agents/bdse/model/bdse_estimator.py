from .bdse_estimator_interface import BDSEEstimatorInterface
from .bdse_model import BDSEmodel
from .bdse_tensors import (get_M_from_P_T_Q_alt, get_M_from_P_T_Q_alt_scaling,
    get_M_from_P_T_Q, get_M_from_Pinv_T_Q)
from astatsa.expectation_weighted import ExpectationWeighted
from astatsa.utils import assert_allclose
from boot_agents.misc_utils import (pub_tensor2_cov, pub_tensor3_slice2,
    pub_tensor2_comp1)
from boot_agents.utils import (Expectation, MeanCovariance, outer,
    check_matrix_finite)
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
    def __init__(self, rcond=1e-10,
                 antisym_T=False, antisym_M=False,
                 use_P_scaling=False, invalid_P_threshold=None):  # invalid_P_threshold = 1.5
        """
            :param rcond: Threshold for computing pseudo-inverse of P.
            :param antisym_T: If True, the estimate of T is antisymmetrized.
            :param antisym_M: If True, the estimate of M is antisymmetrized.
        """
        self.rcond = rcond
        self.antisym_M = antisym_M
        self.antisym_T = antisym_T
        self.use_P_scaling = use_P_scaling
        self.invalid_P_threshold = invalid_P_threshold
        self.info('rcond: %f' % rcond)
        self.info('antisym_T: %s' % antisym_T)
        self.info('antisym_M: %s' % antisym_M)
        self.info('use_P_scaling: %s' % use_P_scaling)
        self.info('discard_P_threshold: %s' % invalid_P_threshold)

        self.T = Expectation()
        self.U = Expectation()
        self.y_stats = MeanCovariance()
        self.u_stats = MeanCovariance()
        self.nsamples = 0
        self.once = False
        
    def merge(self, other):
        assert isinstance(other, BDSEEstimator)
        self.T.merge(other.T)
        self.U.merge(other.U)
        self.y_stats.merge(other.y_stats)
        self.u_stats.merge(other.u_stats)
        self.nsamples += other.nsamples
    
    @contract(u='array[K],K>0,finite',
              y='array[N],N>0,finite',
              y_dot='array[N],finite', w='>0')
    def update(self, y, u, y_dot, w=1.0):
        check_matrix_finite('y', y)
        check_matrix_finite('y_dot', y_dot)
        check_matrix_finite('u', u)

        self.once = True
        self.nsamples += 1
        
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



    @contract(returns='array[NxN],finite')
    def get_P_inv_cond(self):
        P = self.y_stats.get_covariance()
        P2 = P + np.eye(P.shape[0]) * self.rcond
        P_inv = inverse_valid(P2, self.get_valid())
        
        check_matrix_finite('P_inv', P_inv)
        return P_inv

    @contract(returns='finite')
    def get_T(self):
        if isinstance(self.T, ExpectationWeighted):
            T = self.T.get_value(fill_value=0)
        else:
            T = self.T.get_value()
            
        check_matrix_finite('T', T)

        if self.antisym_T:
            self.info('antisymmetrizing T')
            T = antisym(T)

        mask = 1.0 * self.get_valid_2D()
        for k in range(T.shape[2]):
            T[:, :, k] = T[:, :, k] * mask 
            
        return T
    
    def get_U(self):
        if isinstance(self.T, ExpectationWeighted):
            U = self.U.get_value(fill_value=0)
        else:
            U = self.U.get_value()
            
        check_matrix_finite('U', U)
        return U
    
    
    def get_model(self):
        T = self.get_T()
        U = self.get_U()
        
        P = self.y_stats.get_covariance()
        Q = self.u_stats.get_covariance()

        P_inv = self.get_P_inv_cond()
        
        check_matrix_finite('P_inv', P_inv)
        Q_inv = np.linalg.pinv(Q)
        
        check_matrix_finite('Q_inv', Q_inv)

        if False:
            M = get_M_from_P_T_Q(P, T, Q)
        else:
            if self.use_P_scaling:
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
        mask = 1.0 * self.get_valid_2D()
        for k in range(self.k):
            M[:, :, k] = M[:, :, k] * mask 
        
        Myav = np.tensordot(M, self.y_stats.get_mean(), axes=(1, 0))
        N = UQ_inv - Myav

        if self.antisym_M:
            self.info('antisymmetrizing M')
            M = antisym(M)
         
        self.Myav = Myav
        self.UQ_inv = UQ_inv
            
        return BDSEmodel(M, N)

    def publish(self, pub):
        if not self.once:
            pub.text('warning', 'not updated yet')
            return
        
        pub.text('nsamples', '%s' % self.nsamples)
        
        pub.text('rcond', '%g' % self.rcond)
        pub.text('antisym_M', self.antisym_M)
        pub.text('antisym_T', self.antisym_T)
        pub.text('use_P_scaling', self.use_P_scaling)

        with pub.subsection('model') as sub:
            if sub:
                try:
                    model = self.get_model()
                    model.publish(sub)
                except BDSEEstimatorInterface.NotReady as e:
                    pub.text('not-ready', str(e))

        with pub.subsection('tensors') as sub:
            if sub:
                self.publish_learned_tensors(sub)

        with pub.subsection('y_stats') as sub:
            if sub:
                self.y_stats.publish(sub)

        with pub.subsection('u_stats') as sub:
            if sub:
                self.u_stats.publish(sub)
        
        if True:
            with pub.subsection('alternative', robust=True) as sub:
                if sub:
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
            
    @contract(returns='array[N](bool)')
    def get_valid(self):
        """ Returns a bool array indicating whether the corresponding 
            sensel must be considered valid or not. """
        if self.invalid_P_threshold is None:
            return np.ones(self.n, 'bool')
        else:
            P = self.y_stats.get_covariance()
            d = P.diagonal()
            l, m, p = np.percentile(d, [25, 50, 75])
            w = (p - l) * self.invalid_P_threshold 
            threshold = m - w 
            self.info('P threshold: %s' % threshold)
            return d > threshold
    
    def get_valid_2D(self):
        valid = self.get_valid()
        return outer(valid, valid)

    def publish_learned_tensors(self, sub):
        T = self.get_T()
        U = self.get_U()
        P = self.y_stats.get_covariance()
        Q = self.u_stats.get_covariance()
        P_inv = np.linalg.pinv(P)
        P_inv_cond = self.get_P_inv_cond()
        Q_inv = np.linalg.pinv(Q)

        pub_tensor3_slice2(sub, 'T', T)
        pub_tensor2_comp1(sub, 'U', U)
        pub_tensor2_cov(sub, 'P', P, rcond=self.rcond)
        pub_tensor2_cov(sub, 'P_inv', P_inv)
        pub_tensor2_cov(sub, 'P_inv_cond', P_inv_cond)
        pub_tensor2_cov(sub, 'Q', Q)
        pub_tensor2_cov(sub, 'Q_inv', Q_inv)
    
        valid = self.get_valid()
        f = sub.figure()
        with f.plot('diag_cov') as pylab:
            P = self.y_stats.get_covariance()
            d = P.diagonal()
            pylab.plot(d, '.')
        with f.plot('valid') as pylab:
            pylab.plot(valid, '.')
        
        
        
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

@contract(P='array[NxN]', valid='array[N](bool)', returns='array[NxN],finite')
def inverse_valid(P, valid):
    n = P.shape[0]
    nvalid = np.sum(valid)
    projector, = np.nonzero(valid)

    Pp = P[projector, :][:, projector]
    assert_allclose(Pp.shape, (nvalid, nvalid))
    Pp_inv = np.linalg.inv(Pp)
    P_inv = np.zeros((n, n))
#     v = outer(valid, valid)
#     print v.shape
    print Pp_inv.shape
    print P_inv.shape
    for i, p in enumerate(projector):
        P_inv[p, projector] = Pp_inv[i, :] 
#     P_inv[v] = Pp_inv
    return P_inv
    
    
    
    
