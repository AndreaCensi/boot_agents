""" Algebra of BDS tensors """
from contracts import contract
import numpy as np
from numpy.linalg.linalg import LinAlgError

import warnings


@contract(M='array[NxNxK]', P='array[NxN]', Q='array[KxK]', returns='array[NxNxK]')
def get_expected_T_from_M_P_Q(M, P, Q):
    """ 
        Implements the formula: ..
            
            T^{svi} = P^{sx} M^{v}_{xj} Q^{ij}
    """
    T = np.einsum("sx,vxj,ij -> svi", P, M, Q)
    return T


@contract(P='array[NxN]', T='array[NxNxK]', Q='array[KxK]',
          other='None|dict', returns='array[NxNxK]')
def get_M_from_P_T_Q(P, T, Q, other=None):
    """ 
        Implements the formula: ..
    
            M^{v}_{xj} = T^{svi} Pinv_{sx} Qinv_{ij}
            
    """
    TPinv = obtain_TPinv_from_T_P(T, P)   
    Q_inv = np.linalg.inv(Q)
    M = np.tensordot(TPinv, Q_inv, axes=(2, 0))

    if other is not None:
        other['TP_inv'] = TPinv
    return M
 
    
@contract(Pinv='array[NxN]', T='array[NxNxK]', Q='array[KxK]',
          other='None|dict', returns='array[NxNxK]')
def get_M_from_Pinv_T_Q(Pinv, T, Q, other=None):
    warnings.warn('Untested')
    TPinv = obtain_TPinv_from_T_Pinv(T, Pinv)
    # TODO: check LinAlg   
    Q_inv = np.linalg.inv(Q)
    M = np.tensordot(TPinv, Q_inv, axes=(2, 0))

    if other is not None:
        other['TP_inv'] = TPinv
    return M


@contract(Pinv='array[NxN]', T='array[NxNxK]',
          returns='array[NxNxK]')
def obtain_TPinv_from_T_Pinv(T, Pinv):
    _, _, K = T.shape
    TPinv = np.empty_like(T)
    for k in range(K):
        T_k = T[:, :, k]
        warnings.warn('check order')
        TPinv_k = np.dot(Pinv, T_k)
        warnings.warn('should there be a transpose?')
        TPinv[:, :, k] = TPinv_k.T
    return TPinv


def obtain_TPinv_from_T_P(T, P):
    K = T.shape[2]

    TPinv = np.empty_like(T)
    for k in range(K):
        Tk = T[:, :, k]
        try:
            Mk = np.linalg.solve(P, Tk) 
        except LinAlgError:
            # TODO: what shall we do here?
            msg = 'in obtain_TP_inv_from_TP we found a singular matrix for linalg.solve'
            msg += '; I will try to use the least square solution, but I did not test it'
            warnings.warn(msg)
            Mk, residuals, rank, s = np.linalg.lstsq(P, Tk)  # @UnusedVariable
        TPinv[:, :, k] = Mk.T  # note transpose (to check)
    return TPinv


def get_M_from_P_T_Q_alt(P, T, Q, other=None):
    """ 
        Implements the formula: ..
    
            M^{v}_{xj} = T^{svi} Pinv_{sx} Qinv_{ij}
            
    """
    Q_inv = np.linalg.inv(Q)
    P_inv = np.linalg.inv(P)

    if other is not None:
        other['P_inv'] = P_inv
    
    M = np.einsum("svi,sx,ij -> vxj", T, P_inv, Q_inv)
    return M
