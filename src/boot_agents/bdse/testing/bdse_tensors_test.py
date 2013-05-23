from ..model import (get_expected_T_from_M_P_Q, get_M_from_P_T_Q,
    get_M_from_P_T_Q_alt)
from numpy.testing import assert_allclose
import numpy as np
from contracts import contract
from boot_agents.bdse.testing.examples import for_all_bdse_examples
# from geometry.formatting import printm


def get_M_test():
    """ testing get_M_from_P_T_Q() """
    n, k = 3, 2    
    P = get_random_cov(n)
    Q = get_random_cov(k)
    M = np.random.randn(n, n, k)
    T = get_expected_T_from_M_P_Q(M, P, Q)
    M2 = get_M_from_P_T_Q(P, T, Q)
    # print()
    # printm('T', T, 'M', M, 'M2', M2)
    assert_allclose(M, M2)

def get_M_alt_test():
    """ testing get_M_from_P_T_Q_alt() """
    n, k = 3, 2    
    P = get_random_cov(n)
    Q = get_random_cov(k)
    M = np.random.randn(n, n, k)
    T = get_expected_T_from_M_P_Q(M, P, Q)
    M2_alt = get_M_from_P_T_Q_alt(P, T, Q)
    # print()
    # printm('T', T, 'M', M, 'M2_alt', M2_alt)
    assert_allclose(M, M2_alt)

    
# TODO: move to geometry
def get_random_cov(n):
    Psq = np.random.randn(n, n)
    P = np.dot(Psq, Psq.T)
    return P
    
    
@for_all_bdse_examples
def equivariance1(id_bds, bds):  # @UnusedVariable
    n = bds.get_y_shape()
    A = np.random.randn(n, n)
    A_inv = np.linalg.inv(A)
    bds_A = bds.conjugate(A)
    bds2 = bds_A.conjugate(A_inv)
    
    atol = 1e-8  # otherwise 0 and 0.00001 won't match
    assert_allclose(bds.M, bds2.M, atol=atol)
    assert_allclose(bds.N, bds2.N, atol=atol)

@for_all_bdse_examples
def equivariance2(id_bds, bds):  # @UnusedVariable
    n = bds.get_y_shape()
    k = bds.get_u_shape()
    A = np.random.randn(n, n)
    bds_A = bds.conjugate(A)
    
    # Let's make up some random covariance
    P = get_random_cov(n)
    y_mean = np.random.randn(n)
    Q = get_random_cov(k)
    
    # This is what we expect to see 
    T = bds.get_expected_T(P, Q)
    U = bds.get_expected_U(y_mean, Q)
    
    
    # For the other, the covariance is APA*
    Pz = conjugate_covariance(P, A)
    z_mean = np.dot(A, y_mean)
    Tz = bds_A.get_expected_T(Pz, Q)
    Uz = bds_A.get_expected_U(z_mean, Q)
    
    # We expect that it is conjugated to T
    Tz2 = conjugate_T(T, A)  # Tz = A T A*
    Uz2 = conjugate_U(U, A) 
    
    atol = 1e-8  # otherwise 0 and 0.00001 won't match
    assert_allclose(Tz, Tz2, atol=atol)
    assert_allclose(Uz, Uz2, atol=atol)
    
    
    
@contract(P='array[NxN]', A='array[NxN]', returns='array[NxN]')
def conjugate_covariance(P, A):
    """ 
        Formula: ..
            
            Pz^{xa} = A^x_s  P^sv  A^{a}_{v}
    """
    return np.einsum('xs,sv,av -> xa', A, P, A)

@contract(T='array[NxNxK]', A='array[NxN]', returns='array[NxNxK]')
def conjugate_T(T, A):
    """ 
        Formula: ..
            
            Tz ^ {xqi} = A^x_r T^{rsi} A^q_s
    """
    return np.einsum('xr,rsi,qs -> xqi', A, T, A)


@contract(U='array[NxK]', A='array[NxN]', returns='array[NxK]')
def conjugate_U(U, A):
    """ 
        Formula: ..
            
            Uz ^ xj = A^x_r U^{rj}
    """
    return np.einsum('xr,rj -> xj', A, U)

    
     
#    # Suppose there is this perturbation
#    # z = Ay
#    Ainv = np.linalg.inv(A)
#    # meanwhile, it will have 
#    # it will have this covariance 
#    Pz = np.einsum('ab,bc,cd -> ad', A, P, A)
#    
    
