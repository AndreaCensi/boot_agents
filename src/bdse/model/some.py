from .bdse_model import BDSEmodel
from contracts import contract
import numpy as np

__all__ = [
    'get_bds_M_N',
    'bdse_ex_one_command_indip',
    'bdse_random',
    'bdse_zero',
]

@contract(n='int,>=1', k='int,>=1')
def get_bds_M_N(n, k):
    M = np.zeros((n, n, k))
    N = np.zeros((n, k))
    return M, N

@contract(n='int,>=1')
def bdse_ex_one_command_indip(n, k):
    M, N = get_bds_M_N(n=n, k=k)
    for i in range(n):
        M[i, i, :] = 1
    return BDSEmodel(M=M, N=N)

def bdse_random(n, k):
    M, N = get_bds_M_N(n=n, k=k)
    M = np.random.randn(*M.shape)
    N = np.random.randn(*N.shape)
    return BDSEmodel(M=M, N=N)

def bdse_zero(n, k):
    M, N = get_bds_M_N(n=n, k=k)
    return BDSEmodel(M=M, N=N)