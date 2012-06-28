from .. import BDSEmodel
from . import np, contract


def bdse_examples():
    """ Returns some examples of BDSe systems. """
    examples = []
    examples.append(bdse_ex_one_command_indip(n=1))
    examples.append(bdse_ex_one_command_indip(n=3))
    return examples


@contract(n='int,>=1', k='int,>=1')
def get_bds_M_N(n, k):
    M = np.zeros((k, n, n))
    N = np.zeros((k, n))
    return M, N


@contract(n='int,>=1')
def bdse_ex_one_command_indip(n):
    k = n
    M, N = get_bds_M_N(n=n, k=k)
    for i in range(n):
        M[i, i, i] = 1
    return BDSEmodel(M=M, N=N)
