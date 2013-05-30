import numpy as np



def cov2corr(covariance, zero_diagonal=False):
    ''' 
    Compute the correlation matrix from the covariance matrix.
    By convention, if the variance of a variable is 0, its correlation is 0 
    with the other, and self-corr is 1. (so the diagonal is always 1).
    
    If zero_diagonal = True, the diagonal is set to 0 instead of 1. 

    :param zero_diagonal: Whether to set the (noninformative) diagonal to zero.
    :param covariance: A 2D numpy array.
    :return: correlation: The exctracted correlation.
    
    '''
    # TODO: add checks
    outer = np.multiply.outer

    sigma = np.sqrt(covariance.diagonal())
    sigma[sigma == 0] = 1
    one_over = 1.0 / sigma
    M = outer(one_over, one_over)
    correlation = covariance * M

    if zero_diagonal:
        for i in range(covariance.shape[0]):
            correlation[i, i] = 0

    return correlation
