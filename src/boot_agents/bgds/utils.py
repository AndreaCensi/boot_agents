from contracts import contract
import numpy as np
import itertools

__all__ = ['smooth2d', 'outer_first_dim', 'compute_gradient_information_matrix']


@contract(y='array[HxW]', scale='>0', returns='array[HxW]')
def smooth2d(y, scale):
    ''' 
        Smooths the 2D array y with a kernel of the given scale 
        (sigma in sensels). 
    '''
    from scipy.ndimage import gaussian_filter
    # TODO: move somewhere else
    return gaussian_filter(y, sigma=scale)


@contract(P='array[2x2xHxW]', returns='array[2x2xHxW]')
def compute_gradient_information_matrix(P):
    ''' computes the information matrix for the gradient covariance tensor. '''
    # inverse for 2x2 matrix:
    #    1     | +d  -b |
    # -------  |        |
    # ad - bc  | -c   a |
    I = np.zeros(P.shape, 'float32')
    a = P[0, 0, :, :].squeeze()
    b = P[0, 1, :, :].squeeze()
    c = P[1, 0, :, :].squeeze()
    d = P[1, 1, :, :].squeeze()

    det = (a * d - b * c)
    det[det <= 0] = 1
    one_over_det = 1.0 / det

    I[0, 0, :, :] = one_over_det * (+d)
    I[0, 1, :, :] = one_over_det * (-b)
    I[1, 0, :, :] = one_over_det * (-c)
    I[1, 1, :, :] = one_over_det * (+a)

    return I


def outer_first_dim(x):
    K = x.shape[0]
    result_shape = (K,) + x.shape
    result = np.zeros(shape=result_shape, dtype='float32')
    for (i, j) in itertools.product(range(K), range(K)):
        result[i, j, ...] = x[i, ...] * x[j, ...]
    return result

