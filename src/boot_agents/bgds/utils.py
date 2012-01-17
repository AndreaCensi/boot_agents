from . import np, contract
import itertools


@contract(a='array[N],N>3', returns='array[N]')
def gradient1d(a):
    ''' 
        Computes the gradient of a 1D array, using the [-1,0,1] filter.
    '''
    b = np.empty_like(a)
    n = a.size
    b[1:n - 1] = (a[2:n] - a[0:n - 2]) / 2
    b[0] = b[1]
    b[-1] = b[-2]
    return b


@contract(a='array[N],N>3', returns='array[N]')
def gradient1d_slow(a):
    ''' 
        Computes the gradient of a 1D array, using the [-1,0,1] filter.
    '''
    b = np.empty_like(a)
    n = a.size
    for i in xrange(1, n - 1):
        b[i] = (a[i + 1] - a[i - 1]) / 2

    b[0] = b[1]
    b[-1] = b[-2]
    return b


@contract(a='array[HxW]', returns='tuple(array[HxW],array[HxW])')
def gradient2d(a):
    g0 = np.empty_like(a)
    g1 = np.empty_like(a)

    # vertical gradient
    for j in range(a.shape[1]):
        g0[:, j] = gradient1d(a[:, j])

    # horizontal
    for i in range(a.shape[0]):
        g1[i, :] = gradient1d(a[i, :])

    return g0, g1


@contract(y='array[HxW]|array[N]', returns='array')
def generalized_gradient(y):
    assert y.ndim in [1, 2]
    shape = (y.ndim,) + y.shape
    gy = np.zeros(shape, 'float32')
    if y.ndim == 1:
        gy[0, ...] = gradient1d(y)
    else:
        x, y = gradient2d(y)
        gy[0, ...] = x
        gy[1, ...] = y
    return gy


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


@contract(y='array[HxW]', scale='>0', returns='array[HxW]')
def smooth2d(y, scale):
    ''' 
        Smooths the 2D array y with a kernel of the given scale 
        (sigma in sensels). 
    '''
    from scipy.ndimage import gaussian_filter
    # TODO: move somewhere else
    return gaussian_filter(y, sigma=scale)



