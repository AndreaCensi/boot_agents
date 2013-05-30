from contracts import contract
import numpy as np

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
    """
        y = array[HxW]   =>  array[2xHxW]
        y = array[N]     =>  array[1xN]
    """
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

