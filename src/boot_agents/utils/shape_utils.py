from geometry.formatting import formatm
import numpy as np
from astatsa.utils.np_comparisons import show_some


__all__ = ['expect_shape', 'check_matrix_finite']


def expect_shape(name, vector, shape):
    if vector.shape != shape:
        msg = ('Expected shape %s for %r but found %s' % 
               (shape, name, vector.shape))
        if vector.size < 100:
            msg += '\n' + formatm(vector) 
        raise ValueError(msg)


def check_matrix_finite(name, x):
    if not np.all(np.isfinite(x)):
        failures = np.logical_not(np.isfinite(x))
        s = show_some(x, failures, 'finite', MAX_N=4)
        msg = 'Array %r is not finite:\n%s ' % (name, s)
        raise ValueError(msg)
