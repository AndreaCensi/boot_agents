from . import BGDS, np
from numpy.testing.utils import assert_allclose
import itertools


# TODO: finish this
def test_bgds_0():
    h = 10
    w = 10
    K = 3
    H = np.zeros((K, 2, h, w), dtype='float32')

    for k, d, i, j in itertools.product(range(K), range(2),
                                        range(h), range(w)):
        H[k, d, i, j] = k + d + i + j + np.random.rand()

#    for k, a in itertools.product(range(K), range(2)):
#        printm('H[%d,%d,:,:]' % (k, a), H[k, a, :, :])
#        
    bgds = BGDS(H)

    y = np.random.randn(h, w)
    u = np.array(range(K), dtype='float32') + 1
    y_dot = bgds.estimate_y_dot(y, u)
    u_est = bgds.estimate_u(y, y_dot)

#    printm('u', u, 'u_est', u_est, 'ld', u / u_est)
    assert_allclose(u, u_est, rtol=1e-5)

