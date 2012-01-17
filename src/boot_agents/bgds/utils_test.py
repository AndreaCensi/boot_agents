from . import gradient1d, gradient2d, np
from numpy.testing.utils import assert_allclose


g_examples = [
       ([1, 1, 1, 1], [0, 0, 0, 0]),
       ([1, 2, 3, 4], [1, 1, 1, 1]),
       ([-1, -2, -3, -4], [-1, -1, -1, -1]),
]


def test_gradient1d_0():

    for y, gy_expected in g_examples:
        y = np.array(y)
        gy = gradient1d(y)
        assert_allclose(gy, gy_expected)


def test_gradient2d_0():
    for y, gy_expected in g_examples:
        y = np.array(y)

        X1 = np.zeros((y.size, y.size))
        for j in range(X1.shape[1]):
            X1[:, j] = y

        X1_0, _ = gradient2d(X1)
        for j in range(X1.shape[1]):
            assert_allclose(X1_0[:, j], gy_expected)

        X2 = np.zeros((y.size, y.size))
        for i in range(X2.shape[1]):
            X2[i, :] = y

        _, X2_1 = gradient2d(X2)
        for i in range(X2.shape[0]):
            assert_allclose(X2_1[i, :], gy_expected)

