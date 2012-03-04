from . import diffeo_torus, np


@diffeo_torus # actually, it's diffeo_square
def rot90(X):
    R = [[0, 1], [-1, 0]]
    return np.dot(R, X)


@diffeo_torus
def rot90_inv(X):
    R = [[0, -1], [+1, 0]]
    return np.dot(R, X)


def rotx_gen(X, rx, ry):
    a = X[0] + rx
    b = X[1] + ry
    return np.array([a, b])


delta = 0.1


@diffeo_torus
def rotx(X):
    return rotx_gen(X, rx=delta, ry=0)


@diffeo_torus
def rotx_inv(X):
    return rotx_gen(X, rx=(-delta), ry=0)


@diffeo_torus
def roty(X):
    return rotx_gen(X, rx=0, ry=delta)


@diffeo_torus
def roty_inf(X):
    return rotx_gen(X, rx=0, ry=(-delta))


@diffeo_torus
def rotx2(X):
    return rotx(rotx(X))


@diffeo_torus
def rotx2_inv(X):
    return rotx_inv(rotx_inv(X))
