from . import diffeo_torus_reflection, np


@diffeo_torus_reflection
def refx(X):
    a = -X[0]
    b = +X[1]
    return np.array([a, b])


@diffeo_torus_reflection
def refy(X):
    a = +X[0]
    b = -X[1]
    return np.array([a, b])


@diffeo_torus_reflection
def refxy(X):
    a = -X[0]
    b = -X[1]
    return np.array([a, b])


@diffeo_torus_reflection
def tran(X):
    a = +X[1]
    b = +X[0]
    return np.array([a, b])


# all of these are their own self inverse
