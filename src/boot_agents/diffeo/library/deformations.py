from . import np, diffeo_torus
from contracts import contract

P = 3.0


def pow_gen(X, p):
    a = np.power(X[0], p)
    b = np.power(X[1], p)
    return np.array([a, b])


@diffeo_torus
def pow3(X):
    return pow_gen(X, P)


@diffeo_torus
def pow3x(X):
    a = np.power(X[0], P)
    b = X[1]
    return np.array([a, b])


@diffeo_torus
def pow3x_inv(X):
    A = np.abs(X[0])
    S = np.sign(X[0])
    a = S * np.power(A, 1 / P)
    b = X[1]
    return np.array([a, b])

from numpy import cos, pi


def twirlop(X, epsilon=0.1):
    x = X[0]
    y = X[1]
    A1 = epsilon * (cos(pi * x) + 1) * (cos(pi * y) + 1) / 4.0
    A2 = epsilon * cos(pi * x / 2) * cos(pi * y / 2)
    A = (A1 + A2) / 2
    z = x + (-y) * A
    w = y + (x) * A
    return np.array([z, w])


@diffeo_torus
def twirl(X):
    return twirlop(X)


def iterate(f, X, n):
    for _ in range(n):
        X = f(X)
    return X


@diffeo_torus
def twirl10(X):
    return iterate(twirlop, X, 10)


@diffeo_torus
def twirl20(X):
    return iterate(twirlop, X, 20)


@contract(x='>=-1,<=1', returns='>=-1,<=1')
def sinzoom(x):
    return np.sin(x * np.pi / 2)


@contract(y='>=-1,<=1', returns='>=-1,<=1')
def sinzoom_inv(y):
    return np.arcsin(y) * 2 / np.pi


n = 1


@diffeo_torus
def zoom1(X):
    a = iterate(sinzoom, X[0], n)
    b = iterate(sinzoom, X[1], n)
    return np.array([a, b])


@diffeo_torus
def zoom1_inv(X):
    a = iterate(sinzoom_inv, X[0], n)
    b = iterate(sinzoom_inv, X[1], n)
    return np.array([a, b])


@diffeo_torus
def zoom2(X):
    a = iterate(sinzoom, X[0], n)
    b = iterate(sinzoom_inv, X[1], n)
    return np.array([a, b])


@diffeo_torus
def zoom2_inv(X):
    a = iterate(sinzoom_inv, X[0], n)
    b = iterate(sinzoom, X[1], n)
    return np.array([a, b])

