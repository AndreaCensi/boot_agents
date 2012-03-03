from . import np, diffeo_torus

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
