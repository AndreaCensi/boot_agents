from . import BDSEstimator2, bds_dynamics, np
from bootstrapping_olympics.display import ReprepPublisher
import itertools
import sys


def main():
    N = 20
    K = 2

    # M = np.random.rand(N, N, K)
    M = np.zeros((K, N, N))
    for i, j in itertools.product(range(N), range(N)):
        if i <= j:
            M[:, i, j] = 0
        else:
            M[:, i, j] = np.random.rand(K)

    M[:, :, :] = 0
    M[0, 5, 10] = 1
    M[1, 15, 10] = -1

    bds = BDSEstimator2()

    y_mean = np.random.rand(N)
    A = np.random.randn(N, N)
    T = 10000

    Au = np.random.randn(K, K)

    error_M = []
    error_M2 = []
    for t in range(T):
        u = np.dot(Au, np.random.randn(K))
#        if (T - 1) % 100 == 0:
#            u = 0 * u
        y = y_mean + np.dot(A, np.random.randn(N))
        y_dot = bds_dynamics(M, y, u)
        dt = 0.1
        bds.update(y=y, y_dot=y_dot, u=u, dt=dt)

        if t > T / 2:
            bds.fits1.reset()
            bds.fits2.reset()

        if t % 100 == 0:
            Mest = bds.get_M()
            e = np.abs(M - Mest).mean()
            error_M.append(e)
            Mest2, M2info = bds.get_M2() #@UnusedVariable
            e2 = np.abs(M - Mest2).mean()
            error_M2.append(e2)
            sys.stderr.write('%8d/%d %-10.5g %-10.5g  \n' % (t, T, e, e2))
            pass

    rp = ReprepPublisher('bds_estimator_test_learn')
    Mest = bds.get_M()

    Merr = np.abs(M - Mest)
    bds.publish(rp)

    for i in range(K):
        rp.array_as_image('M%s' % i, M[i, :, :])

    for i in range(K):
        rp.array_as_image('est_M%s' % i, Mest[i, :, :])

    for i in range(K):
        rp.array_as_image('Merr%s' % i, Merr[i, :, :])

    with rp.plot('error') as pylab:
        pylab.semilogy(np.array(error_M), 'x')
        pylab.semilogy(np.array(error_M2), 'ro')
        pylab.axis((0, len(error_M), 0, max(error_M) * 1.1))
    rp.r.to_html('bds_estimator_test_learn.html')

if __name__ == '__main__':
    main()
