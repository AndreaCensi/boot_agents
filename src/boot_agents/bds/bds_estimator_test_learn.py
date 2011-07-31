import numpy as np
from boot_agents.bds.bds_estimator import BDSEstimator2
from bootstrapping_olympics.ros_scripts.log_learn.reprep_publisher import ReprepPublisher
import sys
import itertools

def main():   
    N = 20
    K = 2
    
#    M = np.random.rand(N, N, K)
    M = np.zeros((K, N, N))
    for i, j in itertools.product(range(N), range(N)):
        if i <= j:
            M[:, i, j] = 0
        else:
            M[:, i, j] = np.random.rand(K) 
    bds = BDSEstimator2()
    
    y_mean = np.random.rand(N)
    A = np.random.randn(N, N)
    T = 10000
    
    error_M = []
    for t in range(T):
        u = np.random.randn(K)
        y = y_mean + np.dot(A, np.random.randn(N))
        y_dot = np.dot(u, np.dot(M, y))
        dt = 0.1
        bds.update(y=y, y_dot=y_dot, u=u, dt=dt)

        if t % 100 == 0:
            Mest = bds.get_M()
            e = np.abs(M - Mest).mean()
            error_M.append(e)    
            sys.stderr.write('%8d/%d %-10.5g \n' % (t, T, e))
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
        pylab.axis((0, len(error_M), 0, max(error_M) * 1.1))
    rp.r.to_html('bds_estimator_test_learn.html')
    
if __name__ == '__main__':
    main()
