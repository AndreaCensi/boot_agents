import numpy as np
from boot_agents.diffeo import DiffeomorphismEstimator
from bootstrapping_olympics.ros_scripts.log_learn.reprep_publisher import ReprepPublisher
import time

def diffeo_estimator_test1():
    
    shape = (50, 40)
    de = DiffeomorphismEstimator([0.1, 0.1])
    y = np.zeros(shape)
    de.update(y, y)
    K = 50
    t0 = time.clock()
    for k in range(K):
        y = np.random.rand(*shape)
        y1 = y + np.random.rand(*shape)
        de.update(y, y1)
    t1 = time.clock()
    print('%.2f fps' % (K / (t1 - t0)))
    rp = ReprepPublisher('diffeo_estimator_test1')
    de.publish_debug(rp)

    filename = 'out/diffeo_estimator_test1.html'
    print('Writing to %r' % filename)
    rp.r.to_html(filename)
