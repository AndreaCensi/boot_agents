from . import BDSEstimator2, np
from unittest import TestCase


m = lambda k: np.zeros(k)


class BDSTests(TestCase):

    def test_null(self):
        bds = BDSEstimator2()

        self.assertRaises(Exception, bds.get_T)
        self.assertRaises(Exception, bds.get_M)
        self.assertRaises(Exception, bds.get_yy)
        self.assertRaises(Exception, bds.get_uu)


    def test_should_work1(self):
        bds = BDSEstimator2()
        bds.update(y=m(1), y_dot=m(1), u=m(1), dt=0.1)

    def test_catch_invalid_size_y(self):
        bds = BDSEstimator2()
        self.assertRaises(Exception, bds.update, y=m(1), y_dot=m(2), u=m(0), dt=0.1)

    def test_catch_invalid_size_u(self):
        bds = BDSEstimator2()
        self.assertRaises(Exception, bds.update, y=m(2), y_dot=m(2), u=m(0), dt=0.1)

    def test_catch_invalid_time1(self):
        bds = BDSEstimator2()
        self.assertRaises(Exception, bds.update, y=m(1), y_dot=m(1), u=m(1), dt=0)

    def test_catch_invalid_time2(self):
        bds = BDSEstimator2()
        self.assertRaises(Exception, bds.update, y=m(1), y_dot=m(1), u=m(1), dt= -0.3)

    def test_catch_nan_y(self):
        bds = BDSEstimator2()
        y, y_dot, u = m(5), m(5), m(3)
        y[0] = np.NaN
        self.assertRaises(Exception, bds.update, y=y, y_dot=y_dot, u=u, dt=0.3)

    def test_catch_nan_u(self):
        bds = BDSEstimator2()
        y, y_dot, u = m(5), m(5), m(3)
        u[0] = np.NaN
        self.assertRaises(Exception, bds.update, y=y, y_dot=y_dot, u=u, dt=0.3)

    def test_catch_nan_y_dot(self):
        bds = BDSEstimator2()
        y, y_dot, u = m(5), m(5), m(3)
        y_dot[0] = np.NaN
        self.assertRaises(Exception, bds.update, y=y, y_dot=y_dot, u=u, dt=0.3)

    def test_catch_nan_time(self):
        bds = BDSEstimator2()
        y, y_dot, u, dt = m(5), m(5), m(3), np.NaN
        self.assertRaises(Exception, bds.update, y=y, y_dot=y_dot, u=u, dt=dt)


    def test_consistency(self):
        N = 10
        K = 2
        bds = BDSEstimator2()

        y = np.random.rand(N)
        y_dot = np.random.rand(N)
        u = np.random.rand(K)
        dt = np.random.rand()
        bds.update(y=y, y_dot=y_dot, u=u, dt=dt)

        y2 = np.random.rand(N + 1)
        y_dot2 = np.random.rand(N + 1)
        u2 = np.random.rand(K + 2)
        self.assertRaises(Exception, bds.update, y=y2, y_dot=y_dot, u=u, dt=dt)
        self.assertRaises(Exception, bds.update, y=y, y_dot=y_dot2, u=u, dt=dt)
        self.assertRaises(Exception, bds.update, y=y, y_dot=y_dot, u=u2, dt=dt)

    def test_simple(self):
        N = 10
        K = 2
        T = 10
        bds = BDSEstimator2()
        for t in range(T): #@UnusedVariable
            y = np.random.rand(N)
            y_dot = np.random.rand(N)
            u = np.random.rand(K)
            dt = np.random.rand()
            bds.update(y=y, y_dot=y_dot, u=u, dt=dt)

        bds.get_T()
        bds.get_M()
        bds.get_yy()
        bds.get_uu()
