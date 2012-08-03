from . import np
from boot_agents.utils.mean_covariance import MeanCovariance

class MiscStatistics:
    def __init__(self):
        self.y_stats = MeanCovariance()
        self.y_dot_stats = MeanCovariance()
        self.y_dot_abs_stats = MeanCovariance()
        self.u_stats = MeanCovariance()
        self.dt_stats = MeanCovariance()

    def update(self, y, y_dot, u, dt):
        self.y_stats.update(y, dt)
        self.dt_stats.update(np.array([dt]))
        self.u_stats.update(u, dt)
        self.y_dot_stats.update(y_dot, dt)
        self.y_dot_abs_stats.update(np.abs(y_dot), dt)

    def publish(self, pub):
        self.y_stats.publish(pub.section('y_stats'))
        self.u_stats.publish(pub.section('u_stats'))
        self.y_dot_stats.publish(pub.section('y_dot_stats'))
        self.y_dot_abs_stats.publish(pub.section('y_dot_abs_stats'))
        self.dt_stats.publish(pub.section('dt_stats'))
