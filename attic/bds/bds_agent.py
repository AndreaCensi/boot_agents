from . import BDSEstimator2, np, BDSServo
from ..simple_stats import ExpSwitcher
from ..utils import MeanCovariance, DerivativeBox, RemoveDoubles
from bootstrapping_olympics import UnsupportedSpec


__all__ = ['BDSAgent']


class BDSAgent(ExpSwitcher):
    '''
        Skip: only consider every $skip observations. 
    '''
    def __init__(self, beta, skip=1, change_fraction=0.0, servo={}):
        ExpSwitcher.__init__(self, beta)
        self.skip = skip
        self.change_fraction = change_fraction
        self.servo = servo

    def init(self, boot_spec):
        if len(boot_spec.get_observations().shape()) != 1:
            raise UnsupportedSpec('This agent can only work with 1D signals.')

        self.count = 0
        self.rd = RemoveDoubles(self.change_fraction)
        self.y_deriv = DerivativeBox()
        self.bds_estimator = BDSEstimator2()
        self.y_stats = MeanCovariance()

        # All the rest are only statistics
        self.y_dot_stats = MeanCovariance()
        self.y_dot_abs_stats = MeanCovariance()
        self.u_stats = MeanCovariance()
        self.dt_stats = MeanCovariance()

        ExpSwitcher.init(self, boot_spec)
        self.commands_spec = boot_spec.get_commands()

    def process_observations(self, obs):
        ExpSwitcher.process_observations(self, obs)

        self.count += 1
        if self.count % self.skip != 0:
            return

        dt = float(obs['dt'])
        y = obs['observations']
        u = obs['commands']

        self.rd.update(y)
        if not self.rd.ready():
            #self.info('Skipping observation because double observations %s'
            # % self.count)
            return

        # XXX: this is not `dt` anymore FiXME:
        self.y_stats.update(y, dt)

        if obs['episode_start']:
            #self.info('episode_changed: %s' % obs['id_episode'])
            self.y_deriv.reset()
            return

        self.y_deriv.update(y, dt)

        if not self.y_deriv.ready():
            return

        y_sync, y_dot_sync = self.y_deriv.get_value()
        y_mean = self.y_stats.get_mean()
        y_n = y_sync - y_mean
        self.bds_estimator.update(u=u.astype('float32'),
                                  y=y_n.astype('float32'),
                                  y_dot=y_dot_sync.astype('float32'),
                                  dt=dt)

        # Just other statistics
        self.dt_stats.update(np.array([dt]))
        self.u_stats.update(u, dt)
        self.y_dot_stats.update(y_dot_sync, dt)
        self.y_dot_abs_stats.update(np.abs(y_dot_sync), dt)

    def publish(self, pub):
        if self.count < 10:
            self.info('Skipping publishing as count=%d' % self.count)
            return
 
        self.bds_estimator.publish(pub) 

        self.y_stats.publish(pub.section('y_stats'))
        self.u_stats.publish(pub.section('u_stats'))
        self.y_dot_stats.publish(pub.section('y_dot_stats'))
        self.y_dot_abs_stats.publish(pub.section('y_dot_abs_stats')) 

    def get_predictor(self):
        from boot_agents.bds.bds_predictor import BDSPredictor
        return BDSPredictor(self.bds_estimator)

    def get_servo(self):
        return BDSServo(self.bds_estimator, self.commands_spec, **self.servo)


