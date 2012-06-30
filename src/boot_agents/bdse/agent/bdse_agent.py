from . import np
from .. import BDSEEstimator
from boot_agents.utils import DerivativeBox, MeanCovariance, RemoveDoubles
from bootstrapping_olympics import (AgentInterface, BootOlympicsConfig,
    UnsupportedSpec)


__all__ = ['BDSEAgent']


class BDSEAgent(AgentInterface):
    '''
        Skip: only consider every $skip observations. 
    '''
    def __init__(self, explorer, skip=1, change_fraction=0.0, servo={}):
        """
            :param explorer: ID of the explorer agent.
            :param servo: extra parameters for servo.
        """
        agents = BootOlympicsConfig.agents
        self.explorer = agents.instance(explorer) #@UndefinedVariable
        self.skip = skip
        self.change_fraction = change_fraction
        self.servo = servo

    def init(self, boot_spec):
        if len(boot_spec.get_observations().shape()) != 1:
            raise UnsupportedSpec('This agent can only work with 1D signals.')

        self.count = 0
        self.rd = RemoveDoubles(self.change_fraction)
        self.y_deriv = DerivativeBox()
        self.bdse_estimator = BDSEEstimator()
        self.y_stats = MeanCovariance()

        self.explorer.init(boot_spec)
        self.commands_spec = boot_spec.get_commands()

        # All the rest are only statistics
        self.stats = MiscStatistics()

    def choose_commands(self):
        return self.explorer.choose_commands()

    def process_observations(self, obs):
        self.explorer.process_observations(obs)

        self.count += 1
        if self.count % self.skip != 0:
            return

        dt = float(obs['dt'])
        y = obs['observations']
        u = obs['commands']

        # TODO: abstract away
        self.rd.update(y)
        if not self.rd.ready():
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

        self.bdse_estimator.update(u=u.astype('float32'),
                                  y=y_sync.astype('float32'),
                                  y_dot=y_dot_sync.astype('float32'),
                                  w=dt)

        # Just other statistics
        self.stats.update(y_sync, y_dot_sync, u, dt)

    def publish(self, publisher):
        if self.count < 10:
            self.info('Skipping publishing as count=%d' % self.count)
            return

        self.bdse_estimator.publish(publisher.section('estimator'))
        self.stats.publish(publisher.section('stats'))

#    def get_predictor(self):
#        from boot_agents.bds.bds_predictor import BDSPredictor
#        return BDSPredictor(self.bds_estimator)
#
#    def get_servo(self):
#        return BDSServo(self.bds_estimator, self.commands_spec, **self.servo)


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


