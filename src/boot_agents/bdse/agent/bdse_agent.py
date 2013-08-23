from .bdse_predictor import BDSEPredictor
from .misc_statistics import MiscStatistics
from .servo import BDSEServoInterface
from boot_agents.bdse.model import BDSEEstimatorInterface
from boot_agents.utils import DerivativeBox, MeanCovariance, RemoveDoubles
from bootstrapping_olympics import (AgentInterface, UnsupportedSpec,
    get_boot_config)
from conf_tools import instantiate_spec
from contracts import contract, describe_type


__all__ = ['BDSEAgent']


class BDSEAgent(AgentInterface):
    '''
        An agent that uses a BDS model.
    '''
    
    @contract(servo='code_spec', estimator='code_spec')
    def __init__(self, explorer, servo, estimator, skip=1,
                 change_fraction=0.0):
        """
            :param explorer: ID of the explorer agent.
            :param servo: extra parameters for servo; if string, the ID of an agent.
                
            :param skip: only used one every skip observations.
        """
        boot_config = get_boot_config()
        _, self.explorer = boot_config.agents.instance_smarter(explorer)  # @UndefinedVariable
        
        self.skip = skip
        self.change_fraction = change_fraction
        self.servo = servo
        self.estimator_spec = estimator

    def init(self, boot_spec):
        self.boot_spec = boot_spec
        
        if len(boot_spec.get_observations().shape()) != 1:
            raise UnsupportedSpec('This agent can only work with 1D signals.')

        self.count = 0
        self.rd = RemoveDoubles(self.change_fraction)
        self.y_deriv = DerivativeBox()

        self.bdse_estimator = instantiate_spec(self.estimator_spec)
        if not isinstance(self.bdse_estimator, BDSEEstimatorInterface):
            msg = ('Expected a BDSEEstimatorInterface, got %s' 
                   % describe_type(self.estimator))
            raise ValueError(msg)

    
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
            # self.info('episode_changed: %s' % obs['id_episode'])
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

    def publish(self, pub):
        if self.count < 10:
            self.info('Skipping publishing as count=%d' % self.count)
            return
        
        with pub.subsection('estimator') as sub:
            if sub:
                self.bdse_estimator.publish(sub)
            
        with pub.subsection('stats') as sub:
            if sub:
                self.stats.publish(sub)

    def get_predictor(self):
        model = self.bdse_estimator.get_model()
        return BDSEPredictor(model)

    def get_servo(self):
        servo_agent = instantiate_spec(self.servo)
        servo_agent.init(self.boot_spec)
        assert isinstance(servo_agent, BDSEServoInterface)
        model = self.bdse_estimator.get_model()
        servo_agent.set_model(model)
        return servo_agent

    def merge(self, agent2):
        assert isinstance(agent2, BDSEAgent)
        self.bdse_estimator.merge(agent2.bdse_estimator)


