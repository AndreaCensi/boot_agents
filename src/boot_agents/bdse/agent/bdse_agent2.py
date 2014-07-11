from contracts import contract, describe_type

from boot_agents.bdse.model import BDSEEstimatorInterface
from boot_agents.utils import MeanCovariance
from bootstrapping_olympics import (BasicAgent, LearningAgent,
    PredictingAgent, ServoingAgent, ExploringAgent)
from bootstrapping_olympics import (UnsupportedSpec,
    get_boot_config)
from conf_tools import instantiate_spec
from streamels import CompositeStreamSpec

from .bdse_predictor import BDSEPredictor
from .misc_statistics import MiscStatistics
from .servo import BDSEServoInterface


__all__ = ['BDSEAgent2']

def check_composite_signal_and_deriv(stream_spec):
    """ Checks that this is a composite stream with 
        two components 'signal' and 'signal_deriv'. """
    if not isinstance(stream_spec, CompositeStreamSpec):
        msg = 'Expected composite, got %r' % stream_spec
        raise UnsupportedSpec(msg)
    comps = stream_spec.get_components()
    expected = ['signal', 'signal_deriv']
    if not set(comps) == set(expected):
        msg = 'Expected %s components, got %s.' % (set(comps), expected)
        raise UnsupportedSpec(msg)


class BDSEAgent2(BasicAgent, ExploringAgent, LearningAgent, ServoingAgent, PredictingAgent):
    '''
        This agent needs to have pre-computed derivative.
    '''

    @contract(servo='code_spec', estimator='code_spec')
    def __init__(self, explorer, servo, estimator,
                 change_fraction=0.0):
        """
            :param explorer: ID of the explorer agent.
            :param servo: extra parameters for servo; if string, the ID of an agent.
                
        """
        boot_config = get_boot_config()
        _, self.explorer = boot_config.agents.instance_smarter(explorer)  # @UndefinedVariable

        self.change_fraction = change_fraction
        self.servo = servo
        self.estimator_spec = estimator

    def init(self, boot_spec):
        self.boot_spec = boot_spec

        obs_spec = boot_spec.get_observations()
        check_composite_signal_and_deriv(obs_spec)
        signal_spec = obs_spec.get_components()['signal']
        if len(signal_spec.shape()) != 1:
            msg = 'This agent can only work with 1D signals, got %s.' % boot_spec
            raise UnsupportedSpec(msg)

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

    def process_observations(self, bd):
        self.explorer.process_observations(bd)

        observations = bd['observations']
        y = observations['signal']
        y_dot = observations['signal_deriv']
        u = bd['commands']

        self.y_stats.update(y)

        self.bdse_estimator.update(u=u.astype('float32'),
                                   y=y.astype('float32'),
                                   y_dot=y_dot.astype('float32'),
                                   w=1.0)

        # Just other statistics
        self.stats.update(y, y_dot, u, 1.0)

    def display(self, report):
        if not 'bdse_estimator' in self.__dict__:
            report.text('warn', 'not init()ed yet: %s' % self.__dict__)
            return
        with report.subsection('estimator') as sub:
            if sub:
                self.bdse_estimator.publish(sub)

        with report.subsection('stats') as sub:
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
        assert isinstance(agent2, BDSEAgent2)
        self.bdse_estimator.merge(agent2.bdse_estimator)


