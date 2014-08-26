from .bdse_predictor import BDSEPredictor
from .misc_statistics import MiscStatistics
from .servo import BDSEServoInterface
from bdse import BDSEEstimatorInterface, get_conftools_bdse_estimators
from blocks import Sink, check_timed_named
from boot_agents import DerivAgent
from boot_agents.utils import MeanCovariance
from bootstrapping_olympics import (PredictingAgent, ServoingAgent, 
    UnsupportedSpec)
from conf_tools import instantiate_spec
from contracts import check_isinstance, contract


__all__ = [
    'BDSEAgent2',
]


class BDSEAgent2(DerivAgent,  
                 ServoingAgent,
                 PredictingAgent):
    '''
        This agent needs to have pre-computed derivative.
    '''

    @contract(servo='code_spec', estimator='str|code_spec')
    def __init__(self, servo, estimator):
        """
            :param servo: extra parameters for servo; if string, the ID of an agent.
                
        """
        self.servo = servo
        _, self.bdse_estimator = get_conftools_bdse_estimators().instance_smarter(estimator) 
        check_isinstance(self.bdse_estimator, BDSEEstimatorInterface)

    def init(self, boot_spec):
        self.boot_spec = boot_spec

        obs_spec = boot_spec.get_observations()
        if len(obs_spec.shape()) != 1:
            msg = 'This agent can only work with 1D signals, got %s.' % boot_spec
            raise UnsupportedSpec(msg)

        self.y_stats = MeanCovariance()

        self.commands_spec = boot_spec.get_commands()

        # All the rest are only statistics
        self.stats = MiscStatistics()

    def get_servo_system(self):
        raise NotImplementedError()
    
    @contract(returns=Sink)
    def get_learner_u_y_y_dot(self):
        """ 
            Returns a Sink that receives dictionaries
            dict(y=..., y_dot=..., u=...)
        """
        
        class MySync(Sink):
            def __init__(self, agent):
                self.agent = agent
            def reset(self):
                pass
            def put(self, value, block=False, timeout=None):  # @UnusedVariable
                check_timed_named(value)
                (_, (name, x)) = value  # @UnusedVariable
                check_isinstance(x, dict)
                
                y = x['y_dot']
                y_dot = x['y']
                u = x['u']
                
                u=u.astype('float32')
                y=y.astype('float32')
                y_dot=y_dot.astype('float32')
                
                self.agent.bdse_estimator.update(u=u, y=y,y_dot=y_dot, w=1.0)

                # Just other statistics
                self.agent.y_stats.update(y)
                self.agent.stats.update(y, y_dot, u, 1.0)

        return MySync(self) 

    def display(self, report):
        if not 'boot_spec' in self.__dict__:
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




# def check_composite_signal_and_deriv(stream_spec):
#     """ Checks that this is a composite stream with 
#         two components 'signal' and 'signal_deriv'. """
#     if not isinstance(stream_spec, CompositeStreamSpec):
#         msg = 'Expected composite, got %r' % stream_spec
#         raise UnsupportedSpec(msg)
#     comps = stream_spec.get_components()
#     expected = ['signal', 'signal_deriv']
#     if not set(comps) == set(expected):
#         msg = 'Expected %s components, got %s.' % (set(comps), expected)
#         raise UnsupportedSpec(msg)
