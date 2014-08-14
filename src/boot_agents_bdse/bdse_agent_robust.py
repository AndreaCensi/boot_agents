from .bdse_predictor import BDSEPredictor
from .servo import BDSEServoInterface
from bdse import BDSEEstimatorInterface
from blocks import Sink, check_timed_named
from boot_agents import DerivAgentRobust
from bootstrapping_olympics import (PredictingAgent, ServoingAgent, 
    UnsupportedSpec)
from conf_tools import instantiate_spec
from contracts import check_isinstance, contract, describe_type

__all__ = [
    'BDSEAgentRobust',
]


class BDSEAgentRobust(DerivAgentRobust,
                      ServoingAgent,
                      PredictingAgent):
    
    @contract(servo='code_spec', estimator='code_spec')
    def __init__(self, servo, estimator, **others):
        DerivAgentRobust.__init__(self, **others)
        self.servo = servo
        self.estimator_spec = estimator
        
    def init(self, boot_spec):
        DerivAgentRobust.init(self, boot_spec)
        shape = boot_spec.get_observations().shape()
        if len(shape) != 1:
            msg = 'This agent only works with 1D signals'
            raise UnsupportedSpec(msg)

        self.estimator = instantiate_spec(self.estimator_spec)
        if not isinstance(self.estimator, BDSEEstimatorInterface):
            msg = ('Expected a BDSEEstimatorInterface, got %s' 
                   % describe_type(self.estimator))
            raise ValueError(msg)

        self.commands_spec = boot_spec.get_commands()

    def state_vars(self):
        return ['estimator']

    def get_learner_u_y_y_dot_w(self):
         
        class MySink(Sink):
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
                w = x['w']
                
                u=u.astype('float32')
                y=y.astype('float32')
                y_dot=y_dot.astype('float32')
                
                self.agent.estimator.update(u=u, y=y,y_dot=y_dot, w=w)

        return MySink(self) 
     
    def get_servo_system(self):
        raise NotImplementedError()
    
    def publish(self, pub):
        with pub.subsection('estimator') as sub:
            if sub:
                self.estimator.publish(sub)
        DerivAgentRobust.publish(self, pub)
             
 
    def merge(self, agent2):
        assert isinstance(agent2, BDSEAgentRobust)
        self.estimator.merge(agent2.estimator)

    def get_predictor(self):
        if self.count == 0:
            msg = 'get_servo() called but count == 0.'
            raise ValueError(msg)

        model = self.estimator.get_model()
        return BDSEPredictor(model)

    def get_servo(self):
        # XXX :repeated code with BDSEAgent
        # print('Servo: %r' % self.servo)
        if self.count == 0:
            msg = 'get_servo() called but count == 0.'
            raise ValueError(msg)

        servo_agent = instantiate_spec(self.servo)
        assert isinstance(servo_agent, BDSEServoInterface)
        servo_agent.init(self.boot_spec)
        model = self.estimator.get_model()
        servo_agent.set_model(model)
        return servo_agent
