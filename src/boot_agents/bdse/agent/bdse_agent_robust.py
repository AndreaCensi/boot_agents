from contracts import contract, describe_type

from boot_agents.bdse import BDSEEstimatorInterface
from boot_agents.robustness import DerivAgentRobust
from bootstrapping_olympics import UnsupportedSpec
from conf_tools import instantiate_spec

from .bdse_predictor import BDSEPredictor
from .servo import BDSEServoInterface


__all__ = ['BDSEAgentRobust']


class BDSEAgentRobust(DerivAgentRobust):
    
    @contract(servo='code_spec', estimator='code_spec')
    def __init__(self, servo, estimator, **others):
        print('Servo: %r' % servo)
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

    def process_observations_robust(self, y, y_dot, u, w):
        self.estimator.update(y=y, y_dot=y_dot, u=u, w=w)
     
    def publish(self, pub):
        with pub.subsection('estimator') as sub:
            if sub:
                self.estimator.publish(sub)
        DerivAgentRobust.publish(self, pub)
             
    def get_predictor(self):
        model = self.estimator.get_model()
        return BDSEPredictor(model)
 
    def merge(self, agent2):
        assert isinstance(agent2, BDSEAgentRobust)
        self.estimator.merge(agent2.estimator)

    def get_servo(self):
        # XXX :repeated code with BDSEAgent
        # print('Servo: %r' % self.servo)
        servo_agent = instantiate_spec(self.servo)
        assert isinstance(servo_agent, BDSEServoInterface)
        servo_agent.init(self.boot_spec)
        model = self.estimator.get_model()
        servo_agent.set_model(model)
        return servo_agent
