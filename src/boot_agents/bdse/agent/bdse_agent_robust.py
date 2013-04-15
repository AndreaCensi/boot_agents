from bootstrapping_olympics import UnsupportedSpec 
from boot_agents.robustness.deriv_agent_robust import DerivAgentRobust
from boot_agents.bdse.agent.bdse_servo import BDSEServo
from boot_agents.bdse.model.bdse_estimator_robust import BDSEEstimatorRobust
from boot_agents.bdse.agent.bdse_predictor import BDSEPredictor
        
        
class BDSEAgentRobust(DerivAgentRobust):
    
    def __init__(self, rcond=1e-8, servo={}, **others):
        DerivAgentRobust.__init__(self, **others)
        self.servo = servo
        self.rcond = rcond
        
    def init(self, boot_spec):
        DerivAgentRobust.init(self, boot_spec)
        shape = boot_spec.get_observations().shape()
        if len(shape) != 1:
            msg = 'This agent only works with 1D signals'
            raise UnsupportedSpec(msg)

        self.estimator = BDSEEstimatorRobust(rcond=self.rcond)
        
        self.commands_spec = boot_spec.get_commands()

    
    def process_observations_robust(self, y, y_dot, u, w):
        self.estimator.update(y=y, y_dot=y_dot, u=u, w=w)
     
    def publish(self, pub):
        self.estimator.publish(pub.section('bdse_estimator')) 
        DerivAgentRobust.publish(self, pub.section('deriv_agent_robust'))
        
    def get_predictor(self):
        model = self.estimator.get_model()
        return BDSEPredictor(model)
    
    def get_servo(self):
        model = self.estimator.get_model()
        return BDSEServo(model, self.commands_spec, **self.servo)

