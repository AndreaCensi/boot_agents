from boot_agents.bgds.bgds_predictor import BGDSPredictor
from bootstrapping_olympics.interfaces import UnsupportedSpec 
from boot_agents.robustness.deriv_agent_robust import DerivAgentRobust
from boot_agents.bgds.bgds_estimator_robust import BGDSEstimator1DRobust
from .bgds_servo import BGDSServo
        
        
class BGDSAgent1DRobust(DerivAgentRobust):
    
    def init(self, boot_spec):
        # TODO: do the 2D version
        shape = boot_spec.get_observations().shape()
        if len(shape) != 1:
            msg = 'This agent only works with 1D signals'
            raise UnsupportedSpec(msg)

        self.bgds_estimator = BGDSEstimator1DRobust()
    
    def process_observations_robust(self, y, y_dot, u, w):
        self.bgds_estimator.update(y=y, y_dot=y_dot, u=u, w=w)
     
    def publish(self, pub):
        DerivAgentRobust.publish(self, pub.section('deriv_agent_robust'))
        self.bgds_estimator.publish(pub.section('bgds_estimator')) 

    def get_predictor(self):
        model = self.bgds_estimator.get_model()
        return BGDSPredictor(model)
    
    def get_servo(self):
        model = self.bgds_estimator.get_model()
        return BGDSServo(model)

   
