from boot_agents.utils import DerivativeBox2
from bootstrapping_olympics import PredictorAgentInterface

__all__ = ['BDSEPredictor']


class BDSEPredictor(PredictorAgentInterface):

    def __init__(self, model):
        self.model = model
        self.comp_y_dot = DerivativeBox2()
        
    def process_observations(self, obs):
        self.y = obs['observations']
        self.u = obs['commands']
        dt = obs['dt'].item()
        self.comp_y_dot.update(self.y, dt)
        
    def init(self, boot_spec):
        # TODO: add check
        pass
    
    def predict_y(self, dt):
        y_dot = self.model.get_y_dot(y=self.y, u=self.u)
        return self.y + y_dot * dt

    def estimate_u(self):
        if not self.comp_y_dot.ready():
            msg = 'Too soon to estimate u'
            raise Exception(msg)
        y, y_dot = self.comp_y_dot.get_value()
        u_est = self.model.estimate_u(y=y, y_dot=y_dot)
        return u_est
        
