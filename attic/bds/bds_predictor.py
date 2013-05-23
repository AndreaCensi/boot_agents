from . import bds_dynamics


class BDSPredictor():
    
    def __init__(self, bds_estimator):
        self.bds_estimator = bds_estimator
        self.M = bds_estimator.get_M()
        
    def process_observations(self, obs):
        self.u = obs['commands']
        self.y = obs['observations']
        
    def predict_y(self, dt):
        y_dot = bds_dynamics(self.M, y=self.y, u=self.u)
        return self.y + y_dot * dt
