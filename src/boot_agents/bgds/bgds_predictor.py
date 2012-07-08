

class BGDSPredictor():
    
    def __init__(self, bgds_model):
        self.bgds_model = bgds_model 
        
    def process_observations(self, obs):
        self.u = obs['commands']
        self.y = obs['observations']
        
    def predict_y(self, dt):
        y_dot = self.bgds_model.get_y_dot(y=self.y, u=self.u)
        return self.y + y_dot * dt
