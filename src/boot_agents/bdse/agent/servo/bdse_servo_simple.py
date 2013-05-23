from .interface import BDSEServoInterface
from bootstrapping_olympics import BootSpec
from contracts import contract
import numpy as np

__all__ = ['BDSEServoSimple']



class BDSEServoSimple(BDSEServoInterface):
    """ This is the simplest thing that works. """
    
    def __init__(self, gain=0.1):
        self.gain = gain
      
    @contract(boot_spec=BootSpec)
    def init(self, boot_spec):
        self.boot_spec = boot_spec
        self.commands_spec = boot_spec.get_commands()
        
    def set_model(self, model):
        self.bdse_model = model
          
    @contract(goal='array')
    def set_goal_observations(self, goal):
        self.goal = goal
        self.initial_error = None

    def process_observations(self, obs):
        self.y = obs['observations']
        if self.initial_error is None:
            self.initial_error = np.linalg.norm(self.y - self.goal)
 
    def choose_commands_ext(self, K=None):
        u = self.bdse_model.get_servo_descent_direction(self.y, self.goal)        
        if K is not None:
            u = u * K
                
        u_raw = u.copy()
        u_max = np.abs(u).max()
        if u_max > 0:
            u = u / u_max

        u = clip(u, self.commands_spec)
        u = u * self.gain
        u = clip(u, self.commands_spec)
        
        res = {}
        res['u_raw'] = u_raw
        res['u'] = u
         
        return res

    def choose_commands(self):
        res = self.choose_commands_ext()
        return res['u']
        

def clip(x, stream_spec):  # TODO: move away
    x = np.maximum(x, stream_spec.streamels['lower'])
    x = np.minimum(x, stream_spec.streamels['upper'])
    return x

