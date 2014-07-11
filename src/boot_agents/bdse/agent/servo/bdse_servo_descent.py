from abc import abstractmethod

from contracts import contract

from bootstrapping_olympics import BootSpec
import numpy as np

from .interface import BDSEServoInterface


__all__ = ['BDSEServoFromDescent']


class BDSEServoFromDescent(BDSEServoInterface):
    """ All strategies that return a descent direction which is clipped. """

    def __init__(self, gain=0.1):
        self.gain = gain

    @abstractmethod
    @contract(returns='array', observations='array', goal='array')
    def get_descent_direction(self, observations, goal):
        pass
      
    @contract(boot_spec=BootSpec)
    def init(self, boot_spec):
        self.boot_spec = boot_spec
        self.commands_spec = boot_spec.get_commands()
        
    def set_model(self, model):
        self.bdse_model = model
          
    @contract(goal='array')
    def set_goal_observations(self, goal):
        self._goal = goal

    def process_observations(self, obs):
        self._y = obs['observations']
 
    def choose_commands_ext(self, K=None):
        u = self.get_descent_direction(observations=self._y, goal=self._goal)
        res = {}
        res['descent'] = u.copy()
            
        if K is not None:
            u = u * K
                
        u_raw = u.copy()
        u_max = np.abs(u).max()
        if u_max > 0:
            u = u / u_max

        u = clip(u, self.commands_spec)
        u = u * self.gain
        u = clip(u, self.commands_spec)
        
        res['u_raw'] = u_raw.copy()
        res['u'] = u.copy()
        return res

    def choose_commands(self):
        res = self.choose_commands_ext()
        return res['u']


def clip(x, stream_spec):  # TODO: move away
    x = np.maximum(x, stream_spec.streamels['lower'])
    x = np.minimum(x, stream_spec.streamels['upper'])
    return x

