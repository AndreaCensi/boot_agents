from .interface import BDSEServoInterface
from bootstrapping_olympics import BootSpec
from contracts import contract
import numpy as np

__all__ = ['BDSEServo']


class BDSEServo(BDSEServoInterface):
    """ This was used in the BV experiments """
    
    strategies = ['S1', 'S2', 'S1n', 'S2d']
    linpoints = ['current', 'goal', 'middle']
    
    def __init__(self, strategy='S1', gain=0.1, linpoint='current'):
        self.y = None
        self.goal = None
        if not strategy in BDSEServo.strategies:
            raise ValueError('Unknown strategy %r.' % strategy)
        if not linpoint in BDSEServo.linpoints:
            raise ValueError('Unknown linpoint %r.' % linpoint)
        print('Using servo gain %s' % gain)
        self.strategy = strategy
        self.gain = gain
        self.linpoint = linpoint
    
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
        self.u = obs['commands']
        self.y = obs['observations']
        if self.initial_error is None:
            self.initial_error = np.linalg.norm(self.y - self.goal)

    def choose_commands(self): 
        if self.linpoint == 'current':
            u = self.bdse_model.get_servo_descent_direction(self.y, self.goal)
        elif self.linpoint == 'goal':
            u = -self.bdse_model.get_servo_descent_direction(self.goal, self.y)
        elif self.linpoint == 'middle':
            u1 = self.bdse_model.get_servo_descent_direction(self.y, self.goal)
            u2 = -self.bdse_model.get_servo_descent_direction(self.goal, self.y)
            u = 0.5 * u1 + 0.5 * u2
        else:
            raise ValueError('not implemented %r' % self.linpoint)

        u_max = np.abs(u).max()
        if u_max > 0:
            u = u / u_max

        # current_error = np.linalg.norm(error)
        if self.strategy in ['S1', 'S1n', 'S2']:
            pass
        elif self.strategy in ['S2d']:
            initial_error = (self.initial_error
                             if self.initial_error > 0 else 1)
            current_error = np.linalg.norm(self.y - self.goal)
            u = u * current_error / initial_error
        else:
            assert False

        u = clip(u, self.commands_spec)

        u = u * self.gain
 
        u = clip(u, self.commands_spec)
        return u
    
    # this is now in servo simple
    
#     def choose_commands2(self, K=None):
#         warnings.warn('Using choose_commands2')
#         
#         if self.linpoint == 'current':
#             u = self.bdse_model.get_servo_descent_direction(self.y, self.goal)
#         elif self.linpoint == 'goal':
#             u = -self.bdse_model.get_servo_descent_direction(self.goal, self.y)
#         elif self.linpoint == 'middle':
#             u1 = self.bdse_model.get_servo_descent_direction(self.y, self.goal)
#             u2 = -self.bdse_model.get_servo_descent_direction(self.goal, self.y)
#             u = 0.5 * u1 + 0.5 * u2
#         else:
#             raise ValueError('not implemented %r' % self.linpoint)
#         
#         if K is not None:
#             u = u * K
#                 
#         u_raw = u.copy()
#          
# 
#         u_max = np.abs(u).max()
#         if u_max > 0:
#             u = u / u_max
#              
#         u = u * self.gain
#  
#         u = clip(u, self.commands_spec)
#         
#         res = {}
#         res['linpoint'] = self.linpoint
#         res['u_raw'] = u_raw
#         res['u'] = u
#          
#         return res


def clip(x, stream_spec):  # TODO: move away
    x = np.maximum(x, stream_spec.streamels['lower'])
    x = np.minimum(x, stream_spec.streamels['upper'])
    return x

