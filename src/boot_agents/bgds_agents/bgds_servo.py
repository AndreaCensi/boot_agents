from contracts import contract
import numpy as np

__all__ = ['BGDSServo']

class BGDSServo(object):

    def __init__(self, bds_estimator, commands_spec, gain=0.1):
        self.commands_spec = commands_spec
        self.bds_estimator = bds_estimator
        self.y = None
        self.goal = None
        self.gain = gain

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
        if self.y is None or self.goal is None or self.initial_error is None:
            msg = ('Warning: choose_commands() before process_observations()')
            raise Exception(msg)

        raise NotImplementedError()
    
#         error = self.y - self.goal
# 
#         My = np.tensordot(M, self.y, axes=(1, 0))
# 
#         u = -np.tensordot(My, error, axes=(1, 0))
# 
#         # XXX check u=0
#         u = u / np.abs(u).max()
# 
#         # current_error = np.linalg.norm(error)
#         if self.strategy in ['S1', 'S1n', 'S2']:
#             pass
#         elif self.strategy in ['S2d']:
#             initial_error = (self.initial_error
#                              if self.initial_error > 0 else 1)
#             current_error = np.linalg.norm(self.y - self.goal)
#             u = u * current_error / initial_error
#         else:
#             raise Exception('Unknown strategy %r.' % self.strategy)
# 
#         u = clip(u, self.commands_spec)
# 
#         u = u * self.gain
# 
#         u = clip(u, self.commands_spec)
#         return u



