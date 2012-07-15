from . import contract, np


class BDSEServo():

    strategies = ['S1', 'S2', 'S1n', 'S2d']
    linpoints = ['current', 'goal', 'middle']
    
    def __init__(self, bdse_model, commands_spec,
                 strategy='S1', gain=0.1, linpoint='current'):
        self.commands_spec = commands_spec
        self.bdse_model = bdse_model
        self.y = None
        self.goal = None
        if not strategy in BDSEServo.strategies:
            raise ValueError('Unknown strategy %r.' % strategy)
        if not linpoint in BDSEServo.linpoints:
            raise ValueError('Unknown linpoint %r.' % linpoint)
        self.strategy = strategy
        self.gain = gain
        self.linpoint = linpoint
        
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
        
#        error = self.y - self.goal

#        if self.strategy == 'S1n':
#            # Normalize error, so that we can tolerate discontinuity
#            error = np.sign(error) * np.mean(np.abs(error))

#        if self.strategy in ['S1', 'S1n']:
#            M = self.bds_estimator.get_T()
#        elif self.strategy in  ['S2', 'S2d']:
#            M = -self.bds_estimator.get_M()
#        else:
#            raise Exception('Unknown strategy %r.' % self.strategy)
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

        # XXX check u=0
        u = u / np.abs(u).max()

        #current_error = np.linalg.norm(error)
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

#        print self.gain, u
        u = clip(u, self.commands_spec)
        return u


def clip(x, stream_spec): # TODO: move away
    x = np.maximum(x, stream_spec.streamels['lower'])
    x = np.minimum(x, stream_spec.streamels['upper'])
    return x

