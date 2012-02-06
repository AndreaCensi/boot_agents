from . import contract, np


class BDSServo():

    strategies = ['S1', 'S2', 'S1n']

    def __init__(self, bds_estimator, commands_spec,
                 strategy='S1', gain=0.1):
        self.commands_spec = commands_spec
        self.bds_estimator = bds_estimator
        self.y = None
        self.goal = None
        if not strategy in BDSServo.strategies:
            raise Exception('Unknown strategy %r.' % strategy)
        self.strategy = strategy
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
        if self.y is None:
            msg = ('Warning: choose_commands() before process_observations()')
            raise Exception(msg)
            #return self.commands_spec.get_default_value()

        if self.goal is None:
            msg = ('Warning: choose_commands() before set_goal_observations()')
            raise Exception(msg)
            #return self.commands_spec.get_default_value()

        if self.initial_error is None:
            msg = ('Warning: choose_commands() before process_observations()')
            raise Exception(msg)
            #return self.commands_spec.get_default_value()

        error = self.y - self.goal

        if self.strategy == 'S1n':
            # Normalize error, so that we can tolerate discontinuity
            error = np.sign(error) * np.mean(np.abs(error))

        if self.strategy in ['S1', 'S1n']:
            M = self.bds_estimator.get_T()
        elif self.strategy == 'S2':
            M = -self.bds_estimator.get_M()
        else:
            raise Exception('Unknown strategy %r.' % self.strategy)

        My = np.tensordot(M, self.y, axes=(1, 0))

        u = -np.tensordot(My, error, axes=(1, 0))

        u = u / np.abs(u).max()

        u = clip(u, self.commands_spec)

        #current_error = np.linalg.norm(error)
#        eps1 = current_error / self.initial_error

#        eps = 0.25
        u = u * self.gain
        #print('e(k): %10.3f e(k)/e(0) %10.3f u: %s ' %
        #      (current_error, eps1, u))

        #u += np.random.uniform(-1, 1, u.size) * 0.05

        u = clip(u, self.commands_spec)
        return u


def clip(x, stream_spec):
    x = np.maximum(x, stream_spec.streamels['lower'])
    x = np.minimum(x, stream_spec.streamels['upper'])
    return x

