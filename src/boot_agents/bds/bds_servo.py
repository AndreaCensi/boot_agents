from . import contract, np


class BDSServo():

    strategies = ['S1', 'S2', 'S1n', 'S2d']

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
        if self.y is None or self.goal is None or self.initial_error is None:
            msg = ('Warning: choose_commands() before process_observations()')
            raise Exception(msg)

        error = self.y - self.goal

        if self.strategy == 'S1n':
            # Normalize error, so that we can tolerate discontinuity
            error = np.sign(error) * np.mean(np.abs(error))

        if self.strategy in ['S1', 'S1n']:
            M = self.bds_estimator.get_T()
        elif self.strategy in  ['S2', 'S2d']:
            M = -self.bds_estimator.get_M()
        else:
            raise Exception('Unknown strategy %r.' % self.strategy)

        My = np.tensordot(M, self.y, axes=(1, 0))

        u = -np.tensordot(My, error, axes=(1, 0))

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
            raise Exception('Unknown strategy %r.' % self.strategy)

        u = clip(u, self.commands_spec)

        u = u * self.gain

        u = clip(u, self.commands_spec)
        return u


def clip(x, stream_spec): # TODO: move away
    x = np.maximum(x, stream_spec.streamels['lower'])
    x = np.minimum(x, stream_spec.streamels['upper'])
    return x

