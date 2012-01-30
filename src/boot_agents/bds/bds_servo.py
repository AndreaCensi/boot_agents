from . import contract, np


class BDSServo():

    def __init__(self, bds_estimator, commands_spec):
        self.commands_spec = commands_spec
        self.bds_estimator = bds_estimator
        self.y = None
        self.goal = None

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
            print('Warning: choose_commands() before process_observations()')
            return self.commands_spec.get_default_value()

        if self.goal is None:
            print('Warning: choose_commands() before set_goal_observations()')
            return self.commands_spec.get_default_value()

        if self.initial_error is None:
            print('Warning: choose_commands() before process_observations()')
            return self.commands_spec.get_default_value()

        error = self.y - self.goal

        current_error = np.linalg.norm(error)

#        print('y', self.y.shape)
        M = -self.bds_estimator.get_T()
#        print('M', M.shape)
        My = np.tensordot(M, self.y, axes=(1, 0))
#        print('My', My.shape)
#        print('error', error.shape)
        u = np.tensordot(My, error, axes=(1, 0))

        u = u / np.abs(u).max()

        u = clip(u, self.commands_spec)
        #print('clip(u)', u)
#        eps = 0.1
        eps1 = current_error / self.initial_error
#        eps = 0.1
        eps = 0.25
        u = u * eps
        #print('e(k): %10.3f e(k)/e(0) %10.3f u: %s ' %
        #      (current_error, eps1, u))

        #u += np.random.uniform(-1, 1, u.size) * 0.05

        u = clip(u, self.commands_spec)
        return u


def clip(x, stream_spec):
    x = np.maximum(x, stream_spec.streamels['lower'])
    x = np.minimum(x, stream_spec.streamels['upper'])
    return x

