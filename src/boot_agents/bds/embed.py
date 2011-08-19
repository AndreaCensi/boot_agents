from . import ExpSwitcher
from ..utils import MeanCovariance
import numpy as np
from geometry import   inner_product_embedding
from boot_agents.bds.bds_agent import DerivativeBox
from ..utils import scale_score


class Embed(ExpSwitcher):
    ''' A simple agent that estimates the covariance of the observations. '''
    
    def init(self, sensels_shape, commands_spec):
        ExpSwitcher.init(self, sensels_shape, commands_spec)
        if len(sensels_shape) != 1:
            raise ValueError('I assume 1D signals.')
            
        self.y_stats = MeanCovariance()
        self.z_stats = MeanCovariance()
        self.count = 0
        self.y_deriv = DerivativeBox()
        
    def process_observations(self, obs):
        y = obs.sensel_values
        self.y_stats.update(obs.sensel_values, obs.dt)
        self.count += 1
        
        self.y_deriv.update(y, obs.dt)
        if self.y_deriv.ready():
            y, y_dot = self.y_deriv.get_value()
            self.z_stats.update(np.abs(y_dot), obs.dt)
        
    state_vars = ['y_stats', 'z_stats', 'count', 'y_deriv']
    def get_state(self):
        return self.get_state_vars(Embed.state_vars)
    
    def set_state(self, state):
        return self.set_state_vars(state, Embed.state_vars)
    
    def publish(self, pub):
        if self.count < 10: return
        R = self.z_stats.get_correlation()
        Dis = discretize(-R, 2)
        np.fill_diagonal(Dis, 0)
        pub.array_as_image('Dis', Dis)
        R = R * R
        C = np.maximum(R, 0)

        S = inner_product_embedding(Dis, 2)
        for i in range(R.shape[0]):
            R[i, i] = np.NaN
            C[i, i] = np.NaN
        pub.array_as_image('R', R)
        pub.array_as_image('C', C)
        with pub.plot(name='S') as pylab:
            pylab.plot(S[0, :], S[1, :], '.')
        self.z_stats.publish(pub, 'z_stats')
        


def discretize(M, w):
    X = np.zeros(M.shape, dtype='float32')
    for i in range(M.shape[0]):
        score = scale_score(M[i, :])
        which, = np.nonzero(score <= w)
        X[i, which] += 1 
    for j in range(M.shape[0]):
        score = scale_score(M[:, j])
        which, = np.nonzero(score <= w)
        X[which, j] += 1 
    return X 

