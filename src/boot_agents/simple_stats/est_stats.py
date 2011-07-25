from . import ExpSwitcher
from ..utils import MeanCovariance
from boot_agents.utils.statistics import cov2corr
import numpy as np

class EstStats(ExpSwitcher):
    ''' A simple agent that estimates the covariance of the observations. '''
    
    def init(self, sensels_shape, commands_spec):
        ExpSwitcher.init(self, sensels_shape, commands_spec)
        if len(sensels_shape) != 1:
            raise ValueError('I assume 1D signals.')
            
        #self.num_sensels = sensels_shape[0]
        self.y_stats = MeanCovariance()
        
        self.info('Agent %s initialized.' % self)

    def process_observations(self, obs):
        self.y_stats.update(value=obs.sensel_values, dt=obs.dt)
        
    def get_state(self):
        return dict(y_stats=self.y_stats)
    
    def set_state(self, state):
        self.y_stats = state['y_stats']
    
    def publish(self, pub):
        Py = self.y_stats.get_covariance()
        Ry = self.y_stats.get_correlation()
        Py_inv = self.y_stats.get_information()
        Ey = self.y_stats.get_mean()
        y_max = self.y_stats.get_maximum()
        y_min = self.y_stats.get_minimum()
        
        pub.text('stats', 'Num samples: %s' % self.y_stats.mean_accum.num_samples)
        pub.array_as_image('Py', Py)
        Ry0 = Ry.copy()
        np.fill_diagonal(Ry0, np.NaN)
        
        pub.array_as_image('Ry', Ry)
        pub.array_as_image('Py_inv', Py_inv)
        pub.array_as_image('Py_inv_n', cov2corr(Py_inv))
        
        with pub.plot(name='y_stats') as pylab:
            pylab.plot(Ey, label='E(y)')
            pylab.plot(y_max, label='y_max')
            pylab.plot(y_min, label='y_min')
            pylab.legend()

        with pub.plot(name='y_stats_log') as pylab:
            pylab.semilogy(Ey, label='E(y)')
            pylab.semilogy(y_max, label='y_max')
            pylab.semilogy(y_min, label='y_min')
            pylab.legend()


