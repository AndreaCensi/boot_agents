from . import ExpSwitcher, np
from ..utils import MeanCovariance, cov2corr
from bootstrapping_olympics import UnsupportedSpec

__all__ = ['EstStats']

class EstStats(ExpSwitcher):
    ''' A simple agent that estimates various statistics of the observations. '''
    
    def init(self, boot_spec):
        ExpSwitcher.init(self, boot_spec)
        if len(boot_spec.get_observations().shape()) != 1:
            raise UnsupportedSpec('I assume 1D signals.')
            
        self.y_stats = MeanCovariance()
        
        self.info('Agent %s initialized.' % self)

    def process_observations(self, obs):
        y = obs['observations']
        dt = obs['dt'].item()
        self.y_stats.update(y, dt)
        
    def get_state(self):
        return dict(y_stats=self.y_stats)
    
    def set_state(self, state):
        self.y_stats = state['y_stats']
    
    def publish(self, pub):
        if self.y_stats.get_num_samples() == 0:
            pub.text('warning', 'Too early to publish anything.')
            return
        Py = self.y_stats.get_covariance()
        Ry = self.y_stats.get_correlation()
        Py_inv = self.y_stats.get_information()
        Ey = self.y_stats.get_mean()
        y_max = self.y_stats.get_maximum()
        y_min = self.y_stats.get_minimum()
        
        pub.text('stats', 'Num samples: %s' % self.y_stats.get_num_samples())
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


