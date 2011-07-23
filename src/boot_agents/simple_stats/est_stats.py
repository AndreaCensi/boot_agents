from . import ExpSwitcher
from ..utils import MeanCovariance

class EstStats(ExpSwitcher):
    ''' A simple agent that estimates the covariance of the observations. '''
    
    def init(self, sensels_shape, commands_spec):
        ExpSwitcher.init(self, sensels_shape, commands_spec)
        if len(sensels_shape) != 1:
            self.unsupported('I assume 1D signals.')
            
        #self.num_sensels = sensels_shape[0]
        self.y_stats = MeanCovariance()

    def process_observations(self, observations):
        self.y_stats.update(observations)
        
    def get_state(self):
        return dict(y_stats=self.y_stats)
    
    def set_state(self, state):
#        if not isinstance(state, dict) or not 'y_stats' in state:
#            raise ValueError('Invalid state---perhaps I changed the format?')
        self.y_stats = state['y_stats']
    
    def publish(self, pub):
        Py = self.y_stats.get_covariance()
        Ry = self.y_stats.get_correlation()
        Py_inv = self.y_stats.get_information()
        Ey = self.y_stats.get_mean()
        y_max = self.y_stats.get_maximum()
        y_min = self.y_stats.get_minimum()
        
        pub.publish_array_as_image('Py', Py)
        pub.publish_array_as_image('Ry', Ry)
        pub.publish_array_as_image('Py_inv', Py_inv)
        
        with pub.plot(name='y_stats') as pylab:
            pylab.plot(Ey, label='E(y)')
            pylab.plot(y_max, label='y_max')
            pylab.plot(y_min, label='y_min')
            pylab.legend()



