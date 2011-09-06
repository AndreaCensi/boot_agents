from . import BGDSEstimator, smooth2d
from ..simple_stats import ExpSwitcher
from ..utils import DerivativeBox

from . import np, contract

__all__ = ['BGDSAgent']

class BGDSAgent(ExpSwitcher):
    '''
        Skip: only consider every $skip observations.
        
        scales: list of floats, represents the scales at which the 
                sensels are analyzed. 0=raw data, 1= convolved with sigma=1, etc.
    '''
    @contract(scales='list[>=1](number,>=0)')
    def __init__(self, beta, skip=1, scales=[0]):
        ExpSwitcher.__init__(self, beta)
        self.skip = skip
        self.scales = scales
               
    def init(self, sensels_shape, commands_spec):
        #ExpSwitcher.init(self, sensels_shape, commands_spec)
        self.count = 0
        self.y_deriv = DerivativeBox()
        self.bgds_estimator = BGDSEstimator()  
        
    def process_observations(self, obs):
        self.count += 1 
        if self.count % self.skip != 0:
            return
        
        dt = obs.dt 
        u = obs.commands
        y0 = obs.sensel_values
        
        y = create_scales(y0, self.scales)
            
        if obs.episode_changed:
            self.y_deriv.reset()
            return        

        self.y_deriv.update(y, dt)
        
        if not self.y_deriv.ready():
            return
        
        y_sync, y_dot_sync = self.y_deriv.get_value()

        self.bgds_estimator.update(u=u.astype('float32'),
                                   y=y_sync.astype('float32'),
                                   y_dot=y_dot_sync.astype('float32'),
                                   dt=dt)
        self.last_y0 = y0
        self.last_y = y
        
#        
#        if self.count > 2000:
#            if self.count % 1000 == 0 or self.model is None:
#                self.model = self.bgds_estimator.get_model()
#            
        
    def publish(self, publisher):
        if self.count < 10: 
            self.info('Skipping publishing as count=%d' % self.count)
            return
        
        self.bgds_estimator.publish(publisher)

        sec = publisher.section('preprocessing')
        sec.array_as_image('last_y0', self.last_y0, filter='scale')
        sec.array_as_image('last_y', self.last_y, filter='scale')
        example = np.zeros(self.last_y.shape)
        example.flat[150] = 1 
        example_smooth = create_scales(example, self.scales) 
        sec.array_as_image('example_smooth', example_smooth)

def create_scales(y='array[HxW]', scales='list[M](float,>=0)',
                  returns='array[Hx(W*M)]'):
    data = []
    for s in scales:
        if s == 0:
            data.append(y)
        else:
            data.append(smooth2d(y, s))
    return np.hstack(data)
