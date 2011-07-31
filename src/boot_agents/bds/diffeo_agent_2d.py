import numpy as np
from boot_agents.utils.statistics import MeanCovariance, Expectation, outer
from boot_agents.simple_stats.exp_switcher import ExpSwitcher
from boot_agents.bds.bds_agent import DerivativeBox

class DiffeoAgent2D(ExpSwitcher):
    
    def __init__(self, beta, y_dot_tolerance=1):
        ExpSwitcher.__init__(self, beta)
        self.y_dot_tolerance = y_dot_tolerance
    
        
    state_vars = ['T', 'y_stats', 'y_dot_stats',
                  'y_dot_abs_stats', 'y_deriv', 'u_stats',
                  'count', 'qu', 'Pinv0', 'dt_stats']
        
    def init(self, sensels_shape, commands_spec):
        ExpSwitcher.init(self, sensels_shape, commands_spec)
        self.z_stats = MeanCovariance()
        self.z_dot_stats = MeanCovariance()
        self.z_dot_abs_stats = MeanCovariance()
        self.u_stats = MeanCovariance()
        self.dt_stats = MeanCovariance()
        self.T = Expectation()
        self.count = 0
        
        self.y_deriv = DerivativeBox()
        self.Pinv0 = None

    def process_observations(self, obs):
        dt = obs.dt 
        y = obs.sensel_values
        # XXX tmp hack
        y = np.minimum(1.0, y)
        y = np.maximum(0.0, y)
        
        z = popcode(y, resolution=64)
        
        self.z_stats.update(z, dt)
        
        if obs.episode_changed:
            self.info('episode_changed: %s' % obs.id_episode)
            self.y_deriv.reset()
            return 
        
    
    def get_state(self):
        return self.get_state_vars(BDSAgent.state_vars)
    
    def set_state(self, state):
        return self.set_state_vars(state, BDSAgent.state_vars)
    
    
    def publish(self, publisher):
        if self.count < 10: return
        T = self.T.get_value()

        params = dict(filter=publisher.FILTER_POSNEG, filter_params={'skim':2})


        T0 = T[0, :, :]
        T1 = T[1, :, :]
        Tplus = T0 + T1
        Tminus = T0 - T1
        
        K, n, n = T.shape
        for i in range(K):
            Ti = T[i, :, :]
            publisher.array_as_image('T%d' % i, Ti, **params)

        if self.Pinv0 is not None:
            publisher.array_as_image('Pinv0', self.Pinv0, **params)
            M0 = np.dot(self.Pinv0, T0)
            M1 = np.dot(self.Pinv0, T1)                
            publisher.array_as_image(('M', 'M0'), M0, **params)
            publisher.array_as_image(('M', 'M1'), M1, **params)
            Mplus = M0 + M1
            Mminus = M0 - M1
            publisher.array_as_image(('M', 'Mplus'), Mplus, **params)
            publisher.array_as_image(('M', 'Mminus'), Mminus, **params)
             
        self.y_stats.publish(publisher, 'y')
        self.u_stats.publish(publisher, 'u')
        self.y_dot_stats.publish(publisher, 'y_dot')
        self.y_dot_abs_stats.publish(publisher, 'y_dot_abs')
        self.y_dot_abs_stats.publish(publisher, 'dt')
        
        
        
        
        publisher.array_as_image('Tplus', Tplus, **params)
        publisher.array_as_image('Tminus', Tminus, **params)

