import numpy as np
from boot_agents.utils import MeanCovariance, DerivativeBox, Queue
from boot_agents.simple_stats import ExpSwitcher
from contracts import contract
from . import BDSEstimator2


class BDSAgent(ExpSwitcher):
    
    def __init__(self, beta, y_dot_tolerance=1, skip=1):
        ExpSwitcher.__init__(self, beta)
        self.y_dot_tolerance = y_dot_tolerance
        self.skip = skip

        
    state_vars = ['y_stats', 'y_dot_stats',
                  'y_dot_abs_stats', 'y_deriv', 'u_stats',
                  'count', 'qu', 'Pinv0', 'dt_stats', 'bds_estimator']
        
    def init(self, sensels_shape, commands_spec):
        ExpSwitcher.init(self, sensels_shape, commands_spec)
        self.y_stats = MeanCovariance()
        self.y_dot_stats = MeanCovariance()
        self.y_dot_abs_stats = MeanCovariance()
        self.u_stats = MeanCovariance()
        self.dt_stats = MeanCovariance()
        self.count = 0
        
        self.y_deriv = DerivativeBox()
        
        self.qu = Queue(100)
        self.Pinv0 = None
        
        self.bds_estimator = BDSEstimator2()
        
        
    def process_observations(self, obs):
        self.count += 1 
        if self.count % self.skip != 0:
            return
        dt = obs.dt 
        y = obs.sensel_values
        # XXX tmp hack
        y = np.minimum(2.0, y)
        y[100] = np.random.rand()
        w = obs.commands
        # A = np.array([[1, 1], [1, -1]])
        A = np.eye(w.size)
        u = np.dot(A, w)
        
#        s = vec2subspace(u)
#        accept = s[0] == s[1]
        accept = True 
        # accept = (u[0] != 0) and (u[1] == 0)
#        self.info('%s  w %s u %s accept? %s' % (obs.counter, w, u, accept))
        
        
        self.qu.update(u)
        self.y_stats.update(y, dt)
        self.dt_stats.update(np.array([dt]))
        
        if obs.episode_changed:
            self.info('episode_changed: %s' % obs.id_episode)
            self.y_deriv.reset()
            return        

        self.y_deriv.update(y, dt)
        
        if self.y_deriv.ready():
            
            if not accept: return
            
            y_sync, y_dot_sync = self.y_deriv.get_value()
                
            y_mean = self.y_stats.get_mean()
            # T =  (y - E{y}) * y_dot * u
            y_n = y_sync - y_mean      
            self.bds_estimator.update(u=u.astype('float32'),
                                      y=y_n.astype('float32'),
                                      y_dot=y_dot_sync.astype('float32'),
                                      dt=dt)
            
            self.u_stats.update(u, dt)
            self.y_dot_stats.update(y_dot_sync, dt)
            self.y_dot_abs_stats.update(np.abs(y_dot_sync), dt)
    
    def get_state(self):
        return self.get_state_vars(BDSAgent.state_vars)
    
    def set_state(self, state):
        return self.set_state_vars(state, BDSAgent.state_vars)
    
    
    def publish(self, publisher):
        if self.count < 10: 
            self.info('Skipping publishing as count=%d' % self.count)
            return

        params = dict(filter=publisher.FILTER_POSNEG, filter_params={'skim':2})


        T = self.bds_estimator.get_T()
        T0 = T[0, :, :]
        T1 = T[1, :, :]
        Tplus = T0 + T1
        Tminus = T0 - T1
#        
#        
#        K, n, n = T.shape
#        for i in range(K):
#            Ti = T[i, :, :]
#            publisher.array_as_image('T%d' % i, Ti, **params)
        self.bds_estimator.publish(publisher)

        P = self.y_stats.get_covariance()
        if self.Pinv0 is not None:
            publisher.array_as_image('P', P, **params)
            publisher.array_as_image('Pinv0', self.Pinv0, **params)
            M0 = np.dot(self.Pinv0, T0)
            M1 = np.dot(self.Pinv0, T1)                
            publisher.array_as_image(('M', 'M0'), M0, **params)
            publisher.array_as_image(('M', 'M1'), M1, **params)
            Mplus = M0 + M1
            Mminus = M0 - M1
            publisher.array_as_image(('M', 'Mplus'), Mplus, **params)
            publisher.array_as_image(('M', 'Mminus'), Mminus, **params)
             
             
        publisher.array('rand', np.random.rand(10))
        
        self.y_stats.publish(publisher, 'y')
        self.u_stats.publish(publisher, 'u')
        self.y_dot_stats.publish(publisher, 'y_dot')
        self.y_dot_abs_stats.publish(publisher, 'y_dot_abs')
#        self.dt_stats.publish(publisher, 'dt')
        
    
        publisher.array_as_image('Tplus', Tplus, **params)
        publisher.array_as_image('Tminus', Tminus, **params)


@contract(v='array')
def vec2subspace(v):
    ''' Returns x such that x = alpha v and x0 >= 0 '''
    
    n = np.linalg.norm(v)
    if n == 0:
        return v
    else:
        return v / n * np.sign(v[0])
    
