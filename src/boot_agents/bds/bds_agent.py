from . import BDSEstimator2, np
from ..simple_stats import ExpSwitcher
from ..utils import MeanCovariance, DerivativeBox, RemoveDoubles


__all__ = ['BDSAgent']

class BDSAgent(ExpSwitcher):
    '''
        Skip: only consider every $skip observations. 
    '''
    def __init__(self, beta, skip=1, change_fraction=0.0):
        ExpSwitcher.__init__(self, beta)
        self.skip = skip
        self.change_fraction = change_fraction
               
    def init(self, sensels_shape, commands_spec):
        #ExpSwitcher.init(self, sensels_shape, commands_spec)
        # FIXME: fix problem with serialization
        self.count = 0
        self.rd = RemoveDoubles(self.change_fraction)
        self.y_deriv = DerivativeBox()
        self.bds_estimator = BDSEstimator2()        
        self.y_stats = MeanCovariance()
        
        # All the rest are only statistics
        self.y_dot_stats = MeanCovariance()
        self.y_dot_abs_stats = MeanCovariance()
        self.u_stats = MeanCovariance()
        self.dt_stats = MeanCovariance()
        
    def process_observations(self, obs):
        self.count += 1 
        if self.count % self.skip != 0:
            return
        dt = obs.dt 
        y = obs.sensel_values
        u = obs.commands

        # TMP hack y = np.minimum(2.0, y)
        
        self.rd.update(y)
        if not self.rd.ready():
            return
        
        # XXX: this is not `dt` anymore FiXME:
        self.y_stats.update(y, dt)
        
        if obs.episode_changed:
            self.info('episode_changed: %s' % obs.id_episode)
            self.y_deriv.reset()
            return        

        self.y_deriv.update(y, dt)
        
        if not self.y_deriv.ready():
            return
        
        y_sync, y_dot_sync = self.y_deriv.get_value()
        y_mean = self.y_stats.get_mean() 
        y_n = y_sync - y_mean      
        self.bds_estimator.update(u=u.astype('float32'),
                                  y=y_n.astype('float32'),
                                  y_dot=y_dot_sync.astype('float32'),
                                  dt=dt)
        
        # Just other statistics
        self.dt_stats.update(np.array([dt]))
        self.u_stats.update(u, dt)
        self.y_dot_stats.update(y_dot_sync, dt)
        self.y_dot_abs_stats.update(np.abs(y_dot_sync), dt)

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
        self.bds_estimator.publish(publisher)

#        P = self.y_stats.get_covariance()
#        if self.Pinv0 is not None:
#            publisher.array_as_image('P', P, **params)
#            publisher.array_as_image('Pinv0', self.Pinv0, **params)
#            M0 = np.dot(self.Pinv0, T0)
#            M1 = np.dot(self.Pinv0, T1)                
#            publisher.array_as_image(('M', 'M0'), M0, **params)
#            publisher.array_as_image(('M', 'M1'), M1, **params)
#            Mplus = M0 + M1
#            Mminus = M0 - M1
#            publisher.array_as_image(('M', 'Mplus'), Mplus, **params)
#            publisher.array_as_image(('M', 'Mminus'), Mminus, **params)
             
#        publisher.array('rand', np.random.rand(10))
        
        self.y_stats.publish(publisher, 'y_stats')
        self.u_stats.publish(publisher, 'u_stats')
        self.y_dot_stats.publish(publisher, 'y_dot_stats')
        self.y_dot_abs_stats.publish(publisher, 'y_dot_abs_stats')
#        self.dt_stats.publish(publisher, 'dt')
        
    
        publisher.array_as_image('Tplus', Tplus, **params)
        publisher.array_as_image('Tminus', Tminus, **params)

