from bootstrapping_olympics.interfaces import AgentInterface
from bootstrapping_olympics.interfaces.commands_utils import random_commands 
import numpy as np
from boot_agents.utils.statistics import MeanCovariance, Expectation, outer
from boot_agents.simple_stats.exp_switcher import ExpSwitcher

class Queue():
    ''' keeps the last num elements ''' 
    def __init__(self, num):
        self.l = []
        self.num = num
        
    def ready(self):
        ''' True if the list contains num elements. '''
        assert len(self.l) <= self.num
        return len(self.l) == self.num
    
    def update(self, value):
        self.l.append(value)
        while len(self.l) > self.num:
            self.l.pop(0)
            
    def get_all(self):
        # returns array[num, len(value) ]
        return np.array(self.l)
    
    def reset(self):
        self.l = []
    
    def get_list(self):
        return self.l
    
class DerivativeBox():
    
    def __init__(self):
        self.q_y = Queue(3)
        self.q_dt = Queue(3)
        
    def update(self, y, dt):
        ''' returns y, y_dot or None, None if the queue is not full '''
        assert dt > 0 
        self.q_y.update(y)
        self.q_dt.update(dt)

    def ready(self):
        return self.q_y.ready()
    
    def get_value(self):
        assert self.ready()
        y = self.q_y.get_list()
        dt = self.q_dt.get_list()
        tdiff = dt[1] + dt[2]
        delta = y[-1] - y[0]
        sync_y_dot = delta / tdiff 
        sync_y = y[1]
        return sync_y, sync_y_dot

    def reset(self):
        self.q_y.reset()
        self.q_dt.reset()

class BDSAgent(ExpSwitcher):
    
    def __init__(self, beta, y_dot_tolerance=1):
        ExpSwitcher.__init__(self, beta)
        self.y_dot_tolerance = y_dot_tolerance
    
        
    state_vars = ['T', 'y_stats', 'y_dot_stats',
                  'y_dot_abs_stats', 'y_deriv', 'u_stats',
                  'count', 'qu', 'Pinv0', 'dt_stats']
        
    def init(self, sensels_shape, commands_spec):
        ExpSwitcher.init(self, sensels_shape, commands_spec)
        self.y_stats = MeanCovariance()
        self.y_dot_stats = MeanCovariance()
        self.y_dot_abs_stats = MeanCovariance()
        self.u_stats = MeanCovariance()
        self.dt_stats = MeanCovariance()
        self.T = Expectation()
        self.count = 0
        
        self.y_deriv = DerivativeBox()
        
        self.qu = Queue(100)
        self.Pinv0 = None

    def process_observations(self, obs):
        dt = obs.dt 
        y = obs.sensel_values
        # XXX tmp hack
        y = np.minimum(2.0, y)
        y[100] = np.random.rand()
        u = obs.commands
        
        self.qu.update(u)
        self.y_stats.update(y, dt)
        self.dt_stats.update(np.array([dt]))
        
        if obs.episode_changed:
            self.info('episode_changed: %s' % obs.id_episode)
            self.y_deriv.reset()
            return 
        
        if obs.dt > 0.2 or obs.dt < 0:
            self.info('Strange dt: %s, skipping' % obs.dt)
            self.y_deriv.reset()
            return             

        self.y_deriv.update(y, dt)
        
        if self.y_deriv.ready():
            y_sync, y_dot_sync = self.y_deriv.get_value()
            y_mean = self.y_stats.get_mean()
            # T =  (y - E{y}) * y_dot * u
            y_n = y_sync - y_mean        
            T = outer(u, outer(y_n, y_dot_sync))         
            self.T.update(T , dt)
        
            self.u_stats.update(u, dt)
            self.y_dot_stats.update(y_dot_sync, dt)
            self.y_dot_abs_stats.update(np.abs(y_dot_sync), dt)

            self.count += 1

            if self.count % 200 == 0:
                P = self.y_stats.get_covariance().copy()
                for i in range(P.shape[0]):
                    if P[i, i] == 0:
                        P[i, i] = 1
                self.info('updating pinv0')
                self.Pinv0 = np.linalg.pinv(P, rcond=1e-1) # TODO: make param 
                
    
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



##   
##        Eu = self.u_stats.get_mean()
##        Ey = self.y_stats.get_mean()
##        Ey_dot = self.y_dot_stats.get_mean()
##        Ey_dot_abs = self.y_dot_abs_stats.get_mean()
##        try: 
##            with publisher.plot('Ey') as pylab:
##                pylab.plot(Ey, 'x-')
##            with publisher.plot('Ey_dot') as pylab:
##                pylab.plot(Ey_dot, 'x-')
##            with publisher.plot('Ey_dot_abs') as pylab:
##                pylab.plot(Ey_dot_abs, 'x-')
##            with publisher.plot('Eu') as pylab:
##                pylab.plot(Eu, 'x-')
##            with publisher.plot('P_diagonal') as pylab:
##                pylab.plot(P.diagonal(), 'x-')
##            with publisher.plot('Qu') as pylab:
##                qu = self.qu.get_all()
##                N, K = qu.shape
##                if K == 2:
##                    pylab.plot(qu[:, 0], qu[:, 1], 'x')
##                    pylab.xlabel('u0')
##                    pylab.ylabel('u1')
##                    M = np.max(np.abs(qu)) * 1.1
##                    pylab.axis([-M, M, -M, M])
##                    
#        except Exception as e:
#            self.info('Could not plot: %s' % e)


