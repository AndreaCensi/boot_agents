from bootstrapping_olympics.interfaces.commands_utils import random_commands 
import numpy as np
from boot_agents.utils import MeanCovariance, Expectation, outer
from boot_agents.simple_stats import ExpSwitcher
from bootstrapping_olympics.interfaces.agent_interface import AgentInterface
from boot_agents.simple_stats.exp_switcher import RandomSwitcher
import itertools


class DiffeoAgent(AgentInterface):
    
    def __init__(self, rate, delta=1.0):
        self.beta = delta * 2
        self.delta = delta
        
        self.q_times = []
        self.q_y = []
        self.q_u = []
        
    def init(self, sensels_shape, commands_spec):
        choices = [[a, 0, b] for a, b in commands_spec]
        self.cmds = list(itertools.product(*choices))
        self.info('cmds: %s' % self.cmds)
        
        interval = lambda: np.random.exponential(self.beta, 1)
        value = lambda: self.cmds[np.random.randint(len(self.cmds))]
        self.switcher = RandomSwitcher(interval, value)
        self.D = [Expectation() for i in range(len(self.cmds))] #@UnusedVariable
        
        n = sensels_shape[0]
        for i in range(len(self.cmds)):
            self.D[i].update(np.zeros((n, n)))
            
    def cmd2index(self, u):
        for i, cmd in enumerate(self.cmds):
            if np.all(cmd == u):
                return i
        raise ValueError('No %s in %s' % (u, self.cmds))
     
    def process_observations(self, obs):
        self.dt = obs.dt
        y = obs.sensel_values
        
        if obs.episode_changed:
            self.q_y = []
            self.q_u = []
            self.q_times = []
            return 

        self.q_times.append(obs.time)
        self.q_u.append(obs.commands)
        self.q_y.append(y)
        
        length = self.q_times[-1] - self.q_times[0]
#        print('%d elements, len= %s delta= %s' % 
#              (len(self.q_times), length, self.delta))
        if length < self.delta:
            return
        cmds = set(["%s" % u.tolist() for u in self.q_u])
        
        if len(cmds) > 1:
            #self.info('not pure')
            pass
        else:
            ui = self.cmd2index(self.q_u[-1])
            #self.info('pure %s %s' % (ui, self.q_u[-1]))
            
            y0 = self.q_y[0]
            yT = self.q_y[-1]
            
            n = y0.size
            S = np.zeros((n, n))
            for i in range(n):
                diff = y0[i] - yT
                score = np.abs(diff)
                score = scale_score(score)
                S[i, :] = score

            self.D[ui].update(S)


        self.q_u.pop(0)
        self.q_times.pop(0)
        self.q_y.pop(0)
 
    state_vars = ['D', 'q_u', 'q_y', 'q_times', 'cmds']
    
    def get_state(self):
        return self.get_state_vars(DiffeoAgent.state_vars)
    
    def set_state(self, state):
        return self.set_state_vars(state, DiffeoAgent.state_vars)
    
    def choose_commands(self):
        return self.switcher.get_value(dt=self.dt) 

                    
    def publish(self, publisher):
        max_value = max(D.get_value().max() for D in self.D)
        for i in range(len(self.cmds)):
            name = 'D%d' % i
            value = self.D[i].get_value()
            
            publisher.array_as_image(name, value,
                                    filter=publisher.FILTER_SCALE,
                                    filter_params={'max_value':max_value})

        s = ""
        for i in range(len(self.cmds)):
            s += '#%d = %s\n' % (i, self.cmds[i])
        publisher.text('cmds', s)

        self.Ds = [] 
        self.Dn = [] 
        alpha = 0.3
#        alpha = 0.01 # cam
        for i in range(len(self.cmds)):
            self.Ds.append(discretize(self.D[i].get_value()))
            self.Dn.append(normalize(self.D[i].get_value(), alpha))
            
        for i in range(len(self.cmds)):
            name = 'D%dn_normalize' % i
            value = self.Dn[i]
            
            publisher.array_as_image(name, value,
                                    filter=publisher.FILTER_SCALE,
                                    filter_params={})
        for i in range(len(self.cmds)):
            name = 'D%d_discretize' % i
            value = self.Ds[i]
            
            publisher.array_as_image(name, value,
                                    filter=publisher.FILTER_SCALE,
                                    filter_params={})

        for i in range(len(self.cmds)):
            name = 'D%dn_sq' % i
            value = np.dot(self.Ds[i], self.Ds[i])
            
            publisher.array_as_image(name, value,
                                    filter=publisher.FILTER_SCALE,
                                    filter_params={})

        a = 1; b = 3
        a = 0; b = 1 # cam
        Da = self.Dn[a]
        Db = self.Dn[b]
        Dma = Da.T
        Dmb = Db.T
        D_a_b = np.dot(Db, Da)
        D_a_b_ma = np.dot(Dma, D_a_b) 
        D_a_b_ma_mb = np.dot(Dmb, D_a_b_ma)
        publisher.array_as_image("f:a", Da, publisher.FILTER_SCALE)
        publisher.array_as_image("f:b", Db, publisher.FILTER_SCALE)
        publisher.array_as_image("f:a,b", D_a_b, publisher.FILTER_SCALE)
        publisher.array_as_image("f:a,b,ma", D_a_b_ma, publisher.FILTER_SCALE)
        publisher.array_as_image("f:a,b,ma,mb", D_a_b_ma_mb, publisher.FILTER_SCALE)
        

def discretize(M):
    X = np.zeros(M.shape)
    for i in range(M.shape[0]):
        which = np.argmin(M[i, :])
        X[i, which] = 1 
    return X 

def normalize(M, alpha):
    X = np.zeros(M.shape)
    for i in range(M.shape[0]):
        dist = scale_score(M[i, :])
        p = np.exp(-dist * alpha)
        p /= p.sum()
        X[i, :] = p
    return X 


def scale_score(x):
    y = x.copy()
    order = np.argsort(x.flat)
    # Black magic ;-) Probably the smartest thing I came up with today. 
    order_order = np.argsort(order)
    y.flat[:] = order_order.astype(y.dtype)
    return y


