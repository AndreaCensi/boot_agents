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
                S[i, :] = score

            self.D[ui].update(S)


        self.q_u.pop(0)
        self.q_times.pop(0)
        self.q_y.pop(0)

#        
#        
#        # XXX: not unbiased
#        y_dot = (y - self.last_y) / dt
#        
#        # T =  (y - E{y}) * y_dot * u
#        y_n = y - self.y_stats.get_mean()       
#        T = outer(self.u, outer(y_n, y_dot))         
#
#        self.T.update(T)
#        self.y_stats.update(y)
#        self.y_dot_stats.update(y_dot)
#        
#        self.last_y = y
        
#        if int(self.time) % 100 == 0 and int(self.time) != self.time_last_print:     
#            self.print_averages()
#            self.time_last_print = int(self.time)
#        

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
        for i in range(len(self.cmds)):
            self.Ds.append(discretize(self.D[i].get_value()))
            
        for i in range(len(self.cmds)):
            name = 'D%dn' % i
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


def discretize(M):
    X = np.zeros(M.shape)
    for i in range(M.shape[0]):
        which = np.argmin(M[i, :])
        X[i, which] = 1 
    return X
#    
#        publisher.array_as_image(name='P', value=self.P,
#                                         filter=publisher.FILTER_POSNEG,
#                                         filter_params={})
#                
#        max_value = np.abs(self.T).max()
#        
#        for i in range(self.num_commands):
#            Ti = self.T[i, :, :]
#            publisher.array_as_image(name='T%d' % i, value=Ti,
#                                         filter=publisher.FILTER_POSNEG,
#                                         filter_params={'max_value':max_value})
#                



