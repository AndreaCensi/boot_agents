from bootstrapping_olympics.interfaces import AgentInterface
from bootstrapping_olympics.interfaces.commands_utils import random_commands 
import numpy as np
from boot_agents.utils.statistics import MeanCovariance, Expectation, outer
from boot_agents.simple_stats.exp_switcher import ExpSwitcher

class BDSAgent(ExpSwitcher):
    
    def __init__(self, beta, y_dot_tolerance=1):
        self.beta = beta
        self.y_dot_tolerance = y_dot_tolerance
        
    def init(self, sensels_shape, commands_spec):
        ExpSwitcher.init(self, sensels_shape, commands_spec)
        self.y_stats = MeanCovariance()
        self.u_stats = MeanCovariance()
        self.T = Expectation()

    def process_observations(self, obs):
        dt = obs.dt
        y = obs.sensel_values
        
        if obs.episode_changed:
            self.last_y = y
            return 
        
        # XXX: not unbiased
        y_dot = (y - self.last_y) / dt
        
        # T =  (y - E{y}) * y_dot * u
        y_n = y - self.y_stats.get_mean()       
        T = outer(self.u, outer(y_n, y_dot))         

        self.T.update(T)
        self.y_stats.update(y)
        self.y_dot_stats.update(y_dot)
        
        self.last_y = y
        
#        if int(self.time) % 100 == 0 and int(self.time) != self.time_last_print:     
#            self.print_averages()
#            self.time_last_print = int(self.time)
#        

    state_vars = ['T', 'last_y', 'y_stats', 'y_dot_stats']
    
    def get_state(self):
        return self.get_state_vars(BDSAgent.state_vars)
    
    def set_state(self, state):
        return self.set_state_vars(state, BDSAgent.state_vars)
    
    

    def choose_commands(self):
        self.update_commands()
        return self.u

    def update_commands(self):
        time_passed = self.time - self.time_of_switch
        if time_passed < self.T_switch:
            return
        
        self.Eu = weighted_average(self.Eu, self.time, self.u, time_passed)        
        self.T_switch = np.random.exponential(self.beta, 1)
        
        # This function does it properly for multiple commands,
        # also not requiring the explicit bound.
        self.u = random_commands(commands_spec=self.commands_spec)

        self.time_of_switch = self.time
                    
    def publish(self, publisher):
        publisher.array_as_image(name='P', value=self.P,
                                         filter=publisher.FILTER_POSNEG,
                                         filter_params={})
                
        max_value = np.abs(self.T).max()
        
        for i in range(self.num_commands):
            Ti = self.T[i, :, :]
            publisher.array_as_image(name='T%d' % i, value=Ti,
                                         filter=publisher.FILTER_POSNEG,
                                         filter_params={'max_value':max_value})
                



