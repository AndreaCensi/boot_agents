from bootstrapping_olympics import AgentInterface, random_commands

from . import  np
from ..utils import RandomCanonicalCommand

class RandomSwitcher:
    def __init__(self, interval_function, value_function):
        self.interval_function = interval_function
        self.value_function = value_function
        self.time = 0
        self.next_switch = self.time + self.interval_function()
        self.output = self.value_function()
        
    def get_value(self, dt):
        self.time += dt
        if self.time > self.next_switch:
            self.next_switch = self.time + self.interval_function()
            self.output = self.value_function()
        return self.output
    
class ExpSwitcher(AgentInterface):
    ''' A simple agent that switches commands randomly according 
        to an exponential distribution. 
        
        ``beta`` is the scale parameter; E{switch} = beta
    '''

    def __init__(self, beta):
        self.beta = beta 
        
    def init(self, sensels_shape, commands_spec):
        interval = lambda: np.random.exponential(self.beta, 1)
        value = lambda: random_commands(commands_spec)
        self.switcher = RandomSwitcher(interval, value)
        self.dt = 0
        
    def process_observations(self, observations):
        self.dt = observations.dt

    def choose_commands(self):
        return self.switcher.get_value(self.dt)



class ExpSwitcherCanonical(AgentInterface):
    ''' Only canonical commands are chosen. '''

    def __init__(self, beta):
        self.beta = beta 
        
    def init(self, sensels_shape, commands_spec):
        interval = lambda: np.random.exponential(self.beta, 1)
        value = RandomCanonicalCommand(commands_spec).sample
        self.switcher = RandomSwitcher(interval, value)
        self.dt = 0
        
    def process_observations(self, observations):
        self.dt = observations.dt

    def choose_commands(self):
        return self.switcher.get_value(self.dt)


