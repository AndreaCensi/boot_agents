from . import np
from ..utils import RandomCanonicalCommand
from bootstrapping_olympics import AgentInterface

__all__ = ['RandomSwitcher', 'ExpSwitcher', 'ExpSwitcherCanonical']


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
        
    def init(self, boot_spec):
        
        def interval():
            return np.random.exponential(self.beta, 1)
        def value():
            return boot_spec.get_commands().get_random_value()
        self.switcher = RandomSwitcher(interval, value)
        self.dt = 0
        
    def process_observations(self, observations):
        self.dt = float(observations['dt'])

    def choose_commands(self):
        return self.switcher.get_value(self.dt)


class RandomExponential():
    ''' Wrapper for easy pickling. '''
    def __init__(self, beta):
        self.beta = beta
    def __call__(self):
        return np.random.exponential(self.beta, 1)
    
class ExpSwitcherCanonical(AgentInterface):
    ''' Only canonical commands are chosen. '''

    def __init__(self, beta):
        self.beta = beta 
        
    def init(self, boot_spec):
        interval = RandomExponential(self.beta)
        value = RandomCanonicalCommand(boot_spec.get_commands()) 
        self.switcher = RandomSwitcher(interval, value)
        self.dt = 0
        
    def process_observations(self, observations):
        self.dt = float(observations['dt'])

    def choose_commands(self):
        return self.switcher.get_value(self.dt)


