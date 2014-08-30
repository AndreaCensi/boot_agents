from blocks import Instantaneous, check_timed_named
from bootstrapping_olympics import BasicAgent, ExploringAgent, ServoingAgent
import numpy as np


__all__ = [
    'RandomSwitcher', 
    'ExpSwitcher', 
    'ExpSwitcherCanonical',
]


class RandomSwitcher(object):
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


class ExpSwitcher(BasicAgent, ExploringAgent, ServoingAgent):
    ''' 
        A simple agent that switches commands randomly according 
        to an exponential distribution. 
        
        ``beta`` is the scale parameter; E{switch} = beta
    '''

    def __init__(self, beta):
        self.beta = beta
        self.switcher = None

    def init(self, boot_spec):
        from boot_agents.utils import  RandomCommand

        interval = RandomExponential(self.beta)
        value = RandomCommand(boot_spec.get_commands())
        self.switcher = RandomSwitcher(interval, value)
    
    def choose_commands(self, timestamp):
        if self.last_timestamp is not None:
            dt = timestamp - self.last_timestamp
        else:
            dt = 0.0
        return self.switcher.get_value(dt)
    
    def get_explorer(self):
        
        class ExpSwitcherExplorer(Instantaneous):
            def __init__(self, agent):
                self.agent= agent
            
            def reset(self):
                Instantaneous.reset(self)
                self.agent.last_timestamp = None

            def transform_value(self, value):
                check_timed_named(value)
                (timestamp, (signal, _)) = value
                if not signal in ['observations']:
                    msg = 'Invalid signal %r to explorer.' % signal
                    raise ValueError(msg)
                
                cmd = self.agent.choose_commands(timestamp)
                return timestamp, ('commands', cmd)
            
        return ExpSwitcherExplorer(self) 
    

    def get_servo_system(self):
        
        class ExpSwitcherServo(Instantaneous):
            def __init__(self, agent):
                self.agent= agent
            
            def reset(self):
                Instantaneous.reset(self)

            def transform_value(self, value):
                check_timed_named(value)
                (timestamp, (signal, _)) = value
                if not signal in ['observations', 'goal_observations']:
                    msg = 'Invalid signal %r to explorer.' % signal
                    raise ValueError(msg)
                
                cmd = self.agent.choose_commands(timestamp)
                return timestamp, ('commands', cmd)
            
        return ExpSwitcherServo(self.boot_spec) 
    

class RandomExponential():
    ''' Wrapper for easy pickling. '''

    def __init__(self, beta):
        self.beta = beta

    def __call__(self):
        return np.random.exponential(self.beta, 1)


class ExpSwitcherCanonical(ExpSwitcher):
    ''' Only canonical commands are chosen. '''

    def __init__(self, beta):
        self.beta = beta

    def init(self, boot_spec):
        from boot_agents.utils import RandomCanonicalCommand
        interval = RandomExponential(self.beta)
        value = RandomCanonicalCommand(boot_spec.get_commands())
        self.switcher = RandomSwitcher(interval, value)

        self.last_timestamp = None
        self.dt = 0 