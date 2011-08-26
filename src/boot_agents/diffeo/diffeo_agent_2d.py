from . import DiffeomorphismEstimator, PureCommands
from ..simple_stats import RandomSwitcher
from ..utils import RandomCanonicalCommand
from bootstrapping_olympics.interfaces import AgentInterface
from contracts import contract
from numpy.core.numeric import allclose
import numpy as np

__all__ = ['DiffeoAgent2Db']

        
class DiffeoDynamics():
    
    def __init__(self, ratios):
        self.commands2dynamics = {}
        self.ratios = ratios
        self.commands2label = {}
        
    def update(self, commands_index, y0, y1, label=None):
        if not commands_index in self.commands2dynamics:
            self.commands2dynamics[commands_index] = DiffeomorphismEstimator(self.ratios)
            print('-initializing command %d (label: %s)' % (commands_index, label))
            self.commands2label[commands_index] = label
        de = self.commands2dynamics[commands_index]
        de.update(y0, y1) 
    
    def publish(self, pub):
        for ui, de in self.commands2dynamics.items():
            section = pub.section('u_%d:%s' % (ui, self.commands2label[ui]))
            de.publish(section) 
        
                
class DiffeoAgent2Db(AgentInterface):
    
    def __init__(self, rate, delta=1.0, ratios=[0.2, 0.05], pcres=40, yres=90,
                 switching_scale=1):
        self.beta = 1.0 / rate
        self.pcres = pcres 
        self.pure_commands = PureCommands(delta)
        self.delta = delta
        self.diffeo_dynamics = DiffeoDynamics(ratios)
        self.last_y = None
        self.last_data = None
        self.yres = yres
        self.switching_scale = switching_scale
        
    def init(self, sensels_shape, commands_spec):
        interval = lambda: np.random.exponential(self.beta, self.switching_scale)
        value = RandomCanonicalCommand(commands_spec).sample
        self.switcher = RandomSwitcher(interval, value)
        
    state_vars = ['diffeo_dynamics', 'pure_commands', 'last_y']


    def process_observations(self, obs):
        self.dt = obs.dt
        
        self.info('obs t=%10s cmds=%20s reset %s' % (obs.time, obs.commands, obs.episode_changed))
        
        if obs.episode_changed:
            self.pure_commands.reset()

        y = obs.sensel_values
        
        # Temporary HACK
        if True:
            y = y[5:]
            if self.last_y is not None:
                if allclose(y, self.last_y):
                    #self.info('Skip  dt=%.3f' % obs.dt)
                    self.last_y = y
                    return
                else:
                    #self.info('ok    dt=%.3f' % obs.dt)
                    pass
            
            self.last_y = y
             
        if len(y.shape) == 1: # TODO: put in filter
            y = np.maximum(0, y)
            y = np.minimum(1, y)
            
            y = popcode(y, self.pcres)
        
        if y.shape[0] > self.yres:
            ratio = y.shape[0] * 1.0 / self.yres
            ratio_round = int(np.round(ratio))
#            print('Current %s target %s ratio %.2f round %d' % 
#                  (y.shape[0], self.yres, ratio, ratio_round))
            y = y[::ratio_round, :]
            
            
        self.pure_commands.update(obs.time, obs.commands, y)
            
        last = self.pure_commands.last()
        
        if last is None:
            return
        

        self.info('pure delta=%s %s cmd # %s  (q: %s)' % (last.delta,
                                           last.commands, last.commands_index,
                                           last.queue_len))
        self.diffeo_dynamics.update(last.commands_index, last.y0, last.y1,
                                    label="%s" % last.commands)

        self.last_data = last

    def get_state(self):
        return self.get_state_vars(DiffeoAgent2Db.state_vars)
    
    def set_state(self, state):
        return self.set_state_vars(state, DiffeoAgent2Db.state_vars)
    
    def choose_commands(self):
        return self.switcher.get_value(dt=self.dt) 

                    
    def publish(self, pub):

        if self.last_data is not None:
            y0 = self.last_data.y0
            y1 = self.last_data.y1
            none = np.logical_and(y0 == 0, y1 == 0)
            x = y0 - y1
            x[none] = np.nan 
            
            pub.array_as_image('y0', y0, filter='scale')
            pub.array_as_image('y1', y1, filter='scale')
            pub.array_as_image('motion', x, filter='posneg')
            
        if self.diffeo_dynamics.commands2dynamics: # at least one
            de = self.diffeo_dynamics.commands2dynamics[0]
            field = de.get_similarity((10, 10))
            pub.array_as_image('field', field)
        
        self.diffeo_dynamics.publish(pub.section('commands'))
            
    
@contract(y='array[N](>=0,<=1)', M='int,>1,M', returns='array[NxM](float32)')
def popcode(y, M, soft=True):
    N = y.shape[0]
    pc = np.zeros((N, M), 'float32')
    for i in range(N):
        assert 0 <= y[i] <= 1
        j = int(np.round(y[i] * (M - 1)))
        assert 0 <= j < M
        pc[i, j] = 1
        if soft and j > 0:
            pc[i, j - 1] = 0.5
        if soft and j < M - 1:
            pc[i, j + 1] = 0.5
    return pc

