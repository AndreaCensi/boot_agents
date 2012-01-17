from . import DiffeoDynamics, PureCommands
from ..simple_stats import ExpSwitcherCanonical
from bootstrapping_olympics import AgentInterface
from contracts import contract
import numpy as np
from bootstrapping_olympics import UnsupportedSpec

__all__ = ['DiffeoAgent2Db']


class DiffeoAgent2Db(AgentInterface):
    ''''
    
    match_method:
    
        binary
        
    resize_method:
        PIL
        PIL_both
        raw
    '''
    def __init__(self, rate, delta=1.0, ratios=[0.2, 0.05],
                 target_resolution=None,
                 switching_scale=1, match_method='binary',
                 resize_method='PIL'):
        self.beta = 1.0 / rate
        self.pure_commands = PureCommands(delta)
        self.delta = delta
        self.diffeo_dynamics = DiffeoDynamics(ratios, match_method)
        self.last_y = None
        self.last_data = None
        self.target_resolution = target_resolution
        self.switching_scale = switching_scale
        self.resize_method = resize_method

    def init(self, boot_spec):
        self.switcher = ExpSwitcherCanonical(self.beta)
        # TODO: do the 1D version
        if len(boot_spec.get_observations().shape()) != 2:
            raise UnsupportedSpec('Can only work with 2D signals.')

        self.switcher.init(boot_spec)

    def process_observations(self, obs):

        self.dt = float(obs['dt'])

        if obs['episode_start']:
            self.pure_commands.reset()

        y = obs['observations']

        # Temporary HACK
#        if False:
#            y = y[5:]
#            if self.last_y is not None:
#                if allclose(y, self.last_y):
#                    #self.info('Skip  dt=%.3f' % obs.dt)
#                    self.last_y = y
#                    return
#                else:
#                    #self.info('ok    dt=%.3f' % obs.dt)
#                    pass
#            
#            self.last_y = y

        if len(y.shape) == 1:
            y = np.maximum(0, y)
            y = np.minimum(1, y)
            y = popcode(y, self.target_resolution[1])
#        self.info('Now resolution is %s' % str(y.shape))

        if self.target_resolution is not None:
            target_h = self.target_resolution[0]
            if y.shape[0] > target_h:
                if self.resize_method == 'PIL':
                    from scipy.misc import imresize #@UnresolvedImport
                    fraction = float(target_h) / y.shape[0]
                    y = imresize(y, fraction)
                    y = np.array(y, dtype='float32')
                elif self.resize_method == 'PIL_both':
                    from scipy.misc import imresize #@UnresolvedImport@Reimport
                    y = imresize(y, self.target_resolution)
                    y = np.array(y, dtype='float32')
                elif self.resize_method == 'raw':
                    ratio = y.shape[0] * 1.0 / target_h
                    ratio_round = int(np.round(ratio))
                    y = y[::ratio_round, :]
                else:
                    msg = 'Unknown resize method %r' % self.resize_method
                    raise Exception(msg)

        self.pure_commands.update(obs['timestamp'], obs['commands'], y)

        last = self.pure_commands.last()

        if last is None:
            return

        self.info('pure delta=%s %s cmd # %s  (q: %s)' % (last.delta,
                                           last.commands, last.commands_index,
                                           last.queue_len))
        self.diffeo_dynamics.update(last.commands_index, last.y0, last.y1,
                                    label="%s" % last.commands,
                                    u=last.commands)

        self.last_data = last

    def choose_commands(self):
        return self.switcher.choose_commands()

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


# TODO: move somewhere else
@contract(y='array[N](>=0,<=1)', M='int,>1,M', returns='array[NxM](float32)')
def popcode(y, M, soft=True):
    N = y.shape[0]
    pc = np.zeros((N, M), 'float32')
    for i in range(N):
        assert 0 <= y[i] <= 1, 'Strange value of y[%d] = %s' % (i, y[i])
        j = int(np.round(y[i] * (M - 1)))
        assert 0 <= j < M
        pc[i, j] = 1
        if soft and j > 0:
            pc[i, j - 1] = 0.5
        if soft and j < M - 1:
            pc[i, j + 1] = 0.5
    return pc

