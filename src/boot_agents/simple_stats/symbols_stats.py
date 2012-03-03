from . import ExpSwitcher
from ..utils import SymbolsStatistics
from bootstrapping_olympics import UnsupportedSpec, ValueFormats

__all__ = ['SymbolsStats']


class SymbolsStats(ExpSwitcher):
    ''' 
        A simple agent that estimates various statistics 
        of the observations. 
    '''

    def init(self, boot_spec):
        ExpSwitcher.init(self, boot_spec)

        streamels = boot_spec.get_observations().get_streamels()
        if ((streamels.size != 1) or
            (streamels[0]['kind'] != ValueFormats.Discrete)):
            msg = 'I expect the observations to be symbols (one discrete var).'
            raise UnsupportedSpec(msg)

        self.lower = streamels[0]['lower']
        nsymbols = int(1 + (streamels[0]['upper'] - self.lower))
        if nsymbols > 10000:
            msg = ('Are you sure you want to deal with %d symbols?'
                   % nsymbols)
            raise Exception(msg)

        self.y_stats = SymbolsStatistics(nsymbols)

    def process_observations(self, obs):
        y = obs['observations']
        dt = obs['dt'].item()
        symbol = y[0].item() - self.lower
        self.y_stats.update(symbol, dt)

    def publish(self, pub):
        self.y_stats.publish(pub.section('y_stats'))


