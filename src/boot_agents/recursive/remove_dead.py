from contracts import contract

from boot_agents.utils import MeanCovariance
from boot_agents.utils.mean_covariance import get_odd_measurements
from bootstrapping_olympics import RepresentationNuisanceCausal, UnsupportedSpec
from bootstrapping_olympics.library.agents import MultiLevelBase
from bootstrapping_olympics.library.nuisances import Select
from bootstrapping_olympics.library.nuisances_causal import SimpleRNCObs


__all__ = ['RemoveDead']


class RemoveDead(MultiLevelBase):
    ''' 
        A filter module that removes dead sensels,
        as measured by how different their statistics are.
    '''
    @contract(perc='>0,<100', ratio='>0,<1')
    def __init__(self, perc, ratio):
        self.perc = perc
        self.ratio = ratio

    def init(self, boot_spec):
        if len(boot_spec.get_observations().shape()) != 1:
            raise UnsupportedSpec('I assume 1D signals.')

        self.y_stats = MeanCovariance()

    def merge(self, other):
        self.y_stats.merge(other.y_stats)

    def process_observations(self, obs):
        y = obs['observations']
        self.y_stats.update(y)

    def get_state(self):
        return dict(y_stats=self.y_stats)

    def set_state(self, state):
        self.y_stats = state['y_stats']

    @contract(returns=RepresentationNuisanceCausal)
    def get_transform(self):
        """ Returns the nuisance at the end of learning. """
        P = self.y_stats.get_covariance()
        odd, okay = get_odd_measurements(P, self.perc, self.ratio)  # @UnusedVariable
        t = Select(okay)
        return SimpleRNCObs(t)
