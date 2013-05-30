from .exp_switcher import ExpSwitcher
from boot_agents.utils import ImageStats
from bootstrapping_olympics import UnsupportedSpec

__all__ = ['EstStats2D']


class EstStats2D(ExpSwitcher):
    ''' 
        A simple agent that estimates various statistics 
        of the observations. Assumes 2D signals.
    '''

    def init(self, boot_spec):
        ExpSwitcher.init(self, boot_spec)
        # TODO: check float
        if len(boot_spec.get_observations().shape()) != 2:
            raise UnsupportedSpec('I assume 2D signals.')

        self.y_stats = ImageStats()

    def merge(self, other):
        self.y_stats.merge(other.y_stats)
        
    def process_observations(self, obs):
        y = obs['observations']
        dt = obs['dt'].item()
        self.y_stats.update(y, dt)

    def get_state(self):
        return dict(y_stats=self.y_stats)

    def set_state(self, state):
        self.y_stats = state['y_stats']

    def publish(self, pub):
        if self.y_stats.get_num_samples() == 0:
            pub.text('warning', 'Too early to publish anything.')
            return
        
        self.y_stats.publish(pub)
        
