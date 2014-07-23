from boot_agents.utils import ImageStats
from bootstrapping_olympics import UnsupportedSpec
from bootstrapping_olympics.interfaces.agent import LearningAgent, BasicAgent
from blocks.interface import Sink
from blocks.library.timed.checks import check_timed_named

__all__ = ['EstStats2D']


class EstStats2D(BasicAgent, LearningAgent):
    ''' 
        A simple agent that estimates various statistics 
        of the observations. Assumes 2D signals.
    '''

    def init(self, boot_spec):
        # TODO: check float
        if len(boot_spec.get_observations().shape()) != 2:
            raise UnsupportedSpec('I assume 2D signals.')

        self.y_stats = ImageStats()

    def merge(self, other):
        self.y_stats.merge(other.y_stats)

    def get_learner_as_sink(self):
        class LearnSink(Sink):
            def __init__(self, y_stats):
                self.y_stats = y_stats
            def reset(self):
                pass
            def put(self, value, block=True, timeout=None):  # @UnusedVariable
                check_timed_named(value)
                timestamp, (signal, obs) = value  # @UnusedVariable
                if not signal in ['observations', 'commands']:
                    msg = 'Invalid signal %r to learner.' % signal
                    raise ValueError(msg)
                
                if signal == 'observations':
                    self.y_stats.update(obs, dt=1.0)
                
        return LearnSink(self.y_stats)
        
        
    def get_state(self):
        return dict(y_stats=self.y_stats)

    def set_state(self, state):
        self.y_stats = state['y_stats']

    def publish(self, pub):
        if self.y_stats.get_num_samples() == 0:
            pub.text('warning', 'Too early to publish anything.')
            return
        
        self.y_stats.publish(pub)
        
