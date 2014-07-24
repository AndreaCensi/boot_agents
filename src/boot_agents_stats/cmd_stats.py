from blocks import Sink, check_timed_named
from bootstrapping_olympics import BasicAgent, LearningAgent
from collections import defaultdict

__all__ = ['CmdStats']

def zero():
    return 0

class CommandStatistics(object):
    def __init__(self):
        self.commands2count = defaultdict(zero)
    
    def update(self, commands):
        commands = tuple(commands)
        self.commands2count[commands] += 1

    def display(self, report):
        report.text('summary', self.get_summary())
        
    def get_summary(self):
        return "\n".join('%6d %s' % (count, command) for command, count
                         in self.commands2count.items()) 
        
        
class CmdStats(BasicAgent, LearningAgent):
    ''' 
        A simple agent that estimates various statistics 
        of the commands.
    '''

    def init(self, boot_spec):  # @UnusedVariable
        self.episodes = defaultdict(CommandStatistics)


    def get_learner_as_sink(self):
        class LearnSink(Sink):
            def __init__(self, cmd_stats):
                self.cmd_stats = cmd_stats
            def reset(self):
                pass
            def put(self, value, block=True, timeout=None):  # @UnusedVariable
                check_timed_named(value)
                timestamp, (signal, x) = value  # @UnusedVariable
                if not signal in ['observations', 'commands']:
                    msg = 'Invalid signal %r to learner.' % signal
                    raise ValueError(msg)
                
                if signal == 'commands':
                    self.cmd_stats.update(x)
                
        return LearnSink(self)
    
    
    def update(self, x):
#     def process_observations(self, obs):
# #         id_episode = obs['id_episode'].item()
        id_episode = 'all-episodes'
        ##assert isinstance(id_episode, str)
        self.episodes[id_episode].update(x)
    
    def merge(self, other):
        self.episodes.update(other.episodes)  

    def publish(self, pub):
        self.display(pub)
        
    def display(self, report):
        for id_episode, stats in self.episodes.items():
            with report.subsection(id_episode) as sub:
                stats.display(sub)

        
