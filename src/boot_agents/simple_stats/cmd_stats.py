from .exp_switcher import ExpSwitcher
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
        
class CmdStats(ExpSwitcher):
    ''' 
        A simple agent that estimates various statistics 
        of the commands.
    '''

    def init(self, boot_spec):
        ExpSwitcher.init(self, boot_spec)
#         if len(boot_spec.get_observations().shape()) != 1:
#             raise UnsupportedSpec('I assume 1D signals.')

        self.episodes = defaultdict(CommandStatistics)

    def process_observations(self, obs):
        id_episode = obs['id_episode'].item()
        assert isinstance(id_episode, str)
        self.episodes[id_episode].update(obs['commands'])
    
    def merge(self, other):
        self.episodes.update(other.episodes)  

    def publish(self, pub):
        self.display(pub)
        
    def display(self, report):
        for id_episode, stats in self.episodes.items():
            with report.subsection(id_episode) as sub:
                stats.display(sub)

        
