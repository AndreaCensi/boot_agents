from bootstrapping_olympics import AgentInterface
from ..utils.commands import get_canonical_commands


class CanonicalCommandsAgent(AgentInterface):
    """ 
        This is an agent that chooses one "canonical" command
        for episode and sticks to it. 
    """

    def init(self, boot_spec):
        self.commands = get_canonical_commands(boot_spec.get_commands())
        self.nepisode = 0

        self.info('Found %d canonical commands.' % len(self.commands))

    def process_observations(self, observations):
        # We don't need observations, but we check whether the 
        # episode changed
        if observations['episode_start']:
            self.nepisode += 1

    def choose_commands(self):
        i = self.nepisode % len(self.commands)
        return self.commands[i]

