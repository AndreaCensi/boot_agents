from bootstrapping_olympics import BasicAgent, ExploringAgent


__all__ = ['CanonicalCommandsAgent']

class CanonicalCommandsAgent(BasicAgent, ExploringAgent):
    """ 
        This is an agent that chooses one "canonical" command
        for episode and sticks to it. 
    """

    def init(self, boot_spec):
        from boot_agents.utils import get_canonical_commands

        self.commands = get_canonical_commands(boot_spec.get_commands())

        self.info('Found %d canonical commands.' % len(self.commands))
        for i, cmd in enumerate(self.commands):
            self.info('cmd %d  %s' % (i, list(cmd)))

        self.nepisode = -1

    def process_observations(self, observations):
        # We don't need observations, but we check whether the 
        # episode changed
        if observations['episode_start']:
            self.nepisode += 1

    def choose_commands(self):
        i = self.nepisode % len(self.commands)
        return self.commands[i]

