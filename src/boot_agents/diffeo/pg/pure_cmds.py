from boot_agents.diffeo.pure_commands import PureCommands
from procgraph import  Block


class PureCmds(Block):
    ''' Same filtering as PureCommands '''

    Block.alias('pure_commands')

    Block.config('delta', 'Minimum delta')
    Block.config('delta_tol', 'Minimum delta tolerance ratio', default=1.2)

    Block.input('observations')
    Block.output('tuples', '(y0,y1,cmd)')

    def init(self):
        self.pc = PureCommands(self.config.delta, new_behavior=False)

    def update(self):
        obs = self.input.observations

#        self.dt = float(obs['dt'])

        if obs['episode_start']:
            self.info('Episode start: %s' % obs['id_episode'].item())
            self.pc.reset()

        self.pc.update(obs['timestamp'], obs['commands'],
                                        obs['observations'])

        last = self.pc.last()

        if last is None:
            return

        delta = last.delta
        if delta > self.config.delta * self.config.delta_tol:
#            self.info('skip delta %s' % delta)
            return
        
#        self.info('pure delta=%s %s cmd # %s  (qlen: %s)' % (last.delta,
#                                           last.commands, last.commands_index,
#                                           last.queue_len))

#        last.commands_index, 
#        last.y0, last.y1,
#        label="%s" % last.commands,
#        last.commands

        self.output.tuples = last
