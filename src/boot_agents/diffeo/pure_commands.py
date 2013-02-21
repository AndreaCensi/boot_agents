from collections import namedtuple


PureCommandsLast = namedtuple('PureCommandsLast',
                              'y0 y1 delta commands commands_index queue_len')


class PureCommands(object):
    """ 
        Converts a stream of observations/commands pairs to
        discrete (y0, y1, commands), provided that the commands
        were hold for more than delta. 
        
        Note that in case of U U U U, this will return [U U], [U U U], [U U U...]. 
        unless new_behavior=True is given.
        
    """
    def __init__(self, delta, new_behavior=False):
        """ :param delta: minimum length for commands to be the same """
        self.delta = delta
        self.cmdstr2index = {}
        self.reset()
        self.new_behavior = new_behavior
        
    def reset(self):
        self.q = []
        self.cmd = None

    def update(self, time, commands, y):
        if self.cmd is None:
            self.cmd = commands
        else:
            if cmd2key(self.cmd) != cmd2key(commands):
                self.q = []
                self.cmd = commands

        self.q.append((time, y))

        # print('Time: %s' % [x[0] for x in self.q])

    def last(self):
        ''' Returns None if not ready; otherwise it returns a tuple 
            of type Last. '''
        if len(self.q) <= 1:
            return None

        t0, y0 = self.q[0]
        t1, y1 = self.q[-1]

        length = t1 - t0
        #  print('%d elements, len= %s delta= %s' % 
        #   (len(self.q_times), length, self.delta))
        if length < self.delta:
            return None

        commands = self.cmd
        commands_index = self.cmd2index(commands)
        
        res = PureCommandsLast(y0=y0, y1=y1, delta=length, commands=commands,
                    commands_index=commands_index,
                    queue_len=(len(self.q) + 1))

        if not self.new_behavior:
            # remove first
            self.q.pop(0)
        else:
            self.q = [self.q[-1]]
            
        return res 
        
    def cmd2index(self, commands):
        key = "%s" % commands
        if not key in self.cmdstr2index:
            self.cmdstr2index[key] = len(self.cmdstr2index)
        return self.cmdstr2index[key]


def cmd2key(x):
    return "%s" % x.tolist()
