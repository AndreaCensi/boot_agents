from collections import namedtuple


def cmd2key(x):
    return "%s" % x.tolist()
    
PureCommandsLast = namedtuple('PureCommandsLast',
                              'y0 y1 delta commands commands_index queue_len') 
    
class PureCommands(object):
    def __init__(self, delta):
        self.delta = delta
        self.cmdstr2index = {}
        self.reset()
        self.cmd = None
        
    def reset(self):
        self.q = []
        
    def update(self, time, commands, y):
        if self.cmd is None:
            self.cmd = commands
        else:
            if cmd2key(self.cmd) != cmd2key(commands):
                self.reset()
                self.cmd = commands
                
        self.q.append((time, y))
        
        #print('Time: %s' % [x[0] for x in self.q])
        
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
        
        # remove first
        self.q.pop(0) 

        return PureCommandsLast(y0=y0, y1=y1, delta=length, commands=commands,
                    commands_index=commands_index,
                    queue_len=(len(self.q) + 1))
        
    def cmd2index(self, commands):
        key = "%s" % commands
        if not key in self.cmdstr2index:
            self.cmdstr2index[key] = len(self.cmdstr2index)
        return self.cmdstr2index[key] 
