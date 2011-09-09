from . import np
import itertools



class RandomCanonicalCommand:
    ''' Returns a canonical command; if u ~ [min, max], then
        it will sample randomly between [min,0,max]. '''
    
    def __init__(self, commands_spec):
        choices = [[a, 0, b] for a, b in commands_spec]
        self.cmds = list(itertools.product(*choices))

    def sample(self): # TODO: remove this old interface
        i = np.random.randint(len(self.cmds))
        return self.cmds[i]
    
    def __call__(self):
        return self.sample()
