from . import np
import itertools
from contracts import contract
from bootstrapping_olympics import StreamSpec

class RandomCanonicalCommand:
    ''' Returns a canonical command; if u ~ [min, max], then
        it will sample randomly between [min,0,max]. '''
    # TODO: this assumes certain semantics of the data
    def __init__(self, commands_spec): 
        self.cmds = get_canonical_commands(commands_spec)
        for cmd in self.cmds:
            commands_spec.check_valid_value(cmd)
            
    def sample(self): 
        i = np.random.randint(len(self.cmds))
        return self.cmds[i]
    
    def __call__(self):
        return self.sample()

@contract(commands_spec=StreamSpec)
def get_canonical_commands(commands_spec):
    assert len(commands_spec.shape()) == 1 # XXX: proper exception
    choices = []
    for i in range(commands_spec.size()):
        lower = commands_spec.streamels.flat[i]['lower']
        upper = commands_spec.streamels.flat[i]['upper']
        choices.append([lower, 0, upper])
    return list(np.array(x) for x in itertools.product(*choices))
        
