from . import np
from .. import PureCommands

def pure_commands_test():
    
    pc = PureCommands(delta=0.15)
    y = 0
    dt = 0.1
    u1 = np.array([0, 1])
    u2 = np.array([1, 0])
    pc.update(0, u1, y)
    assert pc.last() == None
    pc.update(0, u1, y)
    
    # TODO: finish
