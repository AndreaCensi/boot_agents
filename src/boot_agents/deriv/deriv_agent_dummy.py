from .deriv_agent import DerivAgent
from blocks import Sink
from blocks.library import check_timed_named
from contracts.utils import check_isinstance

__all__ = [
    'DerivAgentDummy',
]

class DummySinkT(Sink):
    def reset(self):
        pass
    def put(self, value, block=True, timeout=None):  # @UnusedVariable
        check_timed_named(value)
        (_, (_, x)) = value
        check_isinstance(x, dict)
        x['y']
        x['y_dot']
        x['u']
        
    
class DerivAgentDummy(DerivAgent):

    def init(self, boot_spec):
        pass
 
    def get_learner_u_y_y_dot(self):
        return DummySinkT()