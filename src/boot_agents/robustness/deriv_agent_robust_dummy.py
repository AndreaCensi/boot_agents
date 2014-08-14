from blocks import Sink
from blocks.library.timed.checks import check_timed_named
from contracts.utils import check_isinstance
from boot_agents.robustness.deriv_agent_robust import DerivAgentRobust

__all__ = [
    'DerivAgentRobustDummy',
]

class DummySinkT(Sink):
    def reset(self):
        pass
    def put(self, value, block=True, timeout=None):  # @UnusedVariable
        check_timed_named(value)
        (_, (_, x)) = value
        check_isinstance(x, dict)
        x['y']
        x['u']
        x['y_dot']
        x['w']
        
    
class DerivAgentRobustDummy(DerivAgentRobust):
 
    def get_learner_u_y_y_dot_w(self):
        return DummySinkT()