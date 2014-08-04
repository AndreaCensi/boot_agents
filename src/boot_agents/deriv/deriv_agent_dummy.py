from .deriv_agent import DerivAgent
from blocks import Sink

__all__ = [
    'DerivAgentDummy',
]

class DummySinkT(Sink):
    def reset(self):
        pass
    def put(self, value, block=True, timeout=None):
        pass
    
class DerivAgentDummy(DerivAgent):

    def get_learner_u_y_y_dot(self):
        return DummySinkT()