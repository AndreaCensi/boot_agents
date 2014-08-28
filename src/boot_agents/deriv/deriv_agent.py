from .sync_box import get_sync_deriv_box
from abc import abstractmethod
from blocks import Info, Sink, series
from bootstrapping_olympics import BasicAgent, LearningAgent
from contracts import contract

__all__ = [
    'DerivAgent',
]

class DerivAgent(BasicAgent, LearningAgent):
    """ 
        Base class for agents that learn by looking at sequences
        of (y, y_dot, u).
    """
    
    @abstractmethod
    @contract(returns=Sink)
    def get_learner_u_y_y_dot(self):
        """ 
            Returns a Sink that receives dictionaries
            dict(y=..., y_dot=..., u=...)
        """
        
    def get_learner_as_sink(self):
        sink1 = self.get_learner_u_y_y_dot()
        
        dbox = get_sync_deriv_box(y_name='observations', 
                                  u_name='commands', 
                                  out_name='y_u')
        return series(dbox, Info('deriv'), sink1)