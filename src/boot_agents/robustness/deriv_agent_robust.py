from abc import abstractmethod
from blocks import check_timed_named, series
from blocks.library import Instantaneous
from boot_agents.deriv.sync_box import get_sync_deriv_box
from bootstrapping_olympics import BasicAgent, LearningAgent
from conf_tools import instantiate_spec
from contracts import check_isinstance, contract

__all__ = [
    'DerivAgentRobust',
]

class DerivAgentRobust(BasicAgent, LearningAgent):
    """ 
        Generic agent that looks at the derivative,
        and knows how to compute the importance 
    """
    
    @contract(importance='code_spec')
    def __init__(self, importance):
        """ importance: spec to instantiate """
        self.importance = instantiate_spec(importance)
        self.count = 0
        
    def init(self, boot_spec):
        # TODO: check
        pass

    def get_learner_as_sink(self):
        sink1 = self.get_learner_u_y_y_dot_w()
        
        class ComputeImportance(Instantaneous):
            def __init__(self, importance):
                self.importance = importance
            def transform_value(self, value):
                check_timed_named(value)
                _, (_, v) = value 
                check_isinstance(v, dict)
                y = v['y']
                y_dot = v['y_dot']            
                w = self.importance.get_importance(y, y_dot).astype('float32')
                u = v['u']
                return dict(u=u,y=y,y_dot=y_dot,w=w)
            
        compute_importance = ComputeImportance(self.importance)
        return series(get_sync_deriv_box(), compute_importance, sink1)


    @abstractmethod
    def get_learner_u_y_y_dot_w(self):
        """ 
            Returns a Sink that receives dictionaries
            dict(y=..., y_dot=..., u=..., w=)
        """

    def publish(self, pub):
        with pub.subsection('importance') as sub:
            if sub:
                self.importance.publish(sub)
