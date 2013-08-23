from abc import abstractmethod
from boot_agents.utils import DerivativeBox, RemoveDoubles
from bootstrapping_olympics import AgentInterface
from conf_tools import instantiate_spec
import warnings


__all__ = ['DerivAgentRobust']


class DerivAgentRobust(AgentInterface):
    """ 
        Generic agent that looks at the derivative,
        and knows how to compute the importance 
    """
    
    def __init__(self, importance, explorer):
        """ importance: spec to instantiate """
        self.importance = instantiate_spec(importance)
        self.explorer = explorer  
        self.y_deriv = DerivativeBox()
        self.rd = RemoveDoubles(0.5)  # XXX
        self.count = 0
        
    def init(self, boot_spec):
        warnings.warn('Must do this properly')
        # self.explorer.init(boot_spec)
        self.boot_spec = boot_spec
        self.commands_spec = boot_spec.get_commands()

    def process_observations(self, obs):
        dt = float(obs['dt'])
        u = obs['commands']
        y0 = obs['observations']
        episode_start = obs['episode_start']
        self.count += 1 
        self.rd.update(y0)
        if not self.rd.ready():
            return

        if episode_start:
            self.y_deriv.reset()
            return

        self.y_deriv.update(y0, dt)

        if not self.y_deriv.ready():
            return

        y_sync, y_dot_sync = self.y_deriv.get_value()

        y_dot = y_dot_sync.astype('float32')
        y = y_sync.astype('float32')
        u = u.astype('float32')
        
        w = self.importance.get_importance(y, y_dot).astype('float32')
        
        self.process_observations_robust(y=y, y_dot=y_dot, u=u, w=w)

        self.last_y = y_sync

    @abstractmethod    
    def process_observations_robust(self, y, y_dot, u, w):
        pass
    
    def publish(self, pub):
        with pub.subsection('importance') as sub:
            if sub:
                self.importance.publish(sub)


    def choose_commands(self):
        return self.explorer.choose_commands()
