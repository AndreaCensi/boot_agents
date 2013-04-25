from bootstrapping_olympics.interfaces.agent import AgentInterface
from abc import abstractmethod
from contracts import contract


class BDSEServoInterface(AgentInterface):
    
    @abstractmethod
    def set_model(self, model):
        pass

    @abstractmethod
    @contract(goal='array')
    def set_goal_observations(self, goal):
        pass
    
    def choose_commands_ext(self):
        """ This can return a dict, of which 'u' are the commands. """
        res = {}
        res['u'] = self.choose_commands()
        return res

    @abstractmethod
    @contract(returns='array')
    def choose_commands(self):
        """ This must return the raw commands. """ 
        pass
