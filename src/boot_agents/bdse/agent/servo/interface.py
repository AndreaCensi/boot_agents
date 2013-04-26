from abc import abstractmethod
from bootstrapping_olympics.interfaces.agent import (
    ServoAgentInterface) 

class BDSEServoInterface(ServoAgentInterface):
    
    @abstractmethod
    def set_model(self, model):
        pass
    
    def choose_commands_ext(self):
        """ This can return a dict, of which 'u' are the commands. """
        res = {}
        res['u'] = self.choose_commands()
        return res
