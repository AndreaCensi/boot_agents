from .bdse_model import BDSEmodel
from abc import abstractmethod
from contracts import ContractsMeta, contract
from decent_logs import WithInternalLog
from reprep import Report

__all__ = ['BDSEEstimatorInterface']


class BDSEEstimatorInterface(WithInternalLog):
    """
        Estimates a BDSE model.
    
    """

    __metaclass__ = ContractsMeta

    @abstractmethod
    @contract(u='array[K],K>0,finite',
              y='array[N],N>0,finite',
              y_dot='array[N],finite')
    def update(self, y, u, y_dot, w=1.0):
        """ merges this data """

    class NotReady(Exception):
        """ The model is not ready to be estimated. """
        
    @abstractmethod
    @contract(returns=BDSEmodel)    
    def get_model(self):
        """ Returns the estimated model or raises NotReady. """

    @abstractmethod
    @contract(pub=Report)
    def publish(self, pub):
        pass
