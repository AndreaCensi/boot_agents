from boot_agents.bdse.model.bdse_estimator_interface import (
    BDSEEstimatorInterface)
from conf_tools import instantiate_spec
from contracts import contract
import numpy as np


__all__ = ['BDSEEstimatorSubsample']


class BDSEEstimatorSubsample(BDSEEstimatorInterface):
    """
        Wraps another estimator and subsamples the data it receives
        by the given fraction.
    """
    
    @contract(fraction='float,>0,<=1', estimator='code_spec')
    def __init__(self, fraction, estimator):
        self.estimator = instantiate_spec(estimator)
        self.fraction = fraction
        
    def update(self, y, u, y_dot, w=1.0):
        lucky = np.random.rand() <= self.fraction
        # self.info('frac: %.3f lucky: %.3f' % (self.fraction, lucky))
        if lucky:
            self.estimator.update(y, u, y_dot, w) 

    def get_model(self):
        return self.estimator.get_model()
    
    def publish(self, pub):
        pub.text('wrap', 'Wrapped by subsample, fraction = %f' % self.fraction)
        self.estimator.publish(pub)
