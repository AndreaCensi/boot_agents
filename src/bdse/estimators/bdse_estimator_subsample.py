from contracts import contract
from numpy.random import RandomState

from .interface import BDSEEstimatorInterface
from conf_tools import instantiate_spec


__all__ = [
    'BDSEEstimatorSubsample',
]


class BDSEEstimatorSubsample(BDSEEstimatorInterface):
    """
        Wraps another estimator and subsamples the data it receives
        by the given fraction.
        
        There is an internal RNG, so that multiple agents with
        different fractions use the same sequence. This implies
        that if fraction1 > fraction2, the subset of observations
        seen by agent1 is larger than for agent2.
    """
    
    @contract(fraction='float,>0,<=1', estimator='code_spec')
    def __init__(self, fraction, estimator, seed=0xc0ffee):
        self.estimator = instantiate_spec(estimator)
        self.fraction = fraction
        self.seed = seed
        self.rng = RandomState(seed)
        self.num_received = 0
        self.num_accepted = 0
        
    def update(self, y, u, y_dot, w=1.0):
        lucky = self.rng.uniform() <= self.fraction
        # self.info('frac: %.3f lucky: %.3f' % (self.fraction, lucky))
        self.num_received += 1
        if lucky:
            self.num_accepted += 1
            self.estimator.update(y, u, y_dot, w) 

    def get_model(self):
        return self.estimator.get_model()
    
    def publish(self, pub):
        pub.text('wrap', 'Wrapped by subsample, fraction = %f, received: %d accepted: %d frac: %f' % 
                 (self.fraction, self.num_received, self.num_accepted,
                  self.num_accepted * 1.0 / (self.num_received + 1)))
        
        # pub.text('rng', str(self.rng))
        # pub.text('rng_state', str(self.rng.get_state()))
        
        self.estimator.publish(pub)
