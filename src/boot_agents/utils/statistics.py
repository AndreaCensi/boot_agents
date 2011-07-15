
#Here are a couple references on computing sample variance.
#
#Chan, Tony F.; Golub, Gene H.; LeVeque, Randall J. (1983). 
#Algorithms for Computing the Sample Variance: Analysis and Recommendations. 
#The American Statistician 37, 242-247.
#
#Ling, Robert F. (1974). Comparison of Several Algorithms for Computing Sample 
#Means and Variances. Journal of the American Statistical Association,
# Vol. 69, No. 348, 859-866. 
 
from contracts import contract
import numpy as np
from numpy.linalg.linalg import pinv, LinAlgError
from numpy import  multiply
outer = multiply.outer

from . import logger

def cov2corr(covariance, zero_diagonal=False):
    ''' 
    Compute the correlation matrix from the covariance matrix.
    If zero_diagonal = True, the diagonal is set to 0 instead of 1. 

    :param zero_diagonal: Whether to set the (noninformative) diagonal to zero.
    :param covariance: A 2D numpy array.
    :return: correlation: The exctracted correlation.
    
    '''
    # TODO: add checks
    outer = np.multiply.outer

    sigma = np.sqrt(covariance.diagonal())
    M = outer(sigma, sigma)
    correlation = covariance / M
    
    if zero_diagonal:
        for i in range(covariance.shape[0]):
            correlation[i, i] = 0
    
    return correlation

@contract(A='array', wA='>=0', B='array', wB='>=0')
def weighted_average(A, wA, B, wB):
    return (wA * A + B * wB) / (wA + wB)


class Expectation:
    ''' A class to compute the mean of a quantity over time '''
    def __init__(self, max_window=None):
        ''' 
            If max_window is given, the covariance is computed
            over a certain interval. 
        '''
        self.num_samples = 0
        self.value = None
        self.max_window = max_window
        
    def update(self, value, dt=1):
        if self.value is None:
            self.value = value
        else:
            self.value = weighted_average(self.value, self.num_samples, value, dt) 
        self.num_samples += dt
        if self.max_window and self.num_samples > self.max_window:
            self.num_samples = self.max_window 
    
    def get_value(self):
        return self.value

class MeanCovariance:
    ''' Computes mean and covariance of a quantity '''
    def __init__(self, max_window=None):
        self.mean_accum = Expectation(max_window)
        self.covariance_accum = Expectation(max_window)
        self.minimum = None
        self.maximum = None
        
    def update(self, value):
        if  self.maximum is None:
            self.maximum = value
        else:
            self.maximum = np.maximum(value, self.maximum)
            
        if self.minimum is None:
            self.minimum = value
        else:
            self.minimum = np.minimum(value, self.minimum)
            
        self.mean_accum.update(value)
        mean = self.mean_accum.get_value()        
        value_norm = value - mean
        P = outer(value_norm, value_norm)
        self.covariance_accum.update(P)
        self.last_value = value
        
    def get_mean(self):
        return self.mean_accum.get_value()

    def get_maximum(self):
        return self.maximum

    def get_minimum(self):
        return self.minimum
    
    def get_covariance(self):
        return self.covariance_accum.get_value()
    
    def get_correlation(self):
        # FIXME: to implement
        return cov2corr(self.covariance_accum.get_value())
    
    def get_information(self, rcond=1e-2):
        try:
            return pinv(self.get_covariance(), rcond=rcond)
        except LinAlgError:
            filename = 'pinv-failure'
            import pickle
            with  open(filename + '.pickle', 'w') as f:
                pickle.dump(self, f)
            logger.error('Did not converge; saved on %s' % filename)
