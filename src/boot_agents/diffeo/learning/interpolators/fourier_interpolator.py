
from . import logger
from . import Interpolator
from compmake.utils import memoize
import itertools
import scipy
import numpy as np

def memoize(f):
    '''
    Support functions for memoize
    :param f:
    '''
    cache = {}
    def memf(*x):
        if x not in cache:
            cache[x] = f(*x)
        return cache[x]
    return memf
class FourierInterpolator(Interpolator):
    
    
    @memoize
    def basis(self, orig_size, new_size):
        '''
        This function calculates (and memorizes) a basis matrix from the 
        discreet Fourier-domain to the image domain.
        :param orig_size:
        :param new_size:
        '''
        
        logger.info('Creating new basis representation with: ' + str(orig_size) + ' and: ' + str(new_size))
        j = complex(0, 1)
        def f(k, n):
            return (2 * float(k) / n) % 1 - (2 * k / n)
        
        resolution = new_size
        C = list(itertools.product(range(orig_size[0]), range(orig_size[1])))
        F = np.array([[f(c[0], orig_size[0]), f(c[1], orig_size[1])] for c in C])
        XY = list(itertools.product(np.linspace(0, orig_size[0] - 1, resolution[0]), np.linspace(0, orig_size[1] - 1, resolution[1])))
        
        M = np.mat([[scipy.exp(j * np.pi * np.sum(F[k] * XY[xy])) for k in range(len(F))] for xy in range(len(XY))])
        return M

    
    def refine(self, Y, new_size):
        '''
        :param Y:        Image to be refined
        :param new_size: size in pixels of the refined image
        '''
#        logger.debug('Refining with fourier_interpolator.py')
        # Get the Fourier coefficients for the image Y 
        S = np.fft.fft2(Y)
        orig_size = Y.shape[:2]
        
        # Reshape to a flat structure to get a vector for multiplication with 
        # the basis matrix
        S_flat = S.reshape(np.prod(orig_size))
        
        # Using a Fourier-basis matrix to interpolate
        M = self.basis(orig_size, new_size)
        Y_new_flat = np.array(M * np.mat(S_flat).T) / np.prod(orig_size)
        
        # Reshape the image to correct shape
        return np.real(Y_new_flat.reshape(new_size))
