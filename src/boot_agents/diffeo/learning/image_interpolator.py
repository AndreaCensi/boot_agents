'''
Created on Nov 5, 2012

@author: adam
'''
from . import logger
import itertools
import scipy
import numpy as np
import pylab

def memoize(f):
    cache = {}
    def memf(*x):
        if x not in cache:
            cache[x] = f(*x)
        return cache[x]
    return memf

class Interpolator():
    def __init__(self):
        pass
            
    @memoize
    def basis(self, orig_size, new_size):
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
    
    def get_local_coord(self, orig_size, new_size, flat_index):
        resolution = new_size
        XY = list(itertools.product(np.linspace(0, orig_size[0] - 1, resolution[0]), np.linspace(0, orig_size[1] - 1, resolution[1])))
        return XY[flat_index]
    
    def refine(self, Y, new_size):
#        logger.info('refining')
        S = np.fft.fft2(Y)
        orig_size = Y.shape[:2]
        S_flat = S.reshape(np.prod(orig_size))
        M = self.basis(orig_size, new_size)
#        pdb.set_trace()
        Y_new_flat = np.array(M * np.mat(S_flat).T)
        return np.real(Y_new_flat.reshape(new_size)) / np.prod(orig_size)

    def extract_wraparound(self, Y, ((xl, xu), (yl, yu))):
        '''
        Y[xl:xu,yl:yu] with a wrap around effect
        '''
        xsize, ysize = Y.shape
        
        # Assert valid dimensions
        assert(xu > xl)
        assert(yu > yl)
        
        # Extract image in x-direction
        if xu < 0 or xl > xsize:
            # Complete wrap around
            Yx = Y[xl % xsize:xu % xsize]
        elif xl < 0:
            # Partial wrap around on lower bound
            Yx = np.concatenate((Y[xl:], Y[:xu]), axis=0)
        elif xu >= xsize:
            # Partial wrap around on upper bound
            Yx = np.concatenate((Y[xl:], Y[:xu % xsize]), axis=0)
        else:
            # Normal interval
            Yx = Y[xl:xu]
        
        
        # Extract image in y-direction from Yx
        if yu < 0 or yl > ysize:
            # Complete wrap around
            Yi_sub = Yx[:, yl % ysize:yu % ysize] 
        elif yl < 0:
            # Partial wrap around on lower bound
            Yi_sub = np.concatenate((Yx[:, yl:], Yx[:, :yu]), axis=1)
        elif yu >= ysize:
            # Partial wrap around on upper bound
            Yi_sub = np.concatenate((Yx[:, yl:], Yx[:, :yu % ysize]), axis=1)
        else:
            # Normal interval
            Yi_sub = Yx[:, yl:yu]
            
        return Yi_sub

if __name__ == '__main__':
    orig_size = (5, 4)
    interp = Interpolator()
#    interp.init_basis((10, 10))
    X, Y = np.meshgrid(range(orig_size[0]), range(orig_size[1]))
    Y0 = np.sin(X + 1) * np.sin(Y + .5) + np.random.ranf((orig_size[1], orig_size[0])) * 0.1
    
    pylab.figure()
    pylab.subplot(2, 2, 1)
    pylab.imshow(Y0, origin='lower', interpolation='none')
    pylab.xlim((-.5, orig_size[0] - .5))
    pylab.ylim((-.5, orig_size[1] - .5))
    pylab.colorbar()
    pylab.title('Original Image')
    
    # 
    pylab.subplot(2, 2, 2)
    
    new_size = (10, 8)
    Yn = interp.refine(Y0, (new_size[1], new_size[0]))
    
    pylab.imshow(Yn, origin='lower', interpolation='none')
    pylab.colorbar()
    pylab.title('Fourier Transformed')
    pylab.xlim((-.5, new_size[0] - .5))
    pylab.ylim((-.5, new_size[1] - .5))

    # Plot 3
    pylab.subplot(2, 2, 3)
    
    new_size = (20, 16)
    Yn = interp.refine(Y0, (new_size[1], new_size[0]))
    
    pylab.imshow(Yn, origin='lower', interpolation='none')
    pylab.colorbar()
    pylab.title('Fourier Transformed')
    pylab.xlim((-.5, new_size[0] - .5))
    pylab.ylim((-.5, new_size[1] - .5))

    # Plot 4
    pylab.subplot(2, 2, 4)
    
    new_size = (40, 32)
    Yn = interp.refine(Y0, (new_size[1], new_size[0]))
    
    pylab.imshow(Yn, origin='lower', interpolation='none')
    pylab.colorbar()
    pylab.title('Fourier Transformed')
    pylab.xlim((-.5, new_size[0] - .5))
    pylab.ylim((-.5, new_size[1] - .5))

    
    pylab.savefig('y.png')
    
#    pdb.set_trace()
