'''
Created on Nov 5, 2012

@author: adam
'''
from . import logger
import itertools
import numpy as np
import pylab
from PIL import Image #@UnresolvedImport
import pdb

class Interpolator():
    def __init__(self, method=Image.BILINEAR):
        self.method = method
            
    def refine(self, Y, new_size):
        '''
        :param Y:        Image to be refined
        :param new_size: size in pixels of the refined image
        '''
#        pdb.set_trace()
#        logger.debug('Refining with image_interpolator.py')
        ret = np.array(Image.fromarray(Y).resize(np.flipud(new_size), self.method))
        assert ret.shape == new_size
        return ret
    
#    def test_local_coord(self):
#        orig_size = (21, 29)
#        new_size = (10, 10)
#        test_indexes = [0, 99, 29 * 10]
#        red = np.zeros(new_size)
#        green = np.zeros(new_size)
#        blue = np.zeros(new_size)
#        for index in test_indexes:
#            red.reshape(())[index]
#        
#        coord = self.get_local_coord(orig_size, new_size, flat_index)
    
    def get_local_coord(self, orig_size, new_size, flat_index):
        '''
        Get the coordinate d \in a continuous SensorDomain of the pixel with 
        index <flat_index> in a refined discreet domain.
        
        E.g.     
        The search area was initially of size 15x15 pixels and was refined to 
        25x25 pixels by interpolating. 
        A pixel a location (7, 7) in the flat structure of the original sized 
        image has index 112, while in the refined image (which has more pixels)
        the pixel number 112 has the coordinate  (2.333, 7.0). 
        This function returns the coordinate in the refined image 
        
            > get_local_coord((15,15),(25,25), 112)
            (2.3333333333333335, 7.0)
            
        :param orig_size:
        :param new_size:
        :param flat_index:
        '''
        res = new_size
        XY = list(itertools.product(np.linspace(0, orig_size[0] - 1, res[0]),
                                    np.linspace(0, orig_size[1] - 1, res[1])))
        local_coord = XY[flat_index]
#        logger.debug('image_interpolator returns local_coord = ' + str(local_coord))
        return local_coord
    
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
    '''
    Test function for the Interpolator class
    '''
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

#from boot_agents.diffeo.learning.interpolators import Interpolator
#from boot_agents.diffeo.learning.interpolators import FourierInterpolator
#import numpy as np
#M = np.array([[10, 20, 30], [40, 50, 60]]).astype('uint8')
#itp = Interpolator()
#ftp = FourierInterpolator()
#itp.refine(M, (2, 3))
#ftp.refine(M, (2, 3))
#ii = itp.refine(M, (9, 11))
#ff = ftp.refine(M, (9, 11)).astype('int')
