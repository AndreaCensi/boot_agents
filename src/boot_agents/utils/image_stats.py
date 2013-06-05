from . import MeanVariance, generalized_gradient
from contracts import contract
import numpy as np
import warnings

__all__ = ['ImageStats']


class ImageStats(object):
    """ A class to compute statistics of an image (2D float) stream. """
    
    def __init__(self):
        # Mean and variance of the luminance
        self.mv = MeanVariance()
        # Mean and variance of the luminance gradient
        self.gmv = [MeanVariance(), MeanVariance()] 
        # Mean and variance of the luminance gradient norm
        self.gnmv = MeanVariance()
        self.num_samples = 0
        self.last_y = None

    def merge(self, other):
        warnings.warn('implemnt initialized(), etc.')
        self.mv.merge(other.mv)
        self.gmv[0].merge(other.gmv[0])
        self.gmv[1].merge(other.gmv[1])
        self.gnmv.merge(other.gnmv)
        self.num_samples += other.num_samples
        
    def get_num_samples(self):
        return self.num_samples
    
    @contract(y='array[HxW]')
    def update(self, y, dt=1.0):
        self.last_y = y.copy()
        
        # luminance
        self.mv.update(y, dt)
        
        # luminance gradient
        gy = generalized_gradient(y)    
        self.gmv[0].update(gy[0, ...], dt)
        self.gmv[1].update(gy[1, ...], dt)

        # gradient norm
        gradient_norm = np.hypot(gy[0, ...], gy[1, ...])
        self.gnmv.update(gradient_norm, dt)
                
        self.num_samples += dt

    def publish(self, pub):
        if self.num_samples == 0:
            pub.text('warning',
                     'Cannot publish anything as I was never updated.')
            return

        stats = "Shape: %s " % str(self.last_y.shape)
        stats += 'Num samples: %s' % self.num_samples
        pub.text('stats', stats)

        publish_section_mean_stddev(pub, 'y', self.mv)
        publish_section_mean_stddev(pub, 'grad0', self.gmv[0])
        publish_section_mean_stddev(pub, 'grad1', self.gmv[1])
        publish_section_mean_stddev(pub, 'gradient_norm', self.gnmv)
            
    
def publish_section_mean_stddev(pub, name, mv):
    sec = pub.section(name)
    mean, stddev = mv.get_mean_stddev()
    sec.array_as_image('mean', mean)
    sec.array_as_image('sdtdev', stddev, filter='scale')
    
    
    
    
    
