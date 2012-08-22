from . import (diffeo_apply, contract, np, diffeo_to_rgb_norm,
    diffeo_to_rgb_angle, diffeo_distance_L2, diffeo_stats, scalaruncertainty2rgb,
    diffeo_local_differences)
from boot_agents.diffeo.diffeo_basic import diffeo_identity
from boot_agents.diffeo.diffeo_apply_quick import FastDiffeoApply
from geometry.utils.numpy_backport import assert_allclose


class Diffeomorphism2D:
                
    @contract(d='valid_diffeomorphism,array[HxWx2]', variance='None|array[HxW]')
    def __init__(self, d, variance=None):
        ''' 
            This is a diffeomorphism + variance.
            
            d: [M, N, 2]
            variance: [M, N]
            
            d: discretized version of what we called phi
               phi : S -> S
               
               d: [1,W]x[1,H] -> [1,W]x[1,H]
                
            variance: \Gamma in the paper 
        '''
        self.d = d
        
        if variance is None:
            self.variance = np.ones((d.shape[0], d.shape[1]))
        else:
            assert variance.shape == d.shape[:2]
            assert np.isfinite(variance).all()
            self.variance = variance.astype('float32')

    @staticmethod
    def identity(shape):
        return Diffeomorphism2D(diffeo_identity(shape))

    @contract(returns='array[HxW]')
    def get_scalar_info(self):
        """ 
            Returns a scalar value which has the interpretation
            of 1 = certain, 0 = unknown. 
        """
        return self.variance
    
    @contract(returns='valid_diffeomorphism')
    def get_discretized_diffeo(self):
        """ 
            Returns a valid_diffeomorphism (discrete cell-to-cell map)
        """
        return self.d

    def _d_apply(self, template):
        if not 'fda' in self.__dict__:
            self.fda = FastDiffeoApply(self.d)
        result = self.fda(template) 
        if False:
            assert_allclose(result, diffeo_apply(self.d, template))
        return result
    
    @contract(im='array[HxWx...]', var='None|array[HxW]',
              returns='tuple(array[HxWx...], array[HxW])')
    def apply(self, im, var=None):
        """
            Apply diffeomorphism <self> to image <im>. 
            <im> is array[HxWx...]
            <var> is the variance of diffeomorphism
        """
        dd_info = self.get_scalar_info()
        
        im2 = self._d_apply(im)
        
        if var is None:
            '''
            var tells how certain we are about the map from pigel (i,j) in var.
            which results in an uncertainty of the corresponding mapped pixel in 
            the new image.  
            '''
            var = np.ones((im.shape[0], im.shape[1]))
            var2 = self._d_apply(var)
        else:
            var2 = dd_info * self._d_apply(var)
            
        return im2, var2
    
    @staticmethod
    def compose(d1, d2):
        """ Composes two Diffeomorphism2D objects. """
        assert isinstance(d1, Diffeomorphism2D)
        assert isinstance(d2, Diffeomorphism2D)
        # XXX: this can be improved
        im, var = d1.apply(d2.d, d2.variance)
        return Diffeomorphism2D(im, var)
            
    @staticmethod
    def distance_L2(d1, d2):
        """ Distance that does not take into account the uncertainty. """
        assert isinstance(d1, Diffeomorphism2D)
        assert isinstance(d2, Diffeomorphism2D)
        return diffeo_distance_L2(d1.d, d2.d)

    @staticmethod
    def distance_L2_infow(d1, d2):
        """ 
            Distance that weights the mismatch by the product
            of the uncertainties. 
        """
        assert isinstance(d1, Diffeomorphism2D)
        assert isinstance(d2, Diffeomorphism2D)
        dd1 = d1.get_discretized_diffeo()
        dd2 = d2.get_discretized_diffeo()
        dd1_info = d1.get_scalar_info()
        dd2_info = d2.get_scalar_info()
        
        x, y = diffeo_local_differences(dd1, dd2)
        dist = np.hypot(x, y)
        info = dd1_info * dd2_info
        info_sum = info.sum()
        if info_sum == 0:
            #print('dd1_info.mean() = %g' % dd1_info.mean())
            #print('dd2_info.mean() = %g' % dd2_info.mean())
            #print('info.mean() = %g' % info.mean())
            return 1.0 # FIXME, need to check it is the bound
            
            # raise NotImplementedError
        
        # not sure what's happening here
        wdist = (dist * info) / info_sum
        
        res = wdist.sum()
        #print('res: %s  but unweighted: %s' % (res, dist.mean()))
        #print('res: %s' % res)
        return res
    
    @staticmethod
    def distance_L2_infow_scaled(d1, d2):
        # XXX: written while rushing
        a = Diffeomorphism2D.distance_L2_infow(d1, d2)
        dd1_info = d1.get_scalar_info()
        dd2_info = d2.get_scalar_info()
        #x = dd1_info
        #print('min %g max %g' % (x.max(), x.min()))
        b = np.mean(np.abs(dd1_info - dd2_info)) # / dd1_info.size
        #print('a, b: %.5f %.5f   mean %g %g' % (a, b, dd1_info.mean(), dd2_info.mean()))
        return b + min(a, 0.5) # a * (1 + b)
         
    def get_shape(self):
        return (self.d.shape[0], self.d.shape[1])
    
    def get_rgb_norm(self):
        return diffeo_to_rgb_norm(self.get_discretized_diffeo())
    
    def get_rgb_angle(self):
        return diffeo_to_rgb_angle(self.get_discretized_diffeo())

    def get_rgb_info(self):
        return scalaruncertainty2rgb(self.get_scalar_info())
    
    def display(self, report, full=False, nbins=100):
        """ Displays this diffeomorphism. """
        stats = diffeo_stats(self.d)
        angle = stats.angle
        norm = stats.norm
        
        norm_rgb = self.get_rgb_norm()
        angle_rgb = self.get_rgb_angle()
        info_rgb = self.get_rgb_info()
        
        f = report.figure(cols=6)
        f.data_rgb('norm_rgb', norm_rgb,
                    caption="Norm(D). white=0, blue=maximum. "
                            "Note: wrong in case of wraparound")
        f.data_rgb('phase_rgb', angle_rgb,
                    caption="Phase(D). Note: wrong in case of wraparound")
        
        f.data_rgb('var_rgb', info_rgb,
                    caption='Uncertainty (green=sure, red=unknown)')

        with f.plot('norm_hist', caption='histogram of norm values') as pylab:
            pylab.hist(norm.flat, nbins)

        angles = np.array(angle.flat)
        valid_angles = angles[np.logical_not(np.isnan(angles))]
        with f.plot('angle_hist', caption='histogram of angle values '
                    '(excluding where norm=0)') as pylab:
            pylab.hist(valid_angles, nbins)

        with f.plot('var_hist', caption='histogram of certainty values') as pylab:
            pylab.hist(self.variance.flat, nbins)

 
