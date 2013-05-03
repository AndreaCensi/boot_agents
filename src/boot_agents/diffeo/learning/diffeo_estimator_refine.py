from . import logger
from .. import (contract, np)
from .interpolators import Interpolator, ImageInterpolatorFast, FourierInterpolator
from PIL import Image  # @UnresolvedImport
from boot_agents.diffeo.diffeomorphism2d_continuous import Diffeomorphism2DContinuous
from boot_agents.diffeo.plumbing import togrid, add_border
from reprep.plot_utils import plot_vertical_line 
from boot_agents.diffeo.plumbing.flat_structure import flat_structure_cache

REFINE_STANDARD_BILINEAR = 'standard-bilinear'
REFINE_STANDARD_BICUBIC = 'standard-bicubic'
REFINE_STANDARD_ANTIALIAS = 'standard-antialias'
REFINE_FAST_BILINEAR = 'fast-bilinear'
REFINE_FAST_BICUBIC = 'fast-bicubic'
REFINE_FAST_ANTIALIAS = 'fast-antialias'
REFINE_FFT = 'fft'

class DiffeomorphismEstimatorRefine():
    '''
    This is the first version of a refining learner. It resizes the image of each  
    search area using a specified interpolator.  
    
    '''
    @contract(max_displ='seq[2](>0,<1)')
    def __init__(self, max_displ, refine_method, resolution, refine_factor):
        """ 
            :param max_displ: Maximum displacement  
            :param inference_method: order, sim
        """
        self.shape = None
        self.max_displ = np.array(max_displ)
        self.last_y0 = None
        self.last_y1 = None
        self.res = tuple(resolution)
        self.refine_factor = refine_factor
        if refine_method == REFINE_STANDARD_BILINEAR:
            self.interpolator = Interpolator(Image.BILINEAR)
            
        elif refine_method == REFINE_STANDARD_BICUBIC:
            self.interpolator = Interpolator(Image.BICUBIC)
            
        elif refine_method == REFINE_STANDARD_ANTIALIAS:
            self.interpolator = Interpolator(Image.ANTIALIAS)
            
        # Fast methods
        elif refine_method == REFINE_FAST_BILINEAR:
            self.interpolator = ImageInterpolatorFast(Image.BILINEAR)
            
        elif refine_method == REFINE_FAST_BICUBIC:
            self.interpolator = ImageInterpolatorFast(Image.BICUBIC)
            
        elif refine_method == REFINE_FAST_ANTIALIAS:
            self.interpolator = ImageInterpolatorFast(Image.ANTIALIAS)
            
        elif refine_method == REFINE_FFT:
            self.interpolator = FourierInterpolator()
            
        else:
            assert False
        self.num_refined = 0
        self.num_samples = 0
        self.buffer_NA = None
        
    def refine_init(self):
        r = self.refine_factor
#        r = 1.5 # ratio to refine area for next learning

        A_sug = np.zeros(self.A.shape)
        B_sug = np.zeros(self.B.shape)
        for i in range(self.nsensels):
            best = np.argmin(self.neig_esim_score[i])
            best_coord_local = self.A[i] + self.interpolator.get_local_coord(self.B[i], self.res, best)
            
            
            # Check some conditions of best_cord_local
            if (best_coord_local < self.A[i]).any():
                # Then something is wrong because such point should not be 
                # possible to find at all
                logger.error('best_coord_local < self.A[i] best_cord_local is: ' 
                             + str(best_coord_local) + 'while A[i] is only: ' 
                             + str(self.A[i]))
            
            if (best_coord_local > self.A[i] + self.B[i]).any():
                # Then something is wrong because such point should not be 
                # possible to find at all
                logger.error('best_coord_local > self.A[i] + self.B[i] ' + 
                             'best_cord_local is: ' + str(best_coord_local) + 
                             'while A[i] is only: ' + str(self.A[i]))
#                pdb.set_trace()
            
            A_sug[i] = np.floor(best_coord_local - np.array(self.B[i]).astype('float64') / (2 * r)).astype('int')
            B_sug[i] = np.ceil(np.array(self.B[i]).astype('float64') / r).astype('int')
            
            
            inval = (A_sug[i] < -np.array(self.area) / 2) 
            if (inval).any():
                A_sug[i, inval] = -np.array(self.area)[inval] / 2 
#                logger.debug('A_sug[i] < -self.area/2: ' + 
#                             str(A_sug[i]) + ' < ' + str(-np.array(self.area) / 2))
      
            inval = (A_sug[i] + B_sug[i] > np.array(self.area) / 2) 
            if (inval).any():
                A_sug[i, inval] = np.array(self.area)[inval] / 2 - B_sug[i, inval] 
#                logger.debug('A_sug[i] + B_sug[i] > self.area/2: ' + 
#                             str(B_sug[i] + B_sug[i]) + ' > ' + str(np.array(self.area) / 2))
      
        self.num_samples = 0
        self.num_refined += 1

        self.A = A_sug
        self.B = B_sug
        
        
    @contract(y0='array[MxN]', y1='array[MxN]')
    def update(self, y0, y1):
        if self.shape is None:
            logger.info('Initiating structure from update()')
            self.init_structures(y0)
        
        res = self.res
        res_size = np.prod(res)
#        pdb.set_trace()
        for i in range(self.nsensels): 
            a = self.A[i]
            b = self.B[i]
            c = self.C[i]
                        
            xl = c[0] + a[0]
            xu = c[0] + a[0] + b[0]
            yl = c[1] + a[1]
            yu = c[1] + a[1] + b[1]
#            pdb.set_trace()
            if self.interpolator.__class__ == ImageInterpolatorFast:
                pass
            else:
                Yi_sub = self.interpolator.extract_wraparound(y0, ((xl, xu), (yl, yu)))
#            if i > 80 * 20 + 40:
#                pdb.set_trace()
                Yi_ref = self.interpolator.refine(Yi_sub, res)
                
            diff = np.abs(Yi_ref.astype('float') - y1[tuple(c)]).reshape(res_size)
#            pdb.set_trace()
            self.neig_esim_score[i] += diff
        self.num_samples += 1

    def init_structures(self, y):
        self.shape = y.shape
        # for each sensel, create an area
        self.area = np.ceil(self.max_displ * np.array(self.shape)).astype('int32')
        
        # ensure it's an odd number of pixels
        for i in range(2):
            if self.area[i] % 2 == 0:
                self.area[i] += 1
        self.area = (int(self.area[0]), int(self.area[1]))
        self.nsensels = y.size
        self.area_size = self.area[0] * self.area[1]
        self.res_size = self.res[0] * self.res[1]
        logger.debug(' Field Shape: %s' % str(self.shape))
        logger.debug('    Fraction: %s' % str(self.max_displ))
        logger.debug(' Search area: %s' % str(self.area))
        logger.debug('Creating FlatStructure...')
        self.flat_structure = flat_structure_cache(self.shape, self.area)
        self.refine_flat_structure = flat_structure_cache(self.shape, self.res)
        logger.debug('done creating')
        
        assert(self.num_refined == 0)
        if self.num_refined == 0:
            logger.debug('First time estimator is initiated, generating A, B and C')
            self.A = np.zeros((self.nsensels, 2))
            self.B = np.zeros((self.nsensels, 2))
            self.C = self.flat_structure.flattening.index2cell
        
        for i in range(self.nsensels):
            a = -np.array(self.area) / 2
            b = np.array(self.area)
            self.A[i] = a
            self.B[i] = b

        buffer_shape = (self.nsensels, self.res_size)
        self.neig_esim_score = np.zeros(buffer_shape, 'float32')
        
        # initialize a buffer of size NxA
        self.buffer_NA = np.zeros(buffer_shape, 'float32') 

    def display(self, report):
        self.show_areas(report, self.dd)
        
        
        report.data('num_samples', self.num_samples)
        f = report.figure(cols=4)
        
        
        def make_best(x):
            return x == np.min(x, axis=1)
        
        @contract(score='array[NxA]', returns='array[HxW],H*W=N')
        def distance_to_border_for_best(score):
            N, _ = score.shape
            best = np.argmin(score, axis=1)
            assert best.shape == (N,)
            D = self.flat_structure.get_distances_to_area_border()
            res = np.zeros(N)
            for i in range(N):
                res[i] = D[i, best[i]]
            return self.flat_structure.flattening.flat2rect(res)

        @contract(score='array[NxA]', returns='array[HxW],H*W=N')
        def distance_from_center_for_best(score):
            N, _ = score.shape
            best = np.argmin(score, axis=1)
            assert best.shape == (N,)
            D = self.flat_structure.get_distances()
            res = np.zeros(N)
            for i in range(N):
                res[i] = D[i, best[i]]
            return self.flat_structure.flattening.flat2rect(res)

        max_d = int(np.ceil(np.hypot(np.floor(self.area[0] / 2.0),
                                     np.floor(self.area[1] / 2.0))))
        safe_d = int(np.floor(np.min(self.area) / 2.0))
        
        bdist_scale = dict(min_value=0, max_value=max_d, max_color=[0, 1, 0])
        cdist_scale = dict(min_value=0, max_value=max_d, max_color=[1, 0, 0])
        bins = range(max_d + 2)
        
        def plot_safe(pylab):
            plot_vertical_line(pylab, safe_d, 'g--')
            plot_vertical_line(pylab, max_d, 'r--')
      
#        pdb.set_trace()
        esim = self.make_grid(self.neig_esim_score)
        report.data('neig_esim_score_rect', esim).display('scale').add_to(f, caption='sim')
#        esim_bdist = distance_to_border_for_best(self.neig_esim_score)
#        esim_cdist = distance_from_center_for_best(self.neig_esim_score)
#        report.data('esim_bdist', esim_bdist).display('scale', **bdist_scale).add_to(f, caption='esim_bdist')
#        report.data('esim_cdist', esim_cdist).display('scale', **cdist_scale).add_to(f, caption='esim_cdist')
    
#        with f.plot('esim_bdist_hist') as pylab:
#            pylab.hist(esim_bdist.flat, bins)
#        with f.plot('esim_cdist_hist') as pylab:
#            pylab.hist(esim_cdist.flat, bins)
#            plot_safe(pylab)
            
    @contract(score='array[NxA]', returns='array[UxV]')  # ,U*V=N*A') not with border
    def make_grid(self, score):
        fourd = self.refine_flat_structure.unrolled2multidim(score)  # HxWxXxY
        return togrid(add_border(fourd))      
    
    def show_areas(self, report, d):
        Y, X = np.meshgrid(range(d.shape[1]), range(d.shape[0]))
        dx_local = d[:, :, 0] - X
        dy_local = d[:, :, 1] - Y 
        f = report.figure(cols=3)
        
        with f.plot('search_box', caption='All search boxes') as pylab:
            pylab.hold(True)
            
            nbx = np.max(dx_local) - np.min(dx_local)
            nby = np.max(dy_local) - np.min(dy_local)
#            hist_range = [[np.min(dx_local), np.max(dx_local)],
#                     [np.min(dy_local), np.max(dy_local)]] 
            hist_range = [[-(self.area[0] + 2.0) / 2, (self.area[0] + 2.0) / 2],
                          [-(self.area[1] + 2.0) / 2, (self.area[1] + 2.0) / 2]]
            
            Hc, xe, ye = np.histogram2d(dx_local.flatten(), dy_local.flatten(),
                                        range=hist_range, bins=(nbx, nby))
#            pdb.set_trace()
            Yh, Xh = np.meshgrid(ye[:-1] + .5 * (ye[1] - ye[0]),
                                 xe[:-1] + .5 * (xe[1] - xe[0]))            
            pylab.contourf(Xh, Yh, Hc, cmap=pylab.get_cmap('Blues'), alpha=1)
            
            
            Hc, xe, ye = np.histogram2d(self.A[:, 0], self.A[:, 1],
                                        range=hist_range, bins=(nbx, nby))
            Yh, Xh = np.meshgrid(ye[:-1] + 0.5 * (ye[1] - ye[0]),
                                 xe[:-1] + 0.5 * (xe[1] - xe[0]))            
            pylab.contourf(Xh, Yh, Hc, cmap=pylab.get_cmap('Reds'), alpha=0.5)
            
#            for i in range(self.nsensels):
#                a = self.A[i]
#                b = self.B[i]
#                
#                boxx = np.array([a[0], a[0] + b[0], a[0] + b[0], a[0], a[0]])
#                boxy = np.array([a[1], a[1], a[1] + b[1], a[1] + b[1], a[1]])
#            
#                pylab.plot(boxx, boxy)
            
            pylab.plot(dx_local.reshape(dx_local.size),
                       dy_local.reshape(dy_local.size),
                       linestyle='none', marker='.')


    def summarize(self):
        ''' 
            Find maximum likelihood estimate for diffeomorphism looking 
            at each pixel singularly. 
            
            Returns a Diffeomorphism2D.
        '''
        dd_local = np.zeros((self.shape[0], self.shape[1], 2))
        dd = np.zeros((self.shape[0], self.shape[1], 2))
        for i in range(self.nsensels):
            best = np.argmin(self.neig_esim_score[i])
#            logger.info('best coord is: ' + str(best))
            best_coord_local = self.A[i] + self.interpolator.get_local_coord(self.B[i],
                                                                             self.res,
                                                                             best)
            best_coord = self.C[i] + best_coord_local
            
            dd_local[tuple(self.C[i])] = best_coord_local
            dd[tuple(self.C[i])] = best_coord
        cert_flat = (self.neig_esim_score - np.min(self.neig_esim_score)) / (np.max(self.neig_esim_score) - np.min(self.neig_esim_score))

        cert = np.min(cert_flat, axis=1).reshape(self.shape)
        self.dd = dd
        return Diffeomorphism2DContinuous(dd, cert)
    
    def publish(self, pub):
        pass
    
    def merge(self, other):
        pass
    
def compare():
    pass
