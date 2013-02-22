'''
This is the very first version of a refining estimator which works with FFTs. 
New version covered by diffeo_estimator_refine.py
'''

#from . import logger
#from .. import contract, np
#from boot_agents.diffeo.learning.image_interpolator import Interpolator
#from boot_agents.diffeo.plumbing.flat_structure import flat_structure_cache
#import pdb
#from boot_agents.diffeo.diffeomorphism2d_continuous import Diffeomorphism2DContinuous
#from boot_agents.diffeo.learning.interpolators.fourier_interpolator import FourierInterpolator
#    
#class DiffeomorphismEstimatorFFT():
#    
#    @contract(max_displ='seq[2](>0,<1)')
#    def __init__(self, max_displ):
#        """ 
#            :param max_displ: Maximum displacement  
#            :param inference_method: order, sim
#        """
#        self.shape = None
#        self.max_displ = np.array(max_displ)
#        self.last_y0 = None
#        self.last_y1 = None
#        self.res = (10, 10)
#        self.interpolator = FourierInterpolator() 
#        self.num_refined = 0
#        self.num_samples = 0
#        self.buffer_NA = None
#        
#    def refine_init(self):
#        r = 1.5 # ratio to refine area for next learning
#
#        A_sug = np.zeros(self.A.shape)
#        B_sug = np.zeros(self.B.shape)
#        for i in range(self.nsensels):
#            best = np.argmin(self.neig_esim_score[i])
#            best_coord_local = self.A[i] + self.interpolator.get_local_coord(self.B[i], self.res, best)
#            
#            
#            # Check some conditions of best_cord_local
#            if (best_coord_local < self.A[i]).any():
#                # Then something is wrong because such point should not be 
#                # possible to find at all
#                logger.error('best_coord_local < self.A[i] best_cord_local is: ' 
#                             + str(best_coord_local) + 'while A[i] is only: ' 
#                             + str(self.A[i]))
#            
#            if (best_coord_local > self.A[i] + self.B[i]).any():
#                # Then something is wrong because such point should not be 
#                # possible to find at all
#                logger.error('best_coord_local > self.A[i] + self.B[i] ' + 
#                             'best_cord_local is: ' + str(best_coord_local) + 
#                             'while A[i] is only: ' + str(self.A[i]))
##                pdb.set_trace()
#            
#            A_sug[i] = np.floor(best_coord_local - np.array(self.B[i]).astype('float64') / (2 * r)).astype('int')
#            B_sug[i] = np.ceil(np.array(self.B[i]).astype('float64') / r).astype('int')
#            
#            
#            inval = (A_sug[i] < -np.array(self.area) / 2) 
#            if (inval).any():
#                A_sug[i, inval] = -np.array(self.area)[inval] / 2 
##                logger.debug('A_sug[i] < -self.area/2: ' + 
##                             str(A_sug[i]) + ' < ' + str(-np.array(self.area) / 2))
#      
#            inval = (A_sug[i] + B_sug[i] > np.array(self.area) / 2) 
#            if (inval).any():
#                A_sug[i, inval] = np.array(self.area)[inval] / 2 - B_sug[i, inval] 
##                logger.debug('A_sug[i] + B_sug[i] > self.area/2: ' + 
##                             str(B_sug[i] + B_sug[i]) + ' > ' + str(np.array(self.area) / 2))
#      
#        self.num_samples = 0
#        self.num_refined += 1
#
#        self.A = A_sug
#        self.B = B_sug
#        
#        
#    @contract(y0='array[MxN]', y1='array[MxN]')
#    def update(self, y0, y1):
#        if self.shape is None:
#            logger.info('Initiating structure from update()')
#            self.init_structures(y0)
#        
#        res = self.res
#        res_size = np.prod(res)
#            
#        for i in range(self.nsensels): 
#            a = self.A[i]
#            b = self.B[i]
#            c = self.C[i]
#                        
#            xl = c[0] + a[0]
#            xu = c[0] + a[0] + b[0]
#            yl = c[1] + a[1]
#            yu = c[1] + a[1] + b[1]
#            
#            Yi_sub = self.interpolator.extract_wraparound(y0, ((xl, xu), (yl, yu)))
#                
#            Yi_ref = self.interpolator.refine(Yi_sub, res)
#            diff = np.abs(Yi_ref - y1[tuple(c)]).reshape(res_size)
#            self.neig_esim_score[i] += diff
#        self.num_samples += 1
#
#    def init_structures(self, y):
#        self.shape = y.shape
#        # for each sensel, create an area
#        self.area = np.ceil(self.max_displ * np.array(self.shape)).astype('int32')
#        
#        # ensure it's an odd number of pixels
#        for i in range(2):
#            if self.area[i] % 2 == 0:
#                self.area[i] += 1
#        self.area = (int(self.area[0]), int(self.area[1]))
#        self.nsensels = y.size
#        self.area_size = self.area[0] * self.area[1]
#        self.res_size = self.res[0] * self.res[1]
#        logger.debug(' Field Shape: %s' % str(self.shape))
#        logger.debug('    Fraction: %s' % str(self.max_displ))
#        logger.debug(' Search area: %s' % str(self.area))
#        logger.debug('Creating FlatStructure...')
#        self.flat_structure = flat_structure_cache(self.shape, self.area)
#        logger.debug('done creating')
#        
#        assert(self.num_refined == 0)
#        if self.num_refined == 0:
#            logger.debug('First time estimator is initiated, generating A, B and C')
#            self.A = np.zeros((self.nsensels, 2))
#            self.B = np.zeros((self.nsensels, 2))
#            self.C = self.flat_structure.flattening.index2cell
#        
#        for i in range(self.nsensels):
#            a = -np.array(self.area) / 2
#            b = np.array(self.area)
#            self.A[i] = a
#            self.B[i] = b
#
#        buffer_shape = (self.nsensels, self.res_size)
#        self.neig_esim_score = np.zeros(buffer_shape, 'float32')
#        
#        # initialize a buffer of size NxA
#        self.buffer_NA = np.zeros(buffer_shape, 'float32') 
#
#    def display(self, report):
#        pass
#    
#    def show_areas(self, report, d):
#        Y, X = np.meshgrid(range(d.shape[1]), range(d.shape[0]))
#        dx_local = d[:, :, 0] - X
#        dy_local = d[:, :, 1] - Y 
#        f = report.figure(cols=3)
#        
#        with f.plot('search_box', caption='All search boxes') as pylab:
#            pylab.hold(True)
#            
#            nbx = np.max(dx_local) - np.min(dx_local)
#            nby = np.max(dy_local) - np.min(dy_local)
##            hist_range = [[np.min(dx_local), np.max(dx_local)],
##                     [np.min(dy_local), np.max(dy_local)]] 
#            hist_range = [[-(self.area[0] + 2.0) / 2, (self.area[0] + 2.0) / 2],
#                          [-(self.area[1] + 2.0) / 2, (self.area[1] + 2.0) / 2]]
#            
#            Hc, xe, ye = np.histogram2d(dx_local.flatten(), dy_local.flatten(),
#                                        range=hist_range, bins=(nbx, nby))
##            pdb.set_trace()
#            Yh, Xh = np.meshgrid(ye[:-1] + .5 * (ye[1] - ye[0]),
#                                 xe[:-1] + .5 * (xe[1] - xe[0]))            
#            pylab.contourf(Xh, Yh, Hc, cmap=pylab.get_cmap('Blues'), alpha=1)
#            
#            
#            Hc, xe, ye = np.histogram2d(self.A[:, 0], self.A[:, 1],
#                                        range=hist_range, bins=(nbx, nby))
#            Yh, Xh = np.meshgrid(ye[:-1] + 0.5 * (ye[1] - ye[0]),
#                                 xe[:-1] + 0.5 * (xe[1] - xe[0]))            
#            pylab.contourf(Xh, Yh, Hc, cmap=pylab.get_cmap('Reds'), alpha=0.5)
#            
##            for i in range(self.nsensels):
##                a = self.A[i]
##                b = self.B[i]
##                
##                boxx = np.array([a[0], a[0] + b[0], a[0] + b[0], a[0], a[0]])
##                boxy = np.array([a[1], a[1], a[1] + b[1], a[1] + b[1], a[1]])
##            
##                pylab.plot(boxx, boxy)
#            
#            pylab.plot(dx_local.reshape(dx_local.size),
#                       dy_local.reshape(dy_local.size),
#                       linestyle='none', marker='.')
#
#
#    def summarize(self):
#        ''' 
#            Find maximum likelihood estimate for diffeomorphism looking 
#            at each pixel singularly. 
#            
#            Returns a Diffeomorphism2D.
#        '''
#        dd_local = np.zeros((self.shape[0], self.shape[1], 2))
#        dd = np.zeros((self.shape[0], self.shape[1], 2))
#        for i in range(self.nsensels):
#            best = np.argmin(self.neig_esim_score[i])
#            best_coord_local = self.A[i] + self.interpolator.get_local_coord(self.B[i],
#                                                                             self.res,
#                                                                             best)
#            best_coord = self.C[i] + best_coord_local
#            
#            dd_local[tuple(self.C[i])] = best_coord_local
#            dd[tuple(self.C[i])] = best_coord
#        cert_flat = (self.neig_esim_score - np.min(self.neig_esim_score)) / (np.max(self.neig_esim_score) - np.min(self.neig_esim_score))
#
#        cert = np.min(cert_flat, axis=1).reshape(self.shape)
#        
#        return Diffeomorphism2DContinuous(dd, cert)
#    
#    def publish(self, pub):
#        pass
#    
#    def merge(self, other):
#        pass
#    
#def compare():
#    pass
