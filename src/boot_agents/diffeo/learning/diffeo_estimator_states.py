'''
This is a experimental estimator for learning with states
'''

#from . import logger
#from .. import (diffeomorphism_to_rgb, contract, np, diffeo_to_rgb_norm,
#    diffeo_to_rgb_angle, angle_legend, diffeo_to_rgb_curv, diffeo_text_stats,
#    Diffeomorphism2D)
#from boot_agents.diffeo.diffeo_basic import diffeo_identity
#from boot_agents.diffeo.plumbing import flat_structure_cache, togrid, add_border
#from boot_agents.utils.nonparametric import scale_score
#from reprep.plot_utils import plot_vertical_line
#import time
#
#Order = 'order'
#Similarity = 'sim'
#Cont = 'quad'
#InferenceMethods = [Order, Similarity, Cont]
#    
#class DiffeomorphismEstimatorState():
#    ''' Learns a diffeomorphism dependent on a state x between two 2D fields. '''
#    
#    @contract(max_displ='seq[2](>0,<1)', inference_method='str')
#    def __init__(self, max_displ, inference_method):
#        """ 
#            :param max_displ: Maximum displacement  
#            :param inference_method: order, sim
#        """
#        self.shape = None
#        self.max_displ = np.array(max_displ)
#        self.last_y0 = None
#        self.last_y1 = None
#        self.inference_method = inference_method
#        
#        if self.inference_method not in InferenceMethods:
#            msg = ('I need one of %s; found %s' % 
#                   (InferenceMethods, inference_method))
#            raise ValueError(msg)
#        
#        self.num_samples = 0
#
#        self.buffer_NA = None
#        
#    @contract(y0='array[MxN]', y1='array[MxN]')
#    def update(self, y0, y1, state):
#        if y0.dtype == np.uint8:
#            y0 = (y0 / 255.0).astype('float32')
#            y1 = (y1 / 255.0).astype('float32')
#        self.num_samples += 1
#
#        # init structures if not already
#        if self.shape is None:
#            self.init_structures(y0)
#
#        # check shape didn't change
#        if self.shape != y0.shape:
#            msg = 'Shape changed from %s to %s.' % (self.shape, y0.shape)
#            raise ValueError(msg)
#
#        # remember last images
#        self.last_y0 = y0
#        self.last_y1 = y1
#        
#        
#        ts = []
#        ts.append(time.time())
#        #self._update_vectorial(y0, y1)
#        ts.append(time.time())
#        self._update_scalar(y0, y1)
#        ts.append(time.time())
#        delta = np.diff(ts)
#        logger.info('Update times: vect %5.3f scal %5.3f seconds' % 
#                    (delta[0], delta[1]))
#        
#    def _update_scalar(self, y0, y1):
#        # unroll the Y1 image
#        Y1 = self.flat_structure.values2unrolledneighbors(y1, out=self.buffer_NA)
#        y0_flat = self.flat_structure.flattening.rect2flat(y0)
#        
#        order_comp = np.array(range(self.area_size), dtype='float32')
#        
#        for k in xrange(self.nsensels):
#            diff = np.abs(y0_flat[k] - Y1[k, :])
#            
#            if self.inference_method == Order:
#                order = np.argsort(diff)
#                #diff_order = scale_score(diff, kind='quicksort')
#                self.neig_eord_score[k, order] += order_comp 
#            elif self.inference_method == Similarity:
#                self.neig_esim_score[k, :] += diff
#            else:
#                assert False
#                
#            self.neig_esimmin_score[k] += np.min(diff)
#            
#            
#    def _update_vectorial(self, y0, y1):
#        Y0 = self.flat_structure.values2repeated(y0)
#        Y1 = self.flat_structure.values2unrolledneighbors(y1, out=self.buffer_NA)
#        
#        difference = np.abs(Y0 - Y1)
#        
#        if self.inference_method == Order:
#            # Yes, double argsort(). This is correct.
#            # (but slow; seee update_scalar above)
#            simorder = np.argsort(np.argsort(difference, axis=1), axis=1)
#            self.neig_eord_score += simorder
#        
#        elif self.inference_method == Similarity:
#            self.neig_esim_score += difference
#        else:
#            assert False
#                    
#        self.neig_esimmin_score += np.min(difference, axis=1)
#        
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
#
#        logger.debug(' Field Shape: %s' % str(self.shape))
#        logger.debug('    Fraction: %s' % str(self.max_displ))
#        logger.debug(' Search area: %s' % str(self.area))
#        logger.debug('Creating FlatStructure...')
#        self.flat_structure = flat_structure_cache(self.shape, self.area)
#        logger.debug('done creating')
#
#        buffer_shape = (self.nsensels, self.area_size)
#        if self.inference_method == Similarity:   
#            self.neig_esim_score = np.zeros(buffer_shape, 'float32')
#        elif self.inference_method == Order:
#            self.neig_eord_score = np.zeros(buffer_shape, 'float32')
#        else:
#            assert False
#              
#        self.neig_esimmin_score = np.zeros(self.nsensels)
#        
#        # initialize a buffer of size NxA
#        self.buffer_NA = np.zeros(buffer_shape, 'float32') 
#    
#    def display(self, report):
#        report.data('num_samples', self.num_samples)
#        f = report.figure(cols=4)
#        
#        
#        def make_best(x):
#            return x == np.min(x, axis=1)
#        
#        @contract(score='array[NxA]', returns='array[HxW],H*W=N')
#        def distance_to_border_for_best(score):
#            N, _ = score.shape
#            best = np.argmin(score, axis=1)
#            assert best.shape == (N,)
#            D = self.flat_structure.get_distances_to_area_border()
#            res = np.zeros(N)
#            for i in range(N):
#                res[i] = D[i, best[i]]
#            return self.flat_structure.flattening.flat2rect(res)
#
#        @contract(score='array[NxA]', returns='array[HxW],H*W=N')
#        def distance_from_center_for_best(score):
#            N, _ = score.shape
#            best = np.argmin(score, axis=1)
#            assert best.shape == (N,)
#            D = self.flat_structure.get_distances()
#            res = np.zeros(N)
#            for i in range(N):
#                res[i] = D[i, best[i]]
#            return self.flat_structure.flattening.flat2rect(res)
#
#        max_d = int(np.ceil(np.hypot(np.floor(self.area[0] / 2.0),
#                                     np.floor(self.area[1] / 2.0))))
#        safe_d = int(np.floor(np.min(self.area) / 2.0))
#        
#        bdist_scale = dict(min_value=0, max_value=max_d, max_color=[0, 1, 0])
#        cdist_scale = dict(min_value=0, max_value=max_d, max_color=[1, 0, 0])
#        bins = range(max_d + 2)
#        
#        def plot_safe(pylab):
#            plot_vertical_line(pylab, safe_d, 'g--')
#            plot_vertical_line(pylab, max_d, 'r--')
#            
#        if self.inference_method == Order:
#            eord = self.make_grid(self.neig_eord_score)
#            report.data('neig_eord_score_rect', eord).display('scale').add_to(f, caption='order')
#            eord_bdist = distance_to_border_for_best(self.neig_eord_score)
#            eord_cdist = distance_from_center_for_best(self.neig_eord_score)
#            report.data('eord_bdist', eord_bdist).display('scale', **bdist_scale).add_to(f, caption='eord_bdist')
#            report.data('eord_cdist', eord_cdist).display('scale', **cdist_scale).add_to(f, caption='eord_cdist')
#            with f.plot('eord_bdist_hist') as pylab:
#                pylab.hist(eord_bdist.flat, bins)
#            with f.plot('eord_cdist_hist') as pylab:
#                pylab.hist(eord_cdist.flat, bins)
#                plot_safe(pylab)
#
#        if self.inference_method == Similarity:
#            esim = self.make_grid(self.neig_esim_score)
#            report.data('neig_esim_score_rect', esim).display('scale').add_to(f, caption='sim')
#            esim_bdist = distance_to_border_for_best(self.neig_esim_score)
#            esim_cdist = distance_from_center_for_best(self.neig_esim_score)
#            report.data('esim_bdist', esim_bdist).display('scale', **bdist_scale).add_to(f, caption='esim_bdist')
#            report.data('esim_cdist', esim_cdist).display('scale', **cdist_scale).add_to(f, caption='esim_cdist')
#        
#            with f.plot('esim_bdist_hist') as pylab:
#                pylab.hist(esim_bdist.flat, bins)
#            with f.plot('esim_cdist_hist') as pylab:
#                pylab.hist(esim_cdist.flat, bins)
#                plot_safe(pylab)
#    
#        
#        
#    @contract(score='array[NxA]', returns='array[UxV]') # ,U*V=N*A') not with border
#    def make_grid(self, score):
#        fourd = self.flat_structure.unrolled2multidim(score) # HxWxXxY
#        return togrid(add_border(fourd))
#        
#    def summarize(self):
#        ''' 
#            Find maximum likelihood estimate for diffeomorphism looking 
#            at each pixel singularly. 
#            
#            Returns a Diffeomorphism2D.
#        '''
#        certainty = np.zeros(self.shape, dtype='float32')
#        certainty[:] = np.nan
#        
#        dd = diffeo_identity(self.shape)
#        dd[:] = -1
#        for i in range(self.nsensels):
#            
#            if self.inference_method == 'order':
#                eord_score = self.neig_eord_score[i, :]
#                best = np.argmin(eord_score)
#            
#            if self.inference_method == 'sim':
#                esim_score = self.neig_esim_score[i, :]
#                best = np.argmin(esim_score)
#                
#            jc = self.flat_structure.neighbor_cell(i, best)
#            ic = self.flat_structure.flattening.index2cell[i]
#            
#            if self.inference_method == 'order':
#                certain = -np.min(eord_score) / np.mean(eord_score)
#                
#            if self.inference_method == 'sim':
#                first = np.sort(esim_score)[:10]
#                certain = -(first[0] - np.mean(first[1:]))
#                #certain = -np.min(esim_score) / np.mean(esim_score)
##            certain = np.min(esim_score) / self.num_samples
##            certain = -np.mean(esim_score) / np.min(esim_score)
#            
#            dd[ic[0], ic[1], 0] = jc[0]
#            dd[ic[0], ic[1], 1] = jc[1]
#            certainty[ic[0], ic[1]] = certain
#        
#        certainty = certainty - certainty.min()
#        vmax = certainty.max()
#        if vmax > 0:
#            certainty *= (1.0 / vmax)
#            
#        return Diffeomorphism2D(dd, certainty)
#    
#    def publish(self, pub):
#        diffeo = self.summarize() 
#        
#        pub.array_as_image('mle', diffeomorphism_to_rgb(diffeo.d))
#        pub.array_as_image('angle', diffeo_to_rgb_angle(diffeo.d))
#        pub.array_as_image('norm', diffeo_to_rgb_norm(diffeo.d, max_value=10))
#        pub.array_as_image('curv', diffeo_to_rgb_curv(diffeo.d))
#        pub.array_as_image('variance', diffeo.variance, filter='scale')
#
#        pub.text('num_samples', self.num_samples)
#        pub.text('statistics', diffeo_text_stats(diffeo.d))
#        pub.array_as_image('legend', angle_legend((50, 50)))
#
#        n = 20
#        M = None
#        for i in range(n): #@UnusedVariable
#            c = self.flattening.random_coords()
#            Mc = self.get_similarity(c)
#            if M is None:
#                M = np.zeros(Mc.shape)
#                M.fill(np.nan)
#
#            ok = np.isfinite(Mc)
#            Mmax = np.nanmax(Mc)
#            if Mmax < 0:
#                Mc = -Mc
#                Mmax = -Mmax
#            if Mmax > 0:
#                M[ok] = Mc[ok] / Mmax
#
#        pub.array_as_image('coords', M, filter='scale')
#
#        if self.last_y0 is not None:
#            y0 = self.last_y0
#            y1 = self.last_y1
#            none = np.logical_and(y0 == 0, y1 == 0)
#            x = y0 - y1
#            x[none] = np.nan
#            pub.array_as_image('motion', x, filter='posneg')
#
#    def merge(self, other):
#        """ Merges the values obtained by "other" with ours. """
#        logger.info('merging %s + %s' % (self.num_samples, other.num_samples)) 
#        self.num_samples += other.num_samples
#        self.neig_esim_score += other.neig_esim_score
#        self.neig_eord_score += other.neig_eord_score
#        self.neig_esimmin_score = other.neig_esimmin_score
#
#
#        
