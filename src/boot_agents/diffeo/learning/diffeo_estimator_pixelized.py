'''
This version only considers learning for specified pixels/sensels. It considers 
all commands and states for those pixels and creates a model for interpolation 
of other commands in between the learned ones. This estimator is more similar 
to a learner and needs to be feed with log items for all commands.     

This estimator is still in a experimental stage

Created on Dec 3, 2012

@author: adam
'''

from . import logger
from .. import (diffeomorphism_to_rgb, contract, np, diffeo_to_rgb_norm,
                diffeo_to_rgb_angle, angle_legend, diffeo_to_rgb_curv,
                diffeo_text_stats, Diffeomorphism2D)
from boot_agents.diffeo.diffeo_basic import diffeo_identity
from boot_agents.diffeo.plumbing import flat_structure_cache, togrid, add_border
from matplotlib.cm import get_cmap
from reprep.plot_utils import plot_vertical_line
from sklearn.gaussian_process.gaussian_process import GaussianProcess
import pdb
Order = 'order'
Similarity = 'sim'
Cont = 'quad'
InferenceMethods = [Similarity]

class DiffeomorphismEstimatorPixelized():
    ''' 
    Learns a diffeomorphism between two 2D fields. 
    
    Only consider pixels in registred list
    '''
    
    @contract(max_displ='seq[2](>0,<1)', inference_method='str')
    def __init__(self, max_displ, inference_method, sensels):
        """ 
            :param max_displ: Maximum displacement  
            :param inference_method: order, sim
        """
        self.shape = None
        self.max_displ = np.array(max_displ)
        self.last_y0 = None
        self.last_y1 = None
        self.inference_method = inference_method
        
        if self.inference_method not in InferenceMethods:
            msg = ('I need one of %s; found %s' % 
                   (InferenceMethods, inference_method))
            raise ValueError(msg)
              
        self.esim_dict = {}
        
        self.num_samples = 0
        
        self.sensels = sensels
        self.sensel_estimators = {}
        
        self.buffer_NA = None
        
#        self.output_cmd_state = [(256, 0, 0, 100), (256, 0, 0, 200),
#                                 (-256, 0, 0, 100), (-256, 0, 0, 200),
#                                 (0, 256, 0, 100), (0, 256, 0, 200),
#                                 (0, -256, 0, 0, 100), (0, -256, 0, 200)]
        self.output_cmd_state = [(256, 0, 0, 100), (-256, 0, 0, 100), (0, 256, 0, 100)]
        
        
        
    @contract(y0='array[MxN]', y1='array[MxN]')
    def update(self, y0, y1, u0, x0):
        '''
        :param y0: initial image
        :param u0: command
        :param y1: after image
        :param x0: state
        '''
        if len(y0.shape) == 3:
            for i in range(y0.shape[2]):
                self.update(y0[:, :, i], y1[:, :, i], u0, x0)
            return 0
        
        logger.info('Pixelized update command: ' + str(u0) + ' state: ' + str(x0))
        
        if y0.dtype == np.uint8:
            try:
                y0 = (y0 / 255.0).astype('float32')
                y1 = (y1 / 255.0).astype('float32')
            except:
                pdb.set_trace()
        self.num_samples += 1

        # init structures if not already
        if self.shape is None:
            self.init_structures(y0, u0, x0)

        # check shape didn't change
        if self.shape != y0.shape:
            msg = 'Shape changed from %s to %s.' % (self.shape, y0.shape)
            raise ValueError(msg)

        # remember last images
        self.last_y0 = y0
        self.last_y1 = y1
        
        Y0 = self.flat_structure.values2unrolledneighbors(y0, out=self.buffer_NA)
        Y1 = self.flat_structure.values2repeated(y1)
        
        for s in self.sensels:
            if not self.sensel_estimators.has_key(s):
                self.sensel_estimators[s] = SenselEstimator(s, self)
                
            self.sensel_estimators[s].update(Y0[s], Y1[s], u0, x0)
            
        
    def init_structures(self, y, u, x):
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

        
        self.flat_structure = flat_structure_cache(self.shape, tuple(self.area))
        
        
        buffer_shape = (self.nsensels, self.area_size)
        # initialize a buffer of size NxA
        self.buffer_NA = np.zeros(buffer_shape, 'float32')
    
    def display(self, report, sufix=''):
        report.data('num_sensels' + sufix, len(self.sensels))
        report.data('sensels' + sufix, self.sensels)
        
        return 0
        report.data('num_samples', self.num_samples)
#        f = report.figure(cols=4)
        for key in self.sensel_estimators.keys():
            self.sensel_estimators[key].summarize()
            self.sensel_estimators[key].display(report)
        
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
        
#        bdist_scale = dict(min_value=0, max_value=max_d, max_color=[0, 1, 0])
#        cdist_scale = dict(min_value=0, max_value=max_d, max_color=[1, 0, 0])
#        bins = range(max_d + 2)
        
        def plot_safe(pylab):
            plot_vertical_line(pylab, safe_d, 'g--')
            plot_vertical_line(pylab, max_d, 'r--')
        
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
    
        
        
    @contract(score='array[NxA]', returns='array[UxV]') # ,U*V=N*A') not with border
    def make_grid(self, score):
        fourd = self.flat_structure.unrolled2multidim(score) # HxWxXxY
        return togrid(add_border(fourd))
        
    def summarize(self, command, state):
        ''' 
            Find maximum likelihood estimate for diffeomorphism looking 
            at each pixel singularly. 
            
            Returns a Diffeomorphism2D.
        '''
#        for cmd_state in self.output_cmd_state:
        
        dd = diffeo_identity(self.shape)
#        dd_g = diffeo_identity(self.shape)
        certainty = np.zeros(self.shape, dtype='float32')
        coords_flat = []
        dmap_flat = []
        for key in self.sensel_estimators.keys():
            se = self.sensel_estimators[key]
            se.summarize()
            c = tuple(se.coord)
            logger.info('summarized sensel at ' + str(c))
            
            coords_flat.append(c)
            
#            pdb.set_trace()
            dmap, cert = se.get_map(command, state)
            
            dmap_flat.append(dmap)
            
            dd[c] = dd[c] + dmap
            
            certainty[c] = np.exp(-cert)
            
#        coords_flat = np.array(coords_flat)
#        dmap_flat = np.array(dmap_flat)
#        
#        # Instanciate a Gaussian Process model
#        diffx = GaussianProcess(corr='cubic', nugget=np.ones(len(coords_flat[:, 0])))
#        diffy = GaussianProcess(corr='cubic', nugget=np.ones(len(coords_flat[:, 0])))
#        
#        diffx.fit(coords_flat, dmap_flat[:, 0])
#        diffy.fit(coords_flat, dmap_flat[:, 1])
#        
#        domain = np.array(list(itertools.product(range(self.shape[0]),
#                                                 range(self.shape[1]))))
#        
#        x_pred, _ = diffx.predict(domain, eval_MSE=True)
#        y_pred, _ = diffy.predict(domain, eval_MSE=True)
#
#        
#        dd_g[:, :, 0] = dd_g[:, :, 0] + x_pred.reshape(self.shape)
#        dd_g[:, :, 1] = dd_g[:, :, 1] + y_pred.reshape(self.shape)
#        pdb.set_trace()
        return Diffeomorphism2D(dd, certainty)
        
    
    def publish(self, pub):
        diffeo = self.summarize() 
        
        pub.array_as_image('mle', diffeomorphism_to_rgb(diffeo.d))
        pub.array_as_image('angle', diffeo_to_rgb_angle(diffeo.d))
        pub.array_as_image('norm', diffeo_to_rgb_norm(diffeo.d, max_value=10))
        pub.array_as_image('curv', diffeo_to_rgb_curv(diffeo.d))
        pub.array_as_image('variance', diffeo.variance, filter='scale')

        pub.text('num_samples', self.num_samples)
        pub.text('statistics', diffeo_text_stats(diffeo.d))
        pub.array_as_image('legend', angle_legend((50, 50)))

        n = 20
        M = None
        for i in range(n): #@UnusedVariable
            c = self.flattening.random_coords()
            Mc = self.get_similarity(c)
            if M is None:
                M = np.zeros(Mc.shape)
                M.fill(np.nan)

            ok = np.isfinite(Mc)
            Mmax = np.nanmax(Mc)
            if Mmax < 0:
                Mc = -Mc
                Mmax = -Mmax
            if Mmax > 0:
                M[ok] = Mc[ok] / Mmax

        pub.array_as_image('coords', M, filter='scale')

        if self.last_y0 is not None:
            y0 = self.last_y0
            y1 = self.last_y1
            none = np.logical_and(y0 == 0, y1 == 0)
            x = y0 - y1
            x[none] = np.nan
            pub.array_as_image('motion', x, filter='posneg')

    def merge(self, other):
        """ Merges the values obtained by "other" with ours. """
#        pdb.set_trace()
        self.sensel_estimators.update(other.sensel_estimators)
        
class SenselEstimator():
    def __init__(self, index, parent):
        self.esim = {}
        self.parent = parent
        self.index = index
        self.coord = self.parent.flat_structure.flattening.index2cell[index]
        self.num_samples = 0
        self.summarized = False
        
    def update(self, Y0, Y1, U0, X0):
        self.summarized = False
        difference = np.abs(Y0 - Y1) ** 2
        
        if not self.esim.has_key((U0, X0)):
#            pdb.set_trace()
            self.esim[(U0, X0)] = difference
            logger.info('sensel estimator for (' + str((U0, X0)) + ') initiated')
        else:
            self.esim[(U0, X0)] += difference
        self.num_samples += 1
        
    def get_map(self, command, state):
        # Assert self is summarized
        if not self.summarized:
            self.summarize()
        
        # Join command and state
#        pdb.set_trace()
        cmd_state = np.array(tuple(command) + tuple(state))
        
        # Predict sensel map (diffeo)
        x_pred, MSEx = self.gpx.predict(cmd_state, eval_MSE=True)
        y_pred, MSEy = self.gpy.predict(cmd_state, eval_MSE=True)
        
        return (float(x_pred), float(y_pred)), float(MSEx + MSEy)
            
    def summarize(self):
        d = {}
        N = len(self.esim.keys())
        X = np.zeros((N, 4))
        Y = np.zeros((N, 2))
        V = np.zeros((N, 1))
        for k, key in enumerate(self.esim.keys()):
            logger.info('Summarizing key: ' + str(key))
            best = np.argmin(self.esim[key])
            logger.info('best index:      %g' % best)
            i = self.index
            jc = self.parent.flat_structure.neighbor_cell(i, best)
            
            
            certain = np.min(self.esim[key]) / self.num_samples
            d_local = jc - self.coord
            
            logger.info('best coordinate  ' + str(d_local))
            d[key] = {'local':d_local, 'map':jc, 'certain': certain}
            
            u0, x0 = key
            X[k, :] = np.array(tuple(u0) + (x0,))
            Y[k, :] = d_local
            V[k] = certain 
            
        self.d = d
        
        # Instanciate a Gaussian Process model
        self.gpx = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, \
                              random_start=100)
        self.gpy = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, \
                              random_start=100)
        
        self.gpx.fit(X, Y[:, 0])
        self.gpy.fit(X, Y[:, 1])
        
        self.summarized = True
        
        
    def display(self, report):
        return 0
        res = np.array((5, 5))
        f = report.figure(cols=5)
        C0, C1 = np.meshgrid(np.linspace(-256, 256, res[0]), np.linspace(-256, 256, res[1]))
        C0fine, C1fine = np.meshgrid(np.linspace(-256 - 50, 256 + 50, res[0] * 10), np.linspace(-256 - 50, 256 + 50, res[1] * 10))
#        zooms = [100, 150, 200]
#        for zoom in zooms:
        for zoom in []:
            evalset = np.zeros((C0.size, 4))
            for i, c0 in enumerate(C0.flatten()):
                evalset[i, :] = np.array([c0, C1.flatten()[i], 0, zoom])
            
            eval_fine = np.zeros((C0fine.size, 4))
            for i, c0 in enumerate(C0fine.flatten()):
                eval_fine[i, :] = np.array([c0, C1fine.flatten()[i], 0, zoom])
            
            x_pred, _ = self.gpx.predict(evalset, eval_MSE=True)
            y_pred, _ = self.gpy.predict(evalset, eval_MSE=True)
            
            _, MSEx = self.gpx.predict(eval_fine, eval_MSE=True)
            _, MSEy = self.gpy.predict(eval_fine, eval_MSE=True)
            
            with f.plot('pix' + str(self.index) + 'esim_field_zoom' + str(zoom)) as pylab:
                pylab.hold(True)
                
#                cmap = colors.Colormap('jet')
                pylab.contourf(C0fine.reshape(res * 10), C1fine.reshape(res * 10),
                               (MSEx + MSEy).reshape(res * 10),
                               cmap=get_cmap('RdYlGn_r'), alpha=0.3)
#                pdb.set_trace()
                pylab.quiver(C0.reshape(res), C1.reshape(res),
                             y_pred.reshape(res) * 10, x_pred.reshape(res) * 10,
                             scale_units='x', scale=1)
                logger.info('pix' + str(self.index) + 'esim_field_zoom' + str(zoom) + ' rendered')
