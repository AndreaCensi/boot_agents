'''
This estimator is an experiment to look at the statistics of the image to 
estimate what the error function should look like and to use that for finding 
the diffeomorphism
'''

from . import logger
from .. import Diffeomorphism2D
from PIL import Image  # @UnresolvedImport
from boot_agents.diffeo.diffeo_basic import diffeo_identity
from boot_agents.diffeo.learning.diffeo_estimator_fast import (
    DiffeomorphismEstimatorFaster)
from matplotlib import cm
from scipy.optimize.minpack import leastsq
import itertools
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

Order = 'order'
Similarity = 'sim'
Cont = 'quad'
InferenceMethods = [Similarity]
    
class DiffeomorphismEstimatorFasterStatistics(DiffeomorphismEstimatorFaster):
    def _update_scalar(self, y0, y1):
        # Call the supercalls method
        DiffeomorphismEstimatorFaster._update_scalar(self, y0, y1)

        Y0 = self.flat_structure.values2unrolledneighbors(y0, out=self.buffer_NA)
        y0_flat = self.flat_structure.flattening.rect2flat(y0)
        
        for k in xrange(self.nsensels):
            self.neig_y_stats[k, :] += np.abs(y0_flat[k] - Y0[k, :])
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
            
        
    def init_structures(self, y):
        # Inherit from super class
        DiffeomorphismEstimatorFaster.init_structures(self, y)
        
        buffer_shape = (self.nsensels, self.area_size)
        self.neig_y_stats = np.zeros(buffer_shape, 'float32')
            
        # Inherited code below
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


    def fit_variation_parameters(self, data):
        area = np.array(self.area)
        X, Y = np.meshgrid(range(-area[1] / 2 + 1, area[1] / 2 + 1), range(-area[0] / 2 + 1, area[0] / 2 + 1))
        
        def residual(A):
            res = (variation_function([0, 0], A, (X, Y)).flatten() - data)
#            res[area/2]
            return res
        
#        def jacobian(A):
#            return variation_jacobian_A([0, 0], A, (X, Y))
        
#        pdb.set_trace()
#        result = leastsq(residual, [1, 1], Dfun=jacobian)
        result = leastsq(residual, [1, 1])
        if result[1] in [1, 2, 3, 4]:
            return result[0]
        else:
            return [0, 0]
    
    def fit_P0(self, data, A):
        area = np.array(self.area)
        X, Y = np.meshgrid(range(-area[1] / 2 + 1, area[1] / 2 + 1), range(-area[0] / 2 + 1, area[0] / 2 + 1))
        
        def residual(P):
            return (variation_function(P, A, (X, Y)).flatten() - data) 
#        pdb.set_trace()
        result = leastsq(residual, [0, 0])
        
        if result[1] in [1, 2, 3, 4]:
            return result[0]
        else:
            return [0, 0]


    def summarize(self):
        ''' 
            Find maximum likelihood estimate for diffeomorphism looking 
            at each pixel singularly. 
            
            Returns a Diffeomorphism2D.
        '''
        certainty = np.zeros(self.shape, dtype='float32')
        certainty[:] = np.nan
        
        area = np.array(self.area)
        c_it = itertools.product(range(-area[0] / 2 + 1, area[0] / 2 + 1), range(-area[1] / 2 + 1, area[1] / 2 + 1))
        dist_n = []
        dist = []
        for c in c_it:
            dist.append(c)
            dist_n.append(la.norm(c))
        dist = np.array(dist)
        dist_n = np.array(dist_n)
        
        dd = diffeo_identity(self.shape)
        dd[:] = -1
        
        X, Y = np.meshgrid(range(-area[1] / 2 + 1, area[1] / 2 + 1), range(-area[0] / 2 + 1, area[0] / 2 + 1))
        for i in range(self.nsensels):
            best = np.argmin(self.neig_esim_score[i, :])     
            if True:
#            if i > 40 * 6 + 15 and i < 40 * (30 - 6) - 15:
#            if i in range(80 * 30 + 40, 80 * 30 + 50):

#                pattern = self.neig_y_stats[i, :].reshape(self.area)
#                border = np.array(pattern.shape) / 4 + [1, 1]
#                sub_pattern = pattern[border[0]:-border[0], border[1]:-border[1]]

                qsize = (5, 5)
                res = 25
                
                image = self.neig_esim_score[i, :].reshape(self.area)
                image0 = self.neig_y_stats[i, :].reshape(self.area)
                
                quad0 = get_quad(image0, qsize, (0., 0., 0.), res)
                quad_score = -np.ones(9 * 9 * 9)
#                dclist = itertools.product(np.linspace(-9, 9, 7 * 5), np.linspace(-9, 9, 7 * 5), [0])
                dclist = itertools.product(np.linspace(-4, 4, 9), np.linspace(-4, 4, 9), np.linspace(0, 2 * np.pi, 9))
                
                bestscore = None
                
                for qi, displ in enumerate(dclist):
                    test_quad = get_quad(image, qsize, displ, res)
                    
                    diff = np.abs(quad0 - test_quad)
                    q_norm = la.norm(diff.flatten())
                    
                    if bestscore is None:
                        bestscore = q_norm
                        bestdispl = displ
                    
                    if q_norm < bestscore:
                        bestscore = q_norm
                        bestdispl = displ
                    
#                    plt.figure()
#                    plt.imshow(test_quad, vmin=0, vmax=100)
#                    plt.savefig('out/peaks/quad' + str(qi) + '.png')
#                    logger.info('quadimsave' + str(qi))
#                    plt.colorbar()
#                    plt.clf()
#                    plt.close()
#                    
#                    plt.figure()
#                    plt.imshow(quad0, vmin=0, vmax=100)
#                    plt.savefig('out/peaks/quad0' + str(qi) + '.png')
#                    logger.info('quadimsave' + str(qi))
#                    plt.colorbar()
#                    plt.clf()
#                    plt.close()
#                    
#                    plt.figure()
#                    plt.imshow(diff, vmin=0, vmax=100)
#                    plt.title('Norm: ' + str(la.norm(diff.flatten())))
#                    plt.savefig('out/peaks/quad_diff' + str(qi) + '.png')
#                    logger.info('quadimsave' + str(qi))
#                    plt.colorbar()
#                    plt.clf()
#                    plt.close()
                    
                logger.info('bestdispl is: ' + str(bestdispl))
                
# #                pdb.set_trace()
#                A = self.fit_variation_parameters(self.neig_y_stats[i, :])
# #                A = [200, .5]
#                logger.debug(A)
#                   
#                P0 = self.fit_P0(self.neig_esim_score[i, :] - np.min(self.neig_esim_score[i, :]), A)
#                logger.info(P0)
#                
#                plt.figure()
#                plt.imshow(self.neig_y_stats[i, :].reshape(self.area), vmin=0, vmax=255)
#                plt.colorbar()
#                plt.savefig('out/peaks/y_stats' + str(i) + '.png')
#                plt.clf()
#                plt.close()
#                
#                plt.figure()
#                plt.imshow(self.neig_esim_score[i, :].reshape(self.area), vmin=0, vmax=255)
#                plt.colorbar()
#                plt.savefig('out/peaks/esim_score' + str(i) + '.png')
#                plt.clf()
#                plt.close()
                

            if self.inference_method == 'order':
                assert False
                eord_score = self.neig_eord_score[i, :]
                best = np.argmin(eord_score)
            
            if self.inference_method == 'sim':
                esim_score = self.neig_esim_score[i, :]

                
            jc = self.flat_structure.neighbor_cell(i, best)
            ic = self.flat_structure.flattening.index2cell[i]
            
            if self.inference_method == 'order':
                certain = -np.min(eord_score) / np.mean(eord_score)
                
            if self.inference_method == 'sim':
                first = np.sort(esim_score)[:10]
                certain = -(first[0] - np.mean(first[1:]))
                # certain = -np.min(esim_score) / np.mean(esim_score)
            certain = np.min(esim_score) / self.num_samples
            certain = -np.mean(esim_score) / np.min(esim_score)
            
#            dd[ic[0], ic[1], 0] = jc[0]
#            dd[ic[0], ic[1], 1] = jc[1]
            dd[ic[0], ic[1], 0] = ic[0] + int(bestdispl[0])
            dd[ic[0], ic[1], 1] = ic[1] + int(bestdispl[1])
            certainty[ic[0], ic[1]] = certain
        
        certainty = certainty - certainty.min()
        vmax = certainty.max()
        if vmax > 0:
            certainty *= (1.0 / vmax)
            
        return Diffeomorphism2D(dd, certainty)

    def merge(self, other):
        """ Merges the values obtained by "other" with ours. """
        logger.info('merging %s + %s' % (self.num_samples, other.num_samples)) 
        self.num_samples += other.num_samples
        self.neig_esim_score += other.neig_esim_score
        
        if hasattr(self, 'neig_eord_score') and hasattr(other, 'neig_eord_score'):
            self.neig_eord_score += other.neig_eord_score
        else:
            logger.warn(('neig_eord_score is missing in at least one estimator.' + 
            'Merged estimator will not have neig_eord_score.'))
            
        if hasattr(self, 'neig_esimmin_score') and hasattr(other, 'neig_esimmin_score'):
            self.neig_esimmin_score = other.neig_esimmin_score
        else:
            logger.warn(('neig_esimmin_score is missing in at least one estimator.' + 
            'Merged estimator will not have neig_eord_score.'))

        # TODO:
        if hasattr(self, 'self.neig_y_stats') and hasattr(other, 'self.neig_y_stats'):
            self.self.neig_y_stats += other.self.neig_y_stats

    def plot3d(self, data, mesh=None, figure=None, alpha=0.5, linewidth=0.5, contourf=True):
        if figure is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = figure
            fig.hold(True)
            
        area = data.shape
        if mesh is None:
            X, Y = np.meshgrid(range(-area[1] / 2 + 1, area[1] / 2 + 1), range(-area[0] / 2 + 1, area[0] / 2 + 1))
        else:
            X, Y = mesh
            
        cmap = cm.jet  # @UnusedVariable @UndefinedVariable
        surf = ax.plot_surface(X, Y, data, linewidth=linewidth, alpha=alpha,
                               rstride=1, cstride=1, antialiased=True, cmap=cmap)
        if contourf:            
            cset = ax.contourf(X, Y, data, zdir='z', offset=0, cmap=cmap)
            cset = ax.contourf(X, Y, data, zdir='x', offset=(-area[1] / 2), cmap=cmap)
            cset = ax.contourf(X, Y, data, zdir='y', offset=area[0] / 2, cmap=cmap)
            cset = ax.contour(X, Y, data, zdir='x', offset=area[1] / 2, colors='k', alpha=0.5)
            cset = ax.contour(X, Y, data, zdir='y', offset=(-area[0] / 2), colors='k', alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlim3d([0, np.max(data) * 2])
        return (fig, ax)
        
def variation_function(P0, A, point):
    if len(point) == 2:
        X = point[0]
        Y = point[1]
    elif len(np.array(point.shape)) == 3:
        X = point[:, :, 0]
        Y = point[:, :, 1]
    else:
        logger.error('Uninterpreted input to variation_function')
    dX = X - P0[0]
    dY = Y - P0[1]
    
    pdist = np.sqrt(dX ** 2 + dY ** 2)
    return np.array(A[0] * (1 - np.exp(-A[1] * pdist)))


def variation_jacobian_A(P0, A, point):
    if len(point) == 2:
        X = point[0]
        Y = point[1]
    elif len(np.array(point.shape)) == 3:
        X = point[:, :, 0]
        Y = point[:, :, 1]
    else:
        logger.error('Uninterpreted input to variation_function')
    dX = X - P0[0]
    dY = Y - P0[1]
    
    pdist = np.sqrt(dX ** 2 + dY ** 2)
    A = np.array(A)
    return np.array([(1 - np.exp(-A[1] * pdist)), A[0] * pdist * (np.exp(-A[1] * pdist))]).T


# blstsquare(residual, y0, bounds)

def clip_center(image, new_shape):
    shape = np.array(image.shape)
    center = shape / 2
    offset = np.array(new_shape) / 2
    image[center[0] - offset[0]:center[0] + offset[0] + 1, center[1] - offset[1]:center[1] + offset[1] + 1]
def get_quad(image, q_shape, displ, resolution):
    shape = np.array(image.shape)
    center = shape / 2
    qcenter = center + displ[:2]
    t = float(displ[2])
    
    pil_img = Image.fromarray(image)
    pil_img_fine = pil_img.resize(np.array(pil_img.size) * resolution)
    
#    w, h = pil_img_fine.size
#    w, h = [15, 15]
    w, h = np.flipud(q_shape)
    
    A = np.mat([[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1],
                [h, 0, 1, 0, 0, 0], [0, 0, 0, h, 0, 1],
                [h, w, 1, 0, 0, 0], [0, 0, 0, h, w, 1],
                [0, w, 1, 0, 0, 0], [0, 0, 0, 0, w, 1]])
    
    w, h = np.array(q_shape) / 2
    R = np.mat([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    c1 = (np.flipud(qcenter + np.array((R * np.mat([-w, -h]).T).T)[0]) * resolution).tolist()
    c2 = (np.flipud(qcenter + np.array((R * np.mat([-w, h]).T).T)[0]) * resolution).tolist()
    c3 = (np.flipud(qcenter + np.array((R * np.mat([w, h]).T).T)[0]) * resolution).tolist()
    c4 = (np.flipud(qcenter + np.array((R * np.mat([w, -h]).T).T)[0]) * resolution).tolist()
    
#    A = np.mat([[c1[0], c1[1], 1, 0, 0, 0], [0, 0, 0, c1[0], c1[1], 1],
#                [c2[0], c2[1], 1, 0, 0, 0], [0, 0, 0, c2[0], c2[1], 1],
#                [c3[0], c3[1], 1, 0, 0, 0], [0, 0, 0, c3[0], c3[1], 1],
#                [c4[0], c4[1], 1, 0, 0, 0], [0, 0, 0, c4[0], c4[1], 1]])
    
    qdata = np.mat(c1 + c2 + c3 + c4).T
#    qdata = np.mat([0, 0, 0, 2 * h, 2 * w, 2 * h, 2 * w, 0]).T
    
#    pdb.set_trace()
    
    adata = np.array((A.T * A).I * A.T * qdata)
    
    quad_img = pil_img_fine.transform(tuple(np.flipud(q_shape)), Image.AFFINE, adata)
#    quad_img = quad_img.resize(np.flipud(q_shape))
    return np.array(quad_img)
    
