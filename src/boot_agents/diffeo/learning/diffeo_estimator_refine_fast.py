'''
This estimator works with a refining learner. It resizes the whole image at 
once. 
'''
from . import logger
from .. import (contract, np)
from PIL import Image  # @UnresolvedImport
from boot_agents.diffeo import coords_iterate
from boot_agents.diffeo.diffeomorphism2d_continuous import Diffeomorphism2DContinuous
from boot_agents.diffeo.plumbing import togrid, add_border
from matplotlib.ticker import MultipleLocator
import itertools
import time
import pdb

# Methods for resizing the image
REFINE_FAST_BILINEAR = 'fast-bilinear'
REFINE_FAST_BICUBIC = 'fast-bicubic'
REFINE_FAST_ANTIALIAS = 'fast-antialias'

class DiffeomorphismEstimatorRefineFast():
    '''
    
    '''
    @contract(max_displ='seq[2](>0,<1)')
    def __init__(self, max_displ, refine_method, resolution, refine_factor, update_uncertainty=False, info={}):
        '''
        
        :param max_displ:          Fraction of the search area
        :param refine_method:      Method to use for resizing the image
        :param resolution:         Resolution of the search_grid
        :param refine_factor:      The ratio which the search area is resized  
        :param update_uncertainty: If true, the uncertainties are 
                                   calculated with new method
        '''
        self.shape = None
        self.max_displ = np.array(max_displ)
        self.last_y0 = None
        self.last_y1 = None
        self.grid_shape = tuple(resolution)
        self.refine_factor = refine_factor
        self.y_gradx = None
        self.y_grady = None
        self.info = info
        
            
        # Fast methods
        if refine_method == REFINE_FAST_BILINEAR:
            self.intp_method = Image.BILINEAR
            
        elif refine_method == REFINE_FAST_BICUBIC:
            self.intp_method = Image.BICUBIC
            
        elif refine_method == REFINE_FAST_ANTIALIAS:
            self.intp_method = Image.ANTIALIAS
            
        else:
            assert False
            
        
        self.num_refined = 0
        self.num_samples = 0
        self.buffer_NA = None
        
    def set_search_areas(self, areas_position, area_shape):
        self.area_positions_coarse = areas_position
        self.area = tuple(area_shape)

    def calculate_areas(self, diffeo, nrefine):
        area_positions_coarse = np.zeros((self.nsensels, 2))
        area = tuple(np.ceil(np.array(self.area) * 1.0 / (self.refine_factor ** nrefine)).astype('int'))
        
        if not area[0] >= 1 or not area[1] >= 1:
            print str(area)
            logger.error('something went wrong with the size of the search area')
        
        for i in range(self.nsensels):
            cx, cy = self.index2cell(i)
            s = np.array(diffeo.d[cx, cy, :]).astype('float')
            s_coarse = np.floor(s * self.grid_shape / area)  # .astype('int')
            area_pos_coarse = s_coarse - (np.array(self.grid_shape) / 2)

            area_positions_coarse[i, :] = np.floor(area_pos_coarse).astype('int')
        return (area_positions_coarse, area)
    
    def update_areas(self, diffeo, nrefine):
        self.area_positions_coarse, self.area = self.calculate_areas(diffeo, nrefine)

            
    def tic(self):
        '''
        Help function for timing calculations
        '''
        self.t0 = time.time()
        
    def toc(self):
        '''
        Help function for timing calculations
        '''
        t = time.time()
        return t - self.t0
         
    @contract(y0='array[MxN]', y1='array[MxN]')
    def update(self, y0, y1):
        # Initiate structures at the first update
        if self.shape is None:
            logger.info('Initiating structure from update()')
            self.init_structures(y0)

#        self.tic()
#        gx, gy = np.gradient(y0)
        gx = np.diff(y0, axis=0)
        gy = np.diff(y0, axis=1)
        self.y_gradx += np.abs(gx)
        self.y_grady += np.abs(gy)

        grid_shape = self.grid_shape
        shape = self.shape
        
        grid_size = grid_shape[0] * grid_shape[1]
        reduced_shape = np.array(shape) * grid_shape / np.array(self.area)
#        self.time_unn += self.toc()
        
        Ycache = {}
        def get_Yi_ref(coords):
            try:
                if not Ycache.has_key(coords):
                    Yi_ref = extract_wraparound(y0_resized, coords).reshape(grid_size)
                    Ycache[coords] = Yi_ref
                    return Yi_ref
                else:
                    return Ycache[coords]
            except ValueError:
                pdb.set_trace()
        
#        self.tic()
        PImage = Image.fromarray(y0.astype('float'))
        PImage_resized = PImage.resize(np.flipud(reduced_shape), self.intp_method)
        y0_resized = np.array(PImage_resized)
#        self.time_resize += self.toc()
        

        self.tic()
        diff = np.zeros(grid_size)
        adiff = np.zeros(grid_size)
        # About 0.06s/iteration for 40x30 ang 5x5
        for i in xrange(self.nsensels):
#        for i in self.unique_indices:
            xl = self.area_positions_coarse[i][0]
            yl = self.area_positions_coarse[i][1]
            xu = xl + grid_shape[0]
            yu = yl + grid_shape[1]
#            Yi_ref = extract_wraparound(y0_resized, ((xl, xu), (yl, yu)))
#            (xl, xu, yl, yu)
            Yi_ref = get_Yi_ref((xl, xu, yl, yu))
            c = self.index2cell_tuple(i)
            np.subtract(Yi_ref, y1[c], out=diff)
            np.fabs(diff, out=adiff)
            # diff = np.abs(Yi_ref - y1[tuple(c)])
            
            self.neig_esim_score[i] += adiff
#            self.neig_esim_score[i] += get_diff(((xl, xu), (yl, yu)))
        self.num_samples += 1
        self.time_sensels += self.toc()
        
#        logger.info('Time spent in average:')
#        logger.info('    unn:      %s' % (self.time_unn / self.num_samples))
#        logger.info('    resizing: %s' % (self.time_resize / self.num_samples))
#        logger.info('    updating: %s' % (self.time_sensels / self.num_samples))
        
    def fill_esim(self):
        for i in range(self.nsensels):
            if not i in self.unique_indices:
                xl, yl = self.area_positions_coarse[i]
                xu, yu = self.area_positions_coarse[i] + self.grid_shape
                i0 = self.unique_coords[((xl, xu), (yl, yu))]
                self.neig_esim_score[i] = self.neig_esim_score[i0]

    def init_structures(self, y):
#        # Time measure variables
#        self.time_unn = 0
#        self.time_resize = 0
        self.time_sensels = 0
        
        gradx_shape = (y.shape[0] - 1, y.shape[1])
        grady_shape = (y.shape[0], y.shape[1] - 1) 
        self.y_gradx = np.zeros(gradx_shape)
        self.y_grady = np.zeros(grady_shape)
        
        self.shape = y.shape
        # for each sensel, create an area
        if not hasattr(self, 'area'):
            self.area = np.ceil(self.max_displ * np.array(self.shape)).astype('int32')
            # ensure it's an odd number of pixels
            for i in range(2):
                if self.area[i] % 2 == 0:
                    self.area[i] += 1
            self.area = (int(self.area[0]), int(self.area[1]))
            self.area = tuple(self.area)
        
        self.nsensels = y.size
#        area_size = self.area[0] * self.area[1]
        grid_size = self.grid_shape[0] * self.grid_shape[1]
        
        
        logger.debug(' Field Shape: %s' % str(self.shape))
        logger.debug('    Fraction: %s' % str(self.max_displ))
        logger.debug(' Search area: %s' % str(self.area))

        logger.debug('done creating')
        
        
#        assert(self.num_refined == 0)
        if not hasattr(self, 'area_positions_coarse'):
#            pdb.set_trace()
            logger.debug('First time estimator is initiated, area definitions')
            
            # Start with phi estimate as identity
            dd = np.zeros((self.shape[0], self.shape[1], 2)).astype('float32')
            for i in range(self.nsensels):
                ic = self.index2cell(i)
                dd[tuple(ic)] = ic
                diffeo = Diffeomorphism2DContinuous(dd)
                
            self.update_areas(diffeo, 0)


        buffer_shape = (self.nsensels, grid_size)
        self.neig_esim_score = np.zeros(buffer_shape, 'float32')
        
        # initialize a buffer of size NxA
        self.buffer_NA = np.zeros(buffer_shape, 'float32')
        
        self.dd = np.zeros((self.shape[0], self.shape[1], 2)) 
        
        # Calculate the indices which will have unique search boxes
        self.unique_coords = {}
        for i in range(self.nsensels):
            xl, yl = self.area_positions_coarse[i]
            xu, yu = self.area_positions_coarse[i] + self.grid_shape
            
            # If the key doesn't already have an index, add it.
            if not self.unique_coords.has_key(((xl, xu), (yl, yu))):
                self.unique_coords[((xl, xu), (yl, yu))] = i
        
        self.unique_indices = self.unique_coords.values()
        
        logger.info('structure initiated')

    def summarize(self):
        ''' 
            Find maximum likelihood estimate for diffeomorphism looking 
            at each pixel singularly. 
            
            Returns a Diffeomorphism2D.
        '''
#        self.fill_esim()
        dd = np.zeros((self.shape[0], self.shape[1], 2))
        for i in range(self.nsensels):
            best = np.argmin(self.neig_esim_score[i])
            if i == 620:
                pass
#                pdb.set_trace()
            best_coord = get_original_coordinate(best, self.grid_shape, self.area, self.area_positions_coarse[i])

            ic = self.index2cell(i)
            
#            logger.info('local coordinate : ' + str(best_coord - ic))
            dd[tuple(ic)] = best_coord

#        cert_flat = (self.neig_esim_score - np.min(self.neig_esim_score)) / (np.max(self.neig_esim_score) - np.min(self.neig_esim_score))
        cert_flat = (self.neig_esim_score - np.min(self.neig_esim_score)) / self.num_samples
        cert = np.min(cert_flat, axis=1).reshape(self.shape)
        
        # Pass plot ranges data to the diffeo
        diffeo = Diffeomorphism2DContinuous(dd, cert)

        if self.info.has_key('plot_ranges'):
            diffeo.plot_ranges = self.info['plot_ranges']
            
        return diffeo
    
    def display(self, report):
        self.show_areas(report)
#        pdb.set_trace()
        
        report.data('num_samples', self.num_samples)
        report.data('grid_shape', self.grid_shape)
        report.data('area_shape', self.area)

        
        report.data('reduced_shape', np.array(self.shape) * self.grid_shape / np.array(self.area))
        f = report.figure(cols=1)
        
    
        esim = self.make_grid(self.neig_esim_score)
        
        with f.plot('esim', caption='esim') as pylab:
            pylab.imshow(esim)
            
        self.show_gradient(report)
        
    def show_gradient(self, report):
        f = report.figure(cols=3)
        
        gradx = self.y_gradx / self.num_samples
        grady = self.y_grady / self.num_samples
#        grad = np.sqrt(gradx ** 2 + grady ** 2)
        vmax = np.max((np.percentile(gradx, 90), np.percentile(grady, 90)))
        with f.plot('gradx', caption='E{gradient_x}, max value: %s' % np.max(gradx)) as pylab:
            pylab.imshow(gradx, vmin=0, vmax=vmax)
        with f.plot('grady', caption='E{gradient_y}, max value: %s' % np.max(grady)) as pylab:
            pylab.imshow(grady, vmin=0, vmax=vmax)
#        with f.plot('grad', caption='E{gradient}, max value: %s' % np.max(grad)) as pylab:
#            pylab.imshow(grad, vmin=0, vmax=np.percentile(grad, 90))
        
        with f.plot('gradxu', caption='E{gradient_x}, max value: %s' % np.max(gradx)) as pylab:
            pylab.imshow(gradx)
        with f.plot('gradyu', caption='E{gradient_y}, max value: %s' % np.max(grady)) as pylab:
            pylab.imshow(grady)
#        with f.plot('gradu', caption='E{gradient}, max value: %s' % np.max(grad)) as pylab:
#            pylab.imshow(grad)    

    def show_interp_images(self, y0, outdir='out/subim/'):
        arrays = self.interpolator.arrays
        for key, array in arrays.items():
            im = Image.fromarray(array.astype('uint8'))
            im.resize(np.array(im.size) * 10).save(outdir + 'fullimage' + str(key) + '.png')


    def show_subimages(self, y0, outdir='out/subim/'):
        i = 0
        for cy in range(y0.shape[0]):
            for cx in range(y0.shape[1]):
                sub_ai = self.interpolator.extract_around((cy, cx))
                sub_ai = (sub_ai - np.min(sub_ai)) / (np.max(sub_ai) - np.min(sub_ai)) * 255
                Image.fromarray(sub_ai.astype('uint8')).resize((300, 300)).save(outdir + 'subim' + str(i) + '.png')
                i += 1

    def show_areas(self, report):
        diffeo = self.summarize()
        
        dd = diffeo.d

        Y, X = np.meshgrid(range(self.shape[1]), range(self.shape[0]))

        dx = np.median(dd[:, :, 0] - X)
        dy = np.median(dd[:, :, 1] - Y)
        
        report.data('median_displ: ', (dx, dy))
        
        sensels = [0, self.shape[1],
                   self.shape[1] * (self.shape[0] - 1),
                   self.shape[1] * self.shape[0] - 1,
                   (self.shape[1] * self.shape[0] + self.shape[1]) / 2,
                   (self.shape[1] * self.shape[0] + self.shape[1]) / 2 + 1,
                   (self.shape[1] * self.shape[0] + self.shape[1]) / 2 + 2,
                   (self.shape[1] * self.shape[0] + self.shape[1]) / 2 + 3,
                   (self.shape[1] * self.shape[0] + self.shape[1]) / 2 + 4,
                   (self.shape[1] * self.shape[0] + self.shape[1]) / 2 + 5,
                   (self.shape[1] * self.shape[0] + self.shape[1]) / 2 + 6,
                   (self.shape[1] * self.shape[0] + self.shape[1]) / 2 + 7]
        
        f = report.figure(cols=6)
        for index in sensels:
            start = self.index2cell(index)
            center = dd[start[0], start[1], :]
#            pdb.set_trace()
            with f.plot('esim_area_%s' % index, caption='esim over %s' % index) as pylab:
                esim_score = self.neig_esim_score[index, :].reshape(self.grid_shape)
                c2f = 1.0 * np.array(self.area) / self.grid_shape
                xl, yl = self.area_positions_coarse[index] * c2f
                xu, yu = (self.area_positions_coarse[index] + self.grid_shape) * c2f
                
                pylab.imshow(esim_score, extent=(yl, yu, xl, xu), interpolation='nearest', origin='lower')

                pylab.xlim((0, self.shape[1]))
                pylab.ylim((self.shape[0], 0))
                
                ax = pylab.subplot(111)
                ax.xaxis.set_major_locator(MultipleLocator(self.shape[1] / 5))
                ax.xaxis.set_minor_locator(MultipleLocator(1))
                ax.yaxis.set_major_locator(MultipleLocator(self.shape[0] / 5))
                ax.yaxis.set_minor_locator(MultipleLocator(1))
                pylab.grid(True, 'major', linewidth=2, linestyle='solid')  # , alpha=0.5
                pylab.grid(True, 'minor', linewidth=.2, linestyle='solid')

                vector = center - start
                logger.info(' vector : %s' % vector)

                offs = 0.0
                if np.sum(vector ** 2) != 0:
                    pylab.arrow(start[1] + offs, start[0] + offs, vector[1], vector[0], head_width=0.5, length_includes_head=True, color='gray')
                else:
                    pylab.plot(start[1] + offs, start[0] + offs, markersize=.5, color='gray')
                
                pylab.grid()
        
                    
    @contract(score='array[NxA]', returns='array[UxV]')  # ,U*V=N*A') not with border
    def make_grid(self, score):
        fourd = self.unrolled2multidim(score)  # HxWxXxY
        return togrid(add_border(fourd, fill=np.max(fourd) * 2))
    
    @contract(v='array[NxA]', returns='array[HxWxXxY],N=H*W,A=X*Y')
    def unrolled2multidim(self, v):
        """ De-unrolls both dimensions to obtain a 4d vector. """
        H, W = self.shape
        X, Y = self.grid_shape
        res = np.zeros((H, W, X, Y), v.dtype)
        for i, j in coords_iterate((H, W)):
            k = self.cell2index([i, j])
            sim = v[k, :]
            sim[sim == np.min(sim)] = 0
            res[i, j, :, :] = sim.reshape((X, Y))
        return res
    
    def index2cell(self, index):
        return np.array((index / self.shape[1], index % self.shape[1]))

    def index2cell_tuple(self, index):
        return index / self.shape[1], index % self.shape[1]
    
    def cell2index(self, cell):
        return self.shape[1] * cell[0] + cell[1]

    def publish(self, pub):
        pass
    
    def merge(self, other):
        assert self.shape == other.shape
        assert self.grid_shape == other.grid_shape
        assert self.area_positions_coarse == other.area_positions_coarse 
        
        # Handles only esim, not eord
        self.neig_esim_score += other.neig_esim_score

def get_original_coordinate(grid_index, grid_shape, area_shape, area_position_coarse):
    '''
    
    :param grid_index:
    :param grid_shape:
    :param area_shape:
    :param area_position:
    '''
    
#    logger.debug(' asked grid index: ' + str(grid_index))

    grid_shape = np.array(grid_shape)
    area_shape = np.array(area_shape)
    area_position_coarse = np.array(area_position_coarse)
    
    area_position = (area_position_coarse.astype('float')) * area_shape / grid_shape
    
#    if not (np.floor(area_position * grid_shape / area_shape) == area_position_coarse).all():
#        logger.warn('something wrong with area transformations')
#        pdb.set_trace()
    
#    XY = list(itertools.product(np.linspace(0, area_shape[0] - 1, grid_shape[0]), np.linspace(0, area_shape[1] - 1, grid_shape[1])))
#    offs = np.array(area_shape).astype('float') / grid_shape / 2
    range0 = 1.0 * np.arange(0, grid_shape[0] + 0) * area_shape[0] / grid_shape[0]
    range1 = 1.0 * np.arange(0, grid_shape[1] + 0) * area_shape[1] / grid_shape[1]
    XY = np.array(list(itertools.product(range0, range1)))
    
#    local_coord_coarse = (grid_index % grid_shape[1], grid_index / grid_shape[1])
#    local_coord = np.array(local_coord_coarse).astype('float') * area_shape / grid_shape
#    logger.info(' %s = %s' % (XY[grid_index], local_coord))
    local_coord = XY[grid_index]
#    pdb.set_trace()
        
    return area_position + local_coord
    

def extract_wraparound(Y, (xl, xu, yl, yu)):
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





def compare():
    pass
