from . import (diffeomorphism_to_rgb, cmap, coords_iterate, Flattening, contract,
    np, diffeo_to_rgb_norm, diffeo_to_rgb_angle, angle_legend, diffeo_to_rgb_curv,
    diffeo_text_stats, Diffeomorphism2D)
from PIL import Image #@UnresolvedImport
from matplotlib import cm
import numpy.linalg as la
# from scipy.signal import convolve2d
# from scipy.special import erf
# TODO: remove "print" statements

def sim_continuous(a, b):
    # XXX strange conversions
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)) ** 2
    #diff = np.abs(a - b)
    return diff


def sim_binary(a, b): # good for 0-1
    return a * b


MATCH_CONTINUOUS = 'continuous'
MATCH_BINARY = 'binary'
diff2_kern = [[0, .5, 0], [.5, -1, .5], [0, .5, 0]]

class DiffeomorphismEstimator():
    ''' Learns a diffeomorphism between two 2D fields. '''

    @contract(max_displ='seq[2](>0,<1)', match_method='str')
    def __init__(self, max_displ, match_method):
        """ 
            :param max_displ: Maximum displacement the diffeomorphism d_max
            :param match_method: Either "continuous" or "binary" (to compute the 
                error function).
        """
        print('diffeo_estimator.py in boot_agents.diffeo is deprecated')
        assert False
        self.shape = None
        self.max_displ = np.array(max_displ)
        self.last_y0 = None
        self.last_y1 = None

        assert match_method in [MATCH_CONTINUOUS, MATCH_BINARY]
        self.match_method = match_method

        self.num_samples = 0

    @contract(y0='array[MxN]', y1='array[MxN]')
    def update(self, y0, y1):
#        ydd += convolve2d(y0, diff2_kern, mode='same')
        self.num_samples += 1

        # init structures if not already
        if self.shape is None:
            self.init_structures(y0)

        # check shape didn't change
        if self.shape != y0.shape:
            msg = 'Shape changed from %s to %s.' % (self.shape, y0.shape)
            raise ValueError(msg)

        # remember last images
        self.last_y0 = y0
        self.last_y1 = y1

        # Chooses the error function based on the parameter in contructor
        if self.match_method == MATCH_CONTINUOUS:
            similarity = sim_continuous
        elif self.match_method == MATCH_BINARY:
            similarity = sim_binary
        else:
            assert False

        # "Converts" the images to one dimensional vectors
        y0_flat = y0.flat
        y1_flat = y1.flat
        # For each pixel in the second image
        for k in range(self.nsensels):
            # Look at its value "a"
            a = y1_flat[k]
            # Look which originally was closer
            # these values: self.neighbor_indices_flat[k] 
            # give you which indices in the flat vectors are 
            # close to k-th pixel
            
            # values of the neighbors 
            b = y0_flat[self.neighbor_indices_flat[k]]
            
            # compute similarity
            neighbor_sim = similarity(a, b)
            
            # compute the similarity order
            neighbor_argsort = np.zeros(len(neighbor_sim))
            neighbor_argsort[np.argsort(neighbor_sim)] += range(len(neighbor_sim))
            
            # compute best similarity
            self.neighbor_num_bestmatch_flat[k] += (neighbor_sim != np.min(neighbor_sim))  
            
            # keep track of which neighbors are more similar on average
            self.neighbor_similarity_flat[k] += neighbor_sim
            self.neighbor_similarity_best[k] += np.min(neighbor_sim)
            self.neighbor_argsort_flat[k] += neighbor_argsort


    def init_structures(self, y):
        self.shape = y.shape
        self.nsensels = y.size

        self.ydd = np.zeros(y.shape, dtype='float32')

        # for each sensel, create an area
        self.lengths = np.ceil(self.max_displ * 
                               np.array(self.shape)).astype('int32')
        print(' Field Shape: %s' % str(self.shape))
        print('    Fraction: %s' % str(self.max_displ))
        print(' Search area: %s' % str(self.lengths))

        self.neighbor_coords = [None] * self.nsensels
        self.neighbor_indices = [None] * self.nsensels
        self.neighbor_indices_flat = [None] * self.nsensels
        self.neighbor_similarity_flat = [None] * self.nsensels
        self.neighbor_similarity_best = np.zeros(self.nsensels, dtype='float32')
        self.neighbor_argsort_flat = [None] * self.nsensels
        self.neighbor_num_bestmatch_flat = [None] * self.nsensels

        self.flattening = Flattening.by_rows(y.shape)
        print('Creating structure shape %s lengths %s' % 
              (self.shape, self.lengths))
        cmg = cmap(self.lengths)
        for coord in coords_iterate(self.shape):
            k = self.flattening.cell2index[coord]
            cm = cmg.copy()
            cm[:, :, 0] += coord[0]
            cm[:, :, 1] += coord[1]
            cm[:, :, 0] = cm[:, :, 0] % self.shape[0]
            cm[:, :, 1] = cm[:, :, 1] % self.shape[1]
            self.neighbor_coords[k] = cm

            indices = np.zeros(self.lengths, 'int32')
            for a, b in coords_iterate(indices.shape):
                c = tuple(cm[a, b, :])
                indices[a, b] = self.flattening.cell2index[c]

            self.neighbor_indices[k] = indices
            self.neighbor_indices_flat[k] = np.array(indices.flat)
            self.neighbor_similarity_flat[k] = np.zeros(indices.size,
                                                        dtype='float32')
            self.neighbor_argsort_flat[k] = np.zeros(indices.size,
                                                        dtype='float32')
            self.neighbor_num_bestmatch_flat[k] = np.zeros(indices.size,
                                                        dtype='uint')
        print('done')
    
    def summarize(self):
        ''' 
            Find maximum likelihood estimate for diffeomorphism looking 
            at each pixel singularly. 
            
            Returns a Diffeomorphism2D.
        '''
        maximum_likelihood_index = np.zeros(self.shape, dtype='int32')
        variance = np.zeros(self.shape, dtype='float32')
        E2 = np.zeros(self.shape, dtype='float32')
        E3 = np.zeros(self.shape, dtype='float32')
        E4 = np.zeros(self.shape, dtype='float32')
        num_problems = 0
        
        order_image = Image.new('L', np.flipud(self.shape) * np.flipud(self.lengths))
        
        i = 0
        # for each coordinate
        for c in coords_iterate(self.shape):
            # find index in flat array
            k = self.flattening.cell2index[c]
            # Look at the average similarities of the neihgbors
            sim = self.neighbor_similarity_flat[k]
            sim_min = sim.min()
            sim_max = sim.max()
            if sim_max == sim_min:
                # if all the neighbors have the same similarity
                best_index = 0
                variance[c] = 0 # minimum information
                maximum_likelihood_index[c] = best_index
            else:
                best = np.argmin(sim)
                best_index = self.neighbor_indices_flat[k][best]
                # uncertainty ~= similarity of the best pixel
                variance[c] = sim[best]   
                maximum_likelihood_index[c] = best_index
            
            

            E2[c] = self.neighbor_similarity_best[k] / self.num_samples
            # Best match error
            E3[c] = np.min(self.neighbor_num_bestmatch_flat[k]) / self.num_samples
            
            E4[c] = np.min(self.neighbor_argsort_flat[k]) / self.num_samples
            
#            pdb.set_trace()
            p0 = tuple(np.flipud(np.array(c) * self.lengths))
            E4_square = (self.neighbor_argsort_flat[k] / self.num_samples).reshape(self.lengths)
            order_image.paste(Image.fromarray((E4_square / np.max(E4_square) * 255).astype('uint8')), p0 + tuple(p0 + self.lengths))

            i += 1
            
        order_image.save('order.png')
        
        d = self.flattening.flat2coords(maximum_likelihood_index)

        if num_problems > 0:
            print('Warning, %d were not informative.' % num_problems)
            pass
        
        sqrt_2_sigma2 = np.sqrt(2 * variance / self.num_samples)
        
#        eps = 1
#        P0 = (erf(-1 / sqrt_2_sigma2) - erf(1 / sqrt_2_sigma2)) / 2
#        pdb.set_trace()
        
        # normalization for this variance measure
        vmin = variance.min()
        variance = variance - vmin
        vmax = variance.max()
        if vmax > 0:
            variance *= (1 / vmax)
            
        # return maximum likelihood plus uncertainty measure
        return Diffeomorphism2D(d, variance, E2, E3, E4)
    
    def summarize_continuous(self, quivername):
        center = np.zeros(list(self.shape) + [2], dtype='float32')
        spread = np.zeros(self.shape, dtype='float32')
        maxerror = np.zeros(self.shape, dtype='float32')
        minerror = np.zeros(self.shape, dtype='float32')
        
        sim_image = Image.new('L', np.flipud(self.shape) * np.flipud(self.lengths))
        zer_image = Image.new('L', np.flipud(self.shape) * np.flipud(self.lengths))
        for c in coords_iterate(self.shape):
            # find index in flat array
            k = self.flattening.cell2index[c]
            # Look at the average similarities of the neihgbors
            sim = self.neighbor_similarity_flat[k]
            
            sim_square = sim.reshape(self.lengths).astype('float32') / self.num_samples
            sim_square, minerror[c], maxerror[c] = sim_square_modify(sim_square, np.min(self.neighbor_similarity_flat) / self.num_samples, np.max(self.neighbor_similarity_flat) / self.num_samples)
            avg_square = (np.max(sim_square) + np.min(sim_square)) / 2
            sim_zeroed = np.zeros(sim_square.shape)
#            pdb.set_trace()
            sim_zeroed[sim_square > 0.85] = sim_square[sim_square > 0.85] 
            center[c], spread[c] = get_cm(sim_square)
#            pdb.set_trace()
            p0 = tuple(np.flipud(np.array(c) * self.lengths))
            sim_image.paste(Image.fromarray((sim_square * 255).astype('uint8')), p0 + tuple(p0 + self.lengths))
            zer_image.paste(Image.fromarray((sim_zeroed * 255).astype('uint8')), p0 + tuple(p0 + self.lengths))
            
        sim_image.save(quivername + 'simimage.png')
        zer_image.save(quivername + 'simzeroed.png')
            
            
        display_disp_quiver(center, quivername)
        display_continuous_stats(center, spread, minerror, maxerror, quivername)
        
#        pdb.set_trace()
        dcont = displacement_to_coord(center)
        diff = dcont.astype('int')
        diff = get_valid_diffeomorphism(diff)
        diffeo2d = Diffeomorphism2D(diff)
        
#        display_diffeo_images(diff, quivername)
#        pdb.set_trace()
        return diffeo2d
        
    def summarize_smooth(self, noise=0.1):
        ''' Find best estimate for diffeomorphism 
            looking at each singularly. '''
        maximum_likelihood_index = np.zeros(self.shape, dtype='int32')
        variance = np.zeros(self.shape, dtype='float32')
        epsilon = None
        for c in coords_iterate(self.shape):
            k = self.flattening.cell2index[c]
            sim = self.neighbor_similarity_flat[k]
            if epsilon is None:
                epsilon = np.random.randn(*sim.shape)
            sim_min = sim.min()
            sim_max = sim.max()
            if sim_max == sim_min:
                best_index = 0
                variance[c] = 0
            else:
                std = noise * (sim_max - sim_min)
                best = np.argmin(sim + epsilon * std)
                best_index = self.neighbor_indices_flat[k][best]
                variance[c] = sim[best]
            maximum_likelihood_index[c] = best_index
        d = self.flattening.flat2coords(maximum_likelihood_index)
#        pdb.set_trace()
        variance = variance - variance.min()
        vmax = variance.max()
        if vmax > 0:
            variance *= (1 / vmax)
        return Diffeomorphism2D(d, variance)

    #    @contract(coords='tuple(int,int)') # XXX: int32 not accepted
    def get_similarity(self, coords):
        ''' Returns the similarity field for one cell. (outside are NaN) '''
        k = self.flattening.cell2index[coords]
        M = np.zeros(self.shape)
        M.fill(np.nan)
        neighbors = self.neighbor_indices_flat[k]
        sim = self.neighbor_similarity_flat[k]
        M.flat[neighbors] = sim

        best = np.argmax(sim)
        M.flat[neighbors[best]] = np.NaN
        return M

    def summarize_averaged(self, n=10, noise=0.1):
        d = []
        for _ in range(n):
            diff = self.summarize_smooth(noise)
            d.append(diff.d)
            print('.')
        ds = np.array(d, 'float')
        avg = ds.mean(axis=0)
        #var  = diff.variance
        var = ds[:, :, :, 0].var(axis=0) + ds[:, :, :, 1].var(axis=0)
        print var.shape
        assert avg.shape == diff.d.shape
        return Diffeomorphism2D(avg, var)

    def publish(self, pub):
        diffeo = self.summarize()
#        diffeo = self.summarize_averaged(10, 0.02) # good for camera
#        diffeo = self.summarize_averaged(2, 0.1)
        print('Publishing')
        #pdb.set_trace()
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


### Test functions

import matplotlib.pyplot as plt

@contract(x='array[MxNx2]')
def get_valid_diffeomorphism(x):
    M, N = x.shape[0], x.shape[1]
    
    #assert (0 <= x[:, :, 0]).all()
    #assert (0 <= x[:, :, 1]).all()
    x[x < 0] = 0
    
    #assert (x[:, :, 0] < M).all()
    x[x[:, :, 0] >= M] = M - 1
    
    #assert (x[:, :, 1] < N).all()
    x[x[:, :, 1] >= N] = N - 1
    
    return x
    
@contract(x='array[MxNx2]')
def displacement_to_coord(x):
    Y, X = np.meshgrid(range(x.shape[1]), range(x.shape[0]))
    
    x[:, :, 0] = x[:, :, 0] + Y
    x[:, :, 1] = x[:, :, 1] + X
    
    return x

@contract(diff='valid_diffeomorphism,array[MxNx2]')
def display_diffeo_images(diff, name):
    im_ang = Image.fromarray(diffeo_to_rgb_angle(diff)).resize((400, 300))
    im_norm = Image.fromarray(diffeo_to_rgb_norm(diff)).resize((400, 300))
    im_ang.save(name + 'ang.png')
    im_ang.save(name + 'norm.png')

    
def sim_square_check(sim_square):
#    pdb.set_trace()
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    X, Y = np.meshgrid(range(sim_square.shape[0]), range(sim_square.shape[0]))
#    surf = ax.plot_surface(X, Y, sim_square, rstride=1, cstride=1, cmap=cm.jet,
#        linewidth=0, antialiased=False)
#    fig.colorbar(surf, shrink=0.5, aspect=5)
#    plt.savefig('test.png')
    Image.fromarray((sim_square * 255).astype('uint8')).resize((400, 300)).save('sim_square.png')
    
def display_disp_quiver(diff, name):
#    pdb.set_trace()
    Y, X = np.meshgrid(range(diff.shape[1]), range(diff.shape[0]))
    fig = plt.figure()
    Q = plt.quiver(X, Y, diff[:, :, 1], -diff[:, :, 0])
    plt.savefig(name)
    
def display_continuous_stats(diff, variance, minerror, maxerror, name):
    Y, X = np.meshgrid(range(diff.shape[1]), range(diff.shape[0]))
    fig = plt.figure(figsize=(16, 12), dpi=600)
    Q = plt.quiver(X, Y, diff[:, :, 1], diff[:, :, 0])
    
    minvar = np.min(variance)
    maxvar = np.max(variance)
    norm_variance = (variance - minvar) / (maxvar - minvar) * 50 + 1
#    plt.scatter(X + diff[:, :, 0], Y + diff[:, :, 1], c=minerror, s=norm_variance)
    plt.scatter(X, Y, c=minerror, s=norm_variance)
    
    plt.savefig(name)
    
    
    
    

def sim_square_modify(sim_square, minval=None, maxval=None):
    if minval is None:
        mi = np.min(sim_square)
    else:
        mi = minval
    if maxval is None:
        ma = np.max(sim_square)
    else:
        ma = maxval
    
    mod_square = -(sim_square - ma) / (ma - mi)
    return mod_square, mi, ma
    
def get_cm(sim_arr):
#    pdb.set_trace()
    shape = np.array(sim_arr.shape)
    cent = (shape.astype('float') - [1, 1]) / 2
#    size = sim_arr.size
    # center of mass
    torque = np.array([0.0, 0.0])
    mass = 0.0
    inertia_sum = 0.0
    
    area = np.zeros(sim_arr.shape)
    area[sim_arr > np.max(sim_arr) * 0.9] = sim_arr[sim_arr > np.max(sim_arr) * 0.9]
        
    for cn in coords_iterate(shape):
        r = (np.array(cn) - cent)
        torque += area[cn] * r
        mass += area[cn]
        
    # if no mass, return zero
    if mass == 0:
        return np.array([0, 0]), 0    
    
    cm = torque / mass
    
    for cn in coords_iterate(shape):
        r = (np.array(cn) - cent)
        a = la.norm(r)
        inertia_sum += a ** 2 * area[cn]
    
    inertia = inertia_sum / mass 
    return cm, inertia
