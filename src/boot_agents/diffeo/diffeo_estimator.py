from . import (diffeomorphism_to_rgb, cmap, coords_iterate, Flattening, contract,
    np, diffeo_to_rgb_norm, diffeo_to_rgb_angle, angle_legend)
from boot_agents.diffeo.diffeo_display import diffeo_to_rgb_curv, \
    diffeo_text_stats


class Diffeomorphism2D:
    @contract(d='valid_diffeomorphism')
    def __init__(self, d, variance=None):
        ''' d[i,j] gives the index '''
        self.d = d
        if variance is None:
            variance = np.ones((d.shape[0], d.shape[1]))
        else:
            assert variance.shape == d.shape[:2]
            assert np.isfinite(variance).all()
        self.variance = variance
        
MATCH_CONTINUOUS = 'continuous'
MATCH_BINARY = 'binary'

# TODO: remove "print" statements
def sim_continuous(a, b):
            diff = np.abs(a - b)
            return -diff
    
def sim_binary(a, b): # good for 0-1
    return a * b

class DiffeomorphismEstimator():
    ''' Learns a diffeomorphism between two 2D fields. '''
    
    @contract(max_displ='seq[2](>0,<1)', match_method='str')
    def __init__(self, max_displ, match_method):
        self.shape = None
        self.max_displ = np.array(max_displ)
        self.last_y0 = None
        self.last_y1 = None
        
        assert match_method in [MATCH_CONTINUOUS, MATCH_BINARY]
        self.match_method = match_method
        
        self.num_samples = 0
        
    @contract(y0='array[MxN]', y1='array[MxN]')
    def update(self, y0, y1):
        self.num_samples += 1
        
        if self.shape is None:
            self.init_structures(y0)
            
        if self.shape != y0.shape:
            msg = 'Shape changed from %s to %s.' % (self.shape, y0.shape)
            raise ValueError(msg) 
        
        self.last_y0 = y0
        self.last_y1 = y1

        if self.match_method == MATCH_CONTINUOUS:
            similarity = sim_continuous
        elif self.match_method == MATCH_BINARY:
            similarity = sim_binary
        else: assert False
        
        y0_flat = y0.flat
        y1_flat = y1.flat
        for k in range(self.nsensels):
            # Fix a sensel in the later image
            a = y1_flat[k]
            # Look which originally was closer
            b = y0_flat[self.neighbor_indices_flat[k]]
            self.neighbor_similarity_flat[k] += similarity(a, b)
            
    def init_structures(self, y):
        self.shape = y.shape
        self.nsensels = y.size
        
        # for each sensel, create an area
        self.lengths = np.ceil(self.max_displ * np.array(self.shape)).astype('int32')
        print(' Field Shape: %s' % str(self.shape))
        print('    Fraction: %s' % str(self.max_displ))
        print(' Search area: %s' % str(self.lengths))
        
        self.neighbor_coords = [None] * self.nsensels
        self.neighbor_indices = [None] * self.nsensels
        self.neighbor_indices_flat = [None] * self.nsensels
        self.neighbor_similarity_flat = [None] * self.nsensels
        
        self.flattening = Flattening.by_rows(y.shape)
        print('Creating structure shape %s lengths %s' % (self.shape, self.lengths))
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
            self.neighbor_similarity_flat[k] = np.zeros(indices.size, dtype='float32') 
        print('done')
        
    def summarize(self):
        ''' Find best estimate for diffeomorphism looking at each singularly. '''
        maximum_likelihood_index = np.zeros(self.shape, dtype='int32')
        variance = np.zeros(self.shape, dtype='float32')
        num_problems = 0
        for c in coords_iterate(self.shape):
            k = self.flattening.cell2index[c]
            sim = self.neighbor_similarity_flat[k]
            sim_min = sim.min()
            sim_max = sim.max()
            if sim_max == sim_min:
                best_index = 0 
                variance[c] = 0
                maximum_likelihood_index[c] = best_index
            else:
#                sim_sort = sorted(sim)
#                if sim_sort[-2] == sim_sort[-1]:
#                    num_problems += 1
#                    # not informative; use self
##                    print('Warning: %s' % sim_sort)
#                    variance[c] = 1
#                    maximum_likelihood_index[c] = self.flattening.cell2index[c]
#                else:
                best = np.argmax(sim) 
                best_index = self.neighbor_indices_flat[k][best]
                variance[c] = sim[best]
#                variance[c] = 0.5        
                maximum_likelihood_index[c] = best_index
        d = self.flattening.flat2coords(maximum_likelihood_index)
                 
        if num_problems > 0:
            print('Warning, %d were not informative.' % num_problems)
            pass
                    #  variance[c] = (sim[best] - sim.mean()) / (sim[best] - sim.min()) 
                    #  variance[c] = (sim[best] - sim.min())
        # TODO: check conditions
        variance = variance - variance.min()
        vmax = variance.max()
        if vmax > 0:
            variance *= (1 / vmax) 
 
        return Diffeomorphism2D(d, variance)
    
    def summarize_smooth(self, noise=0.1):
        ''' Find best estimate for diffeomorphism looking at each singularly. '''
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
                best = np.argmax(sim + epsilon * std) 
                best_index = self.neighbor_indices_flat[k][best]
                variance[c] = sim[best]             
            maximum_likelihood_index[c] = best_index
        d = self.flattening.flat2coords(maximum_likelihood_index)
  
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
