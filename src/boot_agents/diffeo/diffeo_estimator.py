from . import (diffeomorphism_to_rgb, cmap, coords_iterate, Flattening, contract,
    np)


class Diffeomorphism2D:
    @contract(d='valid_diffeomorphism')
    def __init__(self, d, variance=None):
        ''' d[i,j] gives the index '''
        self.d = d
        if variance is None:
            variance = np.ones((d.shape[0], d.shape[1]))
        self.variance = variance
        
class DiffeomorphismEstimator():
    ''' Learns a diffeomorphism between two 2D fields. '''
    
    @contract(max_displ='seq[2](>0,<1)')
    def __init__(self, max_displ):
        self.shape = None
        self.max_displ = np.array(max_displ)
        self.last_y0 = None
        self.last_y1 = None
        
    @contract(y0='array[MxN]', y1='array[MxN]')
    def update(self, y0, y1):
        if self.shape is None:
            self.init_structures(y0)
            
        if self.shape != y0.shape:
            msg = 'Shape changed from %s to %s.' % (self.shape, y0.shape)
            raise ValueError(msg) 
        
        self.last_y0 = y0
        self.last_y1 = y1
        
        
#       Good for continuous
#        def sim1(a, b):
#            diff = np.abs(a - b)
#            return -diff
        def sim2(a, b): # food for 0-1
            return a * b
        
        y0_flat = y0.flat
        y1_flat = y1.flat
        for k in range(self.nsensels):
            # Fix a sensel in the later image
            a = y1_flat[k]
            # Look which originally was closer
            b = y0_flat[self.neighbor_indices_flat[k]]
            self.neighbor_similarity_flat[k] += sim2(a, b)
            
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
        for c in coords_iterate(self.shape):
            k = self.flattening.cell2index[c]
            sim = self.neighbor_similarity_flat[k]
            if (sim == 0).all():
                best_index = 0 
                variance[c] = np.inf
            else:
                best = np.argmax(sim) 
                best_index = self.neighbor_indices_flat[k][best]
                variance[c] = float(sim[best]) / sim.mean() 
            maximum_likelihood_index[c] = best_index
        d = self.flattening.flat2coords(maximum_likelihood_index)
        return Diffeomorphism2D(d, variance)
    
#    def summarize_smooth(self):
#        ''' Tries to enforce a smoothness constraint '''
#        maximum_likelihood_index = np.zeros(self.shape, dtype='int32')
#        variance = np.zeros(self.shape, dtype='float32')
#        for c in coords_iterate(self.shape):
#            k = self.flattening.cell2index[c]
#            sim = self.neighbor_similarity_flat[k]
#            if (sim == 0).all():
#                best_index = 0 
#                variance[c] = np.inf
#            else:
#                best = np.argmax(sim) 
#                best_index = self.neighbor_indices_flat[k][best]
#                variance[c] = float(sim[best]) / sim.mean() 
#            maximum_likelihood_index[c] = best_index
#        d = self.flattening.flat2coords(maximum_likelihood_index)
#        return Diffeomorphism2D(d, variance)

#    @contract(coords='tuple(int,int)')
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
    
    def publish(self, pub):
        diffeo = self.summarize()
        rgb = diffeomorphism_to_rgb(diffeo.d)
        # rgb = diffeomorphism_to_rgb_cont(diffeo.d)

        pub.array_as_image('mle', rgb, filter='scale')
        pub.array_as_image('variance', diffeo.variance, filter='scale')
            
        n = 20
        M = None
        for i in range(n): #@UnusedVariable
            c = self.flattening.random_coords()
            Mc = self.get_similarity(c)
            if M is None:
                M = Mc
                continue
            ok = np.isfinite(Mc)
            max = np.nanmax(Mc)
            if max > 0:
                M[ok] = Mc[ok] / max
            
        pub.array_as_image('coords', M, filter='scale')
        
        if self.last_y0 is not None: 
            y0 = self.last_y0
            y1 = self.last_y1           
            none = np.logical_and(y0 == 0, y1 == 0)
            x = y0 - y1
            x[none] = np.nan 
            pub.array_as_image('motion', x, filter='posneg')

