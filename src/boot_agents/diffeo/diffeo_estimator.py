from . import cmap, coords_iterate, Flattening, contract, np
from boot_agents.diffeo.diffeo_display import diffeomorphism_to_rgb


class Diffeomorphism2D:
    @contract(d='valid_diffeomorphism')
    def __init__(self, d):
        ''' d[i,j] gives the index '''
        self.d = d
        
class DiffeomorphismEstimator():
    ''' Learns a diffeomorphism between two 2D fields. '''
    
    @contract(max_displ='seq[2](>0,<1)')
    def __init__(self, max_displ):
        self.shape = None
        self.max_displ = np.array(max_displ)
        
    @contract(y0='array[MxN]', y1='array[MxN]')
    def update(self, y0, y1):
        if self.shape is None:
            self.init_structures(y0)
            
        if self.shape != y0.shape:
            msg = 'Shape changed from %s to %s.' % (self.shape, y0.shape)
            raise ValueError(msg) 
        
        def similarity(a, b):
            diff = np.abs(a - b)
            return diff
        
        y0_flat = y0.flat
        y1_flat = y1.flat
        for k in range(self.nsensels):
            a = y0_flat[k]
            b = y1_flat[self.neighbor_indices_flat[k]]
            self.neighbor_similarity_flat[k] += similarity(a, b)
            
    def init_structures(self, y):
        self.shape = y.shape
        self.nsensels = y.size
        
        # for each sensel, create an area
        self.lengths = np.ceil(self.max_displ * np.array(self.shape)).astype('int32')
         
        self.neighbor_coords = [None] * self.nsensels
        self.neighbor_indices = [None] * self.nsensels
        self.neighbor_indices_flat = [None] * self.nsensels
        self.neighbor_similarity_flat = [None] * self.nsensels
        
        self.flattening = Flattening.by_rows(y.shape)
        
        for coord in coords_iterate(self.shape):
            k = self.flattening.cell2index[coord]
            cm = cmap(self.lengths)
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
            self.neighbor_similarity_flat[k] = np.zeros(indices.size) 
            
    def summarize(self):
        maximum_likelihood_index = np.zeros(self.shape, dtype='int32')
        for c in coords_iterate(self.shape):
            k = self.flattening.cell2index[c]
            best = np.argmax(self.neighbor_similarity_flat[k]) 
            best_index = self.neighbor_indices_flat[k][best]
            maximum_likelihood_index[c] = best_index
        d = self.flattening.flat2coords(maximum_likelihood_index)
        return Diffeomorphism2D(d)
    
    def publish_debug(self, pub):
        diffeo = self.summarize()
        ks = [(0, 0), (20, 30), (5, 30)]
        for coords in ks:
            k = self.flattening.cell2index[coords]
            M = np.zeros(self.shape)
            M.fill(np.nan)
            neighbors = self.neighbor_indices_flat[k]
            sim = self.neighbor_similarity_flat[k]
            M.flat[neighbors] = sim
            pub.array_as_image('M%s' % k, M, filter='scale')
            
        rgb = diffeomorphism_to_rgb(diffeo.d)
        
        pub.array_as_image('mle', rgb, filter='scale')


