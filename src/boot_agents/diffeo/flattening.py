from . import contract, coords_iterate, np
from compmake.utils import memoize_simple as memoize



class Flattening:
    ''' A Flattening is a way to assign a 1D index to each cell of a 
        2D array. '''
    
    @contract(cell2index='array[MxN](int32,>=0,<M*N)',
              index2cell='array[(M*N)x2](int32,>=0)')
    def __init__(self, cell2index, index2cell):
        Flattening.check_conjugate(cell2index, index2cell)
        self.size = cell2index.size
        self.shape = cell2index.shape
        self.cell2index = cell2index
        self.cell2index_flat = np.array(self.cell2index.flat)
        self.index2cell = index2cell
        
    @contract(returns='array[MxN](int32,>=0,<M*N)')
    def get_cell2index(self):
        return self.cell2index.copy()
        
    @staticmethod
    @contract(cell2index='array[MxN](int32,>=0,<M*N)',
              index2cell='array[(M*N)x2](int32,>=0)')
    def check_conjugate(cell2index, index2cell):
        for k in range(cell2index.size):
            assert cell2index[tuple(index2cell[k, :])] == k
        for c in coords_iterate(cell2index.shape):
            k = cell2index[c]
            assert index2cell[k, 0] == c[0]
            assert index2cell[k, 1] == c[1]


    @staticmethod
    @contract(shape='seq[2](int,>0)')
    @memoize
    def by_rows(shape):
        M = shape[0]
        N = shape[1]
        cell2index = np.zeros((M, N), 'int32')
        index2cell = np.zeros((M * N, 2), 'int32')
        k = 0
        for i, j in coords_iterate(shape):
            cell2index[i, j] = k
            index2cell[k] = [i, j]
            k += 1
        return Flattening(cell2index, index2cell)

    @contract(flat='array[MxN](int32, >=0, <=M*N)')
    def flat2coords(self, flat):
        ''' Converts a representation of the type index[i,j] = k
            to diffeo[i,j]= [i_k, j_k] '''
        if flat.shape != self.shape:
            msg = 'Expected shape %s, got %s.' % (self.shape, flat.shape)
            raise ValueError(msg)
        res = np.zeros((self.shape[0], self.shape[1], 2), dtype='int32')
        for i, j in coords_iterate(self.shape):
            k = flat[i, j]
            res[i, j, :] = self.index2cell[k, :]
        return res

    def random_coords(self):
        k = np.random.randint(self.size)
        return tuple(self.index2cell[k, :])

    @contract(image='array[HxW]', returns='array[H*W]')
    def rect2flat(self, image):
        if image.shape != self.shape:
            msg = ('I expect the shape to be %s but got %s.' 
                   % (self.shape, image.shape))
            raise ValueError(msg)
        # Several alternatives:
        # return image.flatten()[self.cell2index_flat]
        # return image.take(self.cell2index_flat, mode='clip')
        result = image.take(self.cell2index_flat)
        
        assert result.shape == (self.size,)
        return result
        
    @contract(values='array[N]', returns='array[HxW]')
    def flat2rect(self, values):
        if values.shape != (self.size,):
            msg = ('I expect the shape to be (%s,) but got %s.' 
                   % (self.size, values.shape))
            raise ValueError(msg)
        result = values[self.cell2index]
        assert result.shape == self.shape
        return result
    
