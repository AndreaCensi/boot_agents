from . import contract, coords_iterate, np


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
        self.index2cell = index2cell

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
    @contract(shape='seq[2](>0)')
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

