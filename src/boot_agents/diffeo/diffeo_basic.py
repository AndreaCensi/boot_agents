from . import np, coords_iterate, new_contract, contract
from geometry.basic_utils import assert_allclose

@new_contract
@contract(x='array[MxNx2](int32)')
def valid_diffeomorphism(x):
    M, N = x.shape[0], x.shape[1]
    assert (0 <= x[:, :, 0]).all()
    assert (x[:, :, 0] < M).all()
    assert (0 <= x[:, :, 1]).all()
    assert (x[:, :, 1] < N).all()
    
@contract(shape='valid_2d_shape', returns='valid_diffeomorphism')
def diffeomorphism_identity(shape):
    M = shape[0]
    N = shape[1]
    d = np.zeros((M, N, 2), 'int32')
    for i, j in coords_iterate(shape):
        d[i, j, 0] = i
        d[i, j, 1] = j
    return d

def coords_to_X(c, shape):
    a, b = c
    assert 0 <= a <= shape[0]
    assert 0 <= b <= shape[1]
    a = float(a)
    b = float(b)
    u = a / (shape[0] - 1)
    v = b / (shape[1] - 1)
    x = 2 * (u - 0.5)
    y = 2 * (v - 0.5)
    assert -1 <= x <= +1
    assert -1 <= y <= +1
    return (x, y)

def X_to_coords(X, shape):
    x, y = X
    assert -1 <= x <= +1, 'outside bounds: %s' % x
    assert -1 <= y <= +1, 'outside bounds: %s' % y
    a = np.round((x / 2 + 0.5) * (shape[0] - 1))
    b = np.round((y / 2 + 0.5) * (shape[1] - 1))
    a = int(a)
    b = int(b)
    assert 0 <= a <= shape[0]
    assert 0 <= b <= shape[1]
    return (a, b)

@contract(shape='valid_2d_shape', returns='valid_diffeomorphism')
def diffeomorphism_from_function(shape, f):
    ''' f must be a function from M=[-1,1]x[-1,1] to itself. '''
    # let X = (x,y) \in M
    
    M = shape[0]
    N = shape[1]
    D = np.zeros((M, N, 2), dtype='int32')
    for coords in coords_iterate(shape):
        X = coords_to_X(coords, shape)
        Y = f(X)
        a, b = X_to_coords(Y, shape)
        D[coords[0], coords[1], :] = [a, b] 
    return D

        
