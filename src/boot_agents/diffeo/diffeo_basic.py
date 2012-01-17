from . import np, coords_iterate, new_contract, contract


@new_contract
#@contract(x='array[MxNx2](int32|float32)')
@contract(x='array[MxNx2]')
def valid_diffeomorphism(x):
    M, N = x.shape[0], x.shape[1]
    assert (0 <= x[:, :, 0]).all()
    assert (x[:, :, 0] < M).all()
    assert (0 <= x[:, :, 1]).all()
    assert (x[:, :, 1] < N).all()


@contract(shape='valid_2d_shape', returns='valid_diffeomorphism')
def diffeo_identity(shape):
    M = shape[0]
    N = shape[1]
    d = np.zeros((M, N, 2), 'int32')
    for i, j in coords_iterate(shape):
        d[i, j, 0] = i
        d[i, j, 1] = j
    return d

diffeomorphism_identity = diffeo_identity


def coords_to_X(c, shape):
    a, b = c
    assert 0 <= a <= shape[0]
    assert 0 <= b <= shape[1]
    a = float(a)
    b = float(b)
    a = a + 0.5 # center cells
    b = b + 0.5
    u = a / (shape[0])
    v = b / (shape[1])
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
def diffeo_from_function(shape, f):
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

diffeomorphism_from_function = diffeo_from_function


@contract(a='valid_diffeomorphism,array[MxNx2]',
          b='valid_diffeomorphism,array[MxNx2]',
          returns='valid_diffeomorphism,array[MxNx2]')
def diffeo_compose(a, b):
#    M, N = a.shape[0], a.shape[1]
    c = np.empty_like(a)
    c[:, :, 0] = diffeo_apply(b, a[:, :, 0])
    c[:, :, 1] = diffeo_apply(b, a[:, :, 1])
    return c


@contract(a='valid_diffeomorphism,array[MxNx2]',
          returns='valid_diffeomorphism,array[MxNx2]')
def diffeo_inverse(a):
    M, N = a.shape[0], a.shape[1]
    result = np.empty_like(a)
    result.fill(-1) # fill invalid data
    many = np.zeros((M, N))
    many.fill(0)
    for i, j in coords_iterate((M, N)):
        i1 = a[i, j, 0]
        j1 = a[i, j, 1]
        result[i1, j1, 0] = i
        result[i1, j1, 1] = j
        many[i1, j1] += 1

    num = (many == 0).sum()
    if num:
        fill_invalid(result[:, :, 0], -1)
        fill_invalid(result[:, :, 1], -1)

    return result


def fill_invalid(x, invalid_value):
    i, j = np.nonzero(x == invalid_value)
    coords = [np.array(s) for s in zip(i, j)]
    while coords:
        # extract random coordinates
        c = coords.pop(np.random.randint(len(coords)))
        options = []
        for d in [[-1, 0], [+1, 0], [0, 1], [0, -1]]:
            c2 = tuple(np.mod(c + d, x.shape).astype('int'))
            if x[c2] != invalid_value:
                options.append(x[c2])

        if not options:
            coords.append(c)
        else:
            x[tuple(c)] = most_frequent(options)


def most_frequent(a):
    return max(map(lambda val: (a.count(val), val), set(a)))[-1]

#
#def iterate(M0, modN, information, link_weight):
#    
#    print information
#    print M0
#    fixed = information > 0
#    M = M0.astype('float32')
#    Mprev = M
#    for k in range(10000):
##        print 'k=%s' % k
##        print M.astype('int32')
#        deriv = np.zeros(M.shape)
#        deriv += information * (M - M0)
#        deriv += 0.25 * link_weight * dmod(M - rot_L(M), modN) * rot_L(information) 
#        deriv += 0.25 * link_weight * dmod(M - rot_R(M), modN) * rot_R(information)
#        deriv += 0.25 * link_weight * dmod(M - rot_U(M), modN) * rot_U(information)
#        deriv += 0.25 * link_weight * dmod(M - rot_D(M), modN) * rot_D(information)
#        
#        dM = -0.2 * deriv
#        dM[fixed] = 0
#        
#        print np.sign(dM)
#        
#        M = M + dM
##        M[fixed] = M0[fixed]
##        time.sleep(4)
#        diff = np.abs(M - Mprev).max()
#        if diff < 0.001:
#            break
#        print ('%s Diff %s' % (k, diff)) 
#        Mprev = M
#    print M.astype('int32')
#    return M
#    

@contract(diffeo='valid_diffeomorphism,array[MxNx2]',
          template='array[MxNx...]')
def diffeo_apply(diffeo, template):
    M, N = diffeo.shape[0], diffeo.shape[1]
    result = np.empty_like(template)
    for i, j in coords_iterate((M, N)):
        i1 = diffeo[i, j, 0]
        j1 = diffeo[i, j, 1]
        result[i, j, ...] = template[i1, j1, ...]
    return result

def diffeo_local_differences(a, b):
    ''' returns tuple (x,y) with normalized difference fields.
        Each entry is normalized in [-0.5,0.5]  '''
    x = dmod(a[:, :, 0] - b[:, :, 0], a.shape[0] / 2).astype('float32') / (a.shape[0])
    y = dmod(a[:, :, 1] - b[:, :, 1], a.shape[1] / 2).astype('float32') / (a.shape[1])
    return x, y

@contract(a='valid_diffeomorphism,array[MxNx2]',
          b='valid_diffeomorphism,array[MxNx2]',
          returns='>=0,<=0.5')
def diffeo_distance_Linf(a, b):
    ''' Computes the distance between two diffeomorphism.
        This is the maximum difference between the two. 
        The minimum is 0 (of course);
        the maximum is 0.5
    '''
    x, y = diffeo_local_differences(a, b)
    dx = np.abs(x).max()
    dy = np.abs(y).max()
    return float(np.max(dx, dy))

@contract(a='valid_diffeomorphism,array[MxNx2]',
          b='valid_diffeomorphism,array[MxNx2]',
          returns='>=0,<0.71')
def diffeo_distance_L2(a, b):
    ''' The maximum is sqrt(0.5**2 + 0.5**2) = sqrt(0.5) = sqrt(1/2) = sqrt(2)/2 '''
    x, y = diffeo_local_differences(a, b)
    dist = np.sqrt(x * x + y * y)
    return float(dist.mean())

@contract(a='valid_diffeomorphism', returns='>=0, <0.71')
def diffeo_norm_L2(a):
    ''' The norm is the distance from the identity. '''
    # TODO: unittests
    b = diffeo_identity((a.shape[0], a.shape[1]))
    return diffeo_distance_L2(a, b)

def dmod(x, N):
    ''' Normalizes between [-N, +N-1] '''
    return ((x + N) % (2 * N)) - N
#    
#
#def rot_L(X):
#    Y = np.empty_like(X)
#    Y[1:, :] = X[:-1, :]
#    Y[0, :] = X[-1, :]
#    return Y
#
#def rot_R(X):
#    Y = np.empty_like(X)
#    Y[:-1, :] = X[1:, :]
#    Y[-1, :] = X[0, :]
#    return Y
#
#def rot_U(X):
#    Y = np.empty_like(X)
#    Y[:, 1:] = X[:, :-1]
#    Y[:, 0] = X[:, -1]
#    return Y
#
#def rot_D(X):
#    Y = np.empty_like(X)
#    Y[:, :-1] = X[:, 1:]
#    Y[:, -1] = X[:, 0]
#    return Y
