from boot_agents.diffeo import (coords_iterate, coords_to_X, X_to_coords,
    diffeo_identity, diffeo_compose, dmod, diffeo_distance_Linf,
    diffeo_from_function, diffeo_inverse, diffeo_distance_L2)
from boot_agents.diffeo.tests.diffeo_creation_test import (f_roty, f_rotx, f_id,
    f_rotx2, f_rotx_inv)
from numpy.testing.utils import assert_allclose

def X_to_coords_test():
    shape = (40, 50)
    for coords in coords_iterate(shape):
        X = coords_to_X(coords, shape)
        coords2 = X_to_coords(X, shape)
        msg = 'coords: %s  X: %s  coords2: %s' % (coords, X, coords2)
        assert_allclose(coords2, coords, err_msg=msg)


def diffeo_composition_test_1():
    shape = (90, 90)
    identity = diffeo_identity(shape)    
    i2 = diffeo_compose(identity, identity)
    assert_allclose(diffeo_distance_Linf(identity, i2), 0) 


def diffeo_composition_test_5():
    shape = (90, 90)
    f = diffeo_from_function(shape, f_rotx)    
    f2 = diffeo_from_function(shape, f_rotx2)
    ff = diffeo_compose(f, f)
    assert_allclose(diffeo_distance_Linf(ff, f2), 0, atol=0.02) 


def diffeo_distance_test_4():
    shape = (20, 20)
    identity = diffeo_identity(shape)
    identity2 = diffeo_from_function(shape, f_id)
    assert_allclose(diffeo_distance_Linf(identity, identity2), 0)

def diffeo_distance_test_1():
    shape = (20, 20)
    identity = diffeo_identity(shape)
    # They all rotate by 0.1 in [-1,1]; so maximum will be 0.05
    d_rotx = diffeo_from_function(shape, f_rotx)
    assert_allclose(diffeo_distance_Linf(identity, d_rotx), 0.05) 

def diffeo_distance_test_L2_0():
    shape = (20, 20)
    identity = diffeo_identity(shape)
    identity2 = diffeo_from_function(shape, f_id)
    assert_allclose(diffeo_distance_L2(identity, identity2), 0)

def diffeo_distance_test_L2_1():
    shape = (20, 20)
    identity = diffeo_identity(shape)
    # They all rotate by 0.1 in [-1,1]; so maximum will be 0.05
    d_rotx = diffeo_from_function(shape, f_rotx)
    assert_allclose(diffeo_distance_L2(identity, d_rotx), 0.05, rtol=0.01) 

def diffeo_inverse_test_1():
    shape = (20, 20)
    # They all rotate by 0.1 in [-1,1]; so maximum will be 0.05
    d = diffeo_from_function(shape, f_rotx)
    d_inv0 = diffeo_from_function(shape, f_rotx_inv)
    d_inv = diffeo_inverse(d)
    eps = 0
    
    def dclose(d1, d2):
        assert_allclose(diffeo_distance_Linf(d1, d2), 0, atol=eps)
        
    dclose(d_inv, d_inv0)
    
    d2 = diffeo_inverse(d_inv) 
    dclose(d, d2)

    identity = diffeo_identity(shape)
    identity2 = diffeo_compose(d, d_inv)
    identity3 = diffeo_compose(d_inv, d)
    dclose(identity, identity2)
    dclose(identity, identity3) 
    
def diffeo_inverse_suite(f):
    shape = (30, 30)
    # They all rotate by 0.1 in [-1,1]; so maximum will be 0.05
    d = diffeo_from_function(shape, f)
    d_inv = diffeo_inverse(d)
    d2 = diffeo_inverse(d_inv) 
    eps = 0.05
    
    def dclose(d1, d2):
        assert_allclose(diffeo_distance_Linf(d1, d2), 0, atol=eps)
    
    dclose(d, d2)

    identity = diffeo_identity(shape)
    identity2 = diffeo_compose(d, d_inv)
    identity3 = diffeo_compose(d_inv, d)
    dclose(identity, identity2)
    dclose(identity, identity3) 
    
def diffeo_inverse_test():
    use = [f_id, f_rotx, f_roty]
    for f in use:
        yield diffeo_inverse_suite, f

def dmod_test():
    N = 5
    values = [
        ((+0, N), +0),
        ((+1, N), +1),
        ((-1, N), -1),
        ((+N, N), -N),
        ((+N - 1, N), +N - 1),
        ((-N, N), -N),
        ((-N + 1, N), -N + 1),
    ]
    for params, result in values:
        obtained = dmod(*params)
        assert_allclose(obtained, result)
#
#def rot_test():
#    A = np.random.rand(5, 10)
#    assert_allclose(rot_R(rot_L(A)), A)
#    assert_allclose(rot_U(rot_D(A)), A)
    
