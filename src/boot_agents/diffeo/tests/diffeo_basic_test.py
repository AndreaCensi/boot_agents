from boot_agents.diffeo import coords_to_X, X_to_coords
from boot_agents.diffeo import coords_iterate
from numpy.testing.utils import assert_allclose

def X_to_coords_test():
    shape = (40, 50)
    for coords in coords_iterate(shape):
        X = coords_to_X(coords, shape)
        coords2 = X_to_coords(X, shape)
        msg = 'coords: %s  X: %s  coords2: %s' % (coords, X, coords2)
        assert_allclose(coords2, coords, err_msg=msg)
