from boot_agents.diffeo.flattening import Flattening
import numpy as np
from geometry.utils.numpy_backport import assert_allclose


def test_flattening1():
    """ Make sure that image2flat and flat2image are invertible """
    shape = (120, 160)
    flattening = Flattening.by_rows(shape)
    image = np.random.randn(*shape)
    flat = flattening.rect2flat(image)
    image2 = flattening.flat2rect(flat)
    assert_allclose(image, image2)

def test_flattening2():
    """ Make sure that image2flat and flat2image are invertible """
    shape = (120, 160)
    flattening = Flattening.by_rows(shape)
    flat = np.random.randn(shape[0] * shape[1])    
    image = flattening.flat2rect(flat)
    flat2 = flattening.rect2flat(image)
    assert_allclose(flat, flat2)
    
