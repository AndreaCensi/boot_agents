from . import Action
from .. import Diffeomorphism2D, diffeo_from_function, diffeo_inverse
from ..tests import f_rotx
from .compress import actions_compress
from numpy.testing.utils import assert_allclose # XXX


def action_test_1():
    shape = (20, 20)
    d_rotx = diffeo_from_function(shape, f_rotx)

    d1 = Diffeomorphism2D(d_rotx)
    d2 = Diffeomorphism2D(diffeo_inverse(d_rotx))
    a1 = Action(d1, label='rotx', primitive=True,
                invertible=False, original_cmd='rotx')
    a2 = Action(d2, label='rotx_inv', primitive=True,
                invertible=False, original_cmd='rotx_inv')

    assert_allclose(Action.similarity(a1, a1), +1)
    assert_allclose(Action.similarity(a1, a2), -1)
    assert_allclose(Action.similarity(a2, a1), -1)
    assert_allclose(Action.similarity(a2, a2), +1)


def action_compress_test_1():
    shape = (20, 20)
    d_rotx = diffeo_from_function(shape, f_rotx)
    d1 = Diffeomorphism2D(d_rotx)
    d2 = Diffeomorphism2D(diffeo_inverse(d_rotx))
    a1 = Action(d1, label='rotx', primitive=True,
                invertible=False, original_cmd='rotx')
    a2 = Action(d2, label='rotx_inv', primitive=True,
                invertible=False, original_cmd='rotx_inv')

    threshold = 0.999

    actions2, info = actions_compress([a1], threshold) #@UnusedVariable
    assert_allclose(len(actions2), 1)
    assert(not actions2[0].invertible)

    actions2, info = actions_compress([a1, a2], threshold) #@UnusedVariable
    assert_allclose(len(actions2), 1)
    assert(actions2[0].invertible)

