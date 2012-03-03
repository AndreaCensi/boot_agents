''' Some utilities for testing. '''
from boot_agents.diffeo.tests.utils.generation import fancy_test_decorator
from . import DiffeoLibrary


for_all_diffeos = fancy_test_decorator(
        lister=lambda: list(DiffeoLibrary.diffeos.keys()),
        arguments=lambda fid: (fid, DiffeoLibrary.diffeos[fid]),
        attributes=lambda fid: dict(diffeo=fid))

for_all_diffeo_pairs = fancy_test_decorator(
        lister=lambda: list(DiffeoLibrary.get_invertible_diffeos()),
        arguments=lambda pair: (pair[0].__name__, pair[0],
                                pair[1].__name__, pair[1]),
        naming=lambda pair: '%s-%s' % (pair[0].__name__, pair[1].__name__))


@for_all_diffeos
def check_diffeo(id_diffeo, diffeo):
    """ Just a simple listing of the available functions. """
    print('Testing %s (%s)' % (id_diffeo, diffeo))


@for_all_diffeo_pairs
def check_diffeo_pairs(id_f, f, id_f_inv, f_inv):
    """ Just a simple listing of the available pairs. """
    print('Testing %s and %s (%s, %s)' % (id_f, id_f_inv, f, f_inv))
