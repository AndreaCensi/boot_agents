from . import np, contract, DiffeoLibrary
import functools
from contracts import new_contract

__all__ = ['diffeo_torus', 'diffeo_torus_reflection']

new_contract('coordinates_can', 'array[2x...],array(>=-1,<=1),shape(x)')
new_contract('coordinates_tuple', 'seq[2](>=-1,<=+1)')
new_contract('coordinates', 'coordinates_can|coordinates_tuple')


def mod1d(x):
    ''' bounds in [-1,1] '''
    return np.fmod(np.fmod(x + 1, 2) + 2, 2) - 1


@contract(X='seq[2](float)|array[2x...]', returns='coordinates_can')
def mod_toroidal(X):
    ''' Bounds in [-1,+1]x[-1,+1] '''
    if len(X) == 0:
        raise ValueError('Invalid value %s' % X)
    return np.array([mod1d(X[0]), mod1d(X[1])])


def diffeo_torus_reflection(f):
    """ Declares this is a diffeomorphis which is its own inverse. """
    def f_inv(X):
        return f(X)
    f_inv.__name__ = '%s_inv' % f.__name__
    diffeo_torus(f_inv)
    return diffeo_torus(f)


def diffeo_torus(f):
    ''' 
        Adds some checks to the function f, and puts it in the library
        index.
    '''
    @contract(X='coordinates', returns='coordinates_can') # XXX: shape?
    @functools.wraps(f)
    def wrapper(X):
        X = np.array(X) # if tuple
        Y = f(X)
        #print('Y: %s' % Y)
        return mod_toroidal(Y)
    DiffeoLibrary.diffeos[f.__name__] = wrapper
    wrapper.__name__ = f.__name__
    return wrapper
