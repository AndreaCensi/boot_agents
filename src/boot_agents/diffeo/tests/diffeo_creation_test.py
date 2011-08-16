import numpy as np
from numpy.testing.utils import assert_allclose
from boot_agents.diffeo import diffeomorphism_from_function
from reprep import Report
from boot_agents.diffeo import diffeomorphism_to_rgb

def f_identity(X): 
    return X

def f_pow3(X):
    return np.power(X[0], 3), np.power(X[1], 3)

def f_pow3x(X):
    return np.power(X[0], 3), X[1]

def f_pow3x_inv(X):
    A = np.abs(X[0])
    S = np.sign(X[0])
    return S * np.power(A, 1 / 3.0), X[1] 

def mod1d(x):
    ''' bounds in [-1,1] '''
    return np.fmod(np.fmod(x + 1, 2), 2) - 1

def mod(X):
    ''' bounds in [-1,+1]x[-1,+1] '''
    return mod1d(X[0]), mod1d(X[1])
    
def f_rotx(X):
    return mod((X[0] + 0.1, X[1]))
           
def f_roty(X):
    return mod((X[0], X[1] + 0.1))
                
diffeomorphisms = [f_identity, f_pow3, f_pow3x, f_pow3x_inv, f_rotx, f_roty]

def mod_test():
    assert_allclose(mod1d(0), 0)
    assert_allclose(mod1d(1), -1)
    assert_allclose(mod1d(2), 0)
    assert_allclose(mod1d(-1), -1)
    assert_allclose(mod1d(1.1), -0.9)
    assert_allclose(mod1d(0.1), 0.1)
    assert_allclose(mod1d(-0.1), -0.1)
    
def diffeo_creation_tests():
    for f in diffeomorphisms:
        diffeo_creation_suite(f)


def diffeo_creation_suite(f):
    shape = [50, 50]
    D = diffeomorphism_from_function(shape, f)
    
    name = f.__name__
    r = Report(name)
    fig = r.figure()
    with r.data_pylab('grid') as pylab:
        grids = 10
        N = 100
        
        def plot_line(px, py):
            k = px.size
            for k in range(N):
                p = f((px[k], py[k]))
                px[k] = p[0]
                py[k] = p[1]
            pylab.plot(px, py, 'b.')
            
        for x in np.linspace(-1, +1, grids):
            px = np.ones(N) * x
            py = np.linspace(-1, +1, N)
            plot_line(px, py)
            
        for y in np.linspace(-1, +1, grids):
            py = np.ones(N) * y
            px = np.linspace(-1, +1, N)
            plot_line(px, py)
           
        pylab.axis((-1.1, 1.1, -1.1, 1.1))
        pylab.axis('equal')
    r.last().add_to(fig)
        
    rgb = diffeomorphism_to_rgb(D)
    r.data_rgb('diffeomorphism_rgb', rgb).add_to(fig)
    filename = 'out/diffeo_creation_suite/%s.html' % name
    print('Writing to %r.' % filename)
    r.to_html(filename)
    
    
