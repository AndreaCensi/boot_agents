from boot_agents.misc_utils import plot_matrix_svd
from geometry import distances_from_angles
from reprep import Report
from vehicles.library.sensors.utils import get_uniform_directions
import numpy as np


def main():
    theta = get_uniform_directions(360, 180)
    D = distances_from_angles(theta)
    
    def identity(x):
        return x
    
    def exp1(x):
        return np.exp(-x)
    
    def exp2(x):
        return np.exp(-5 * x)

    def exp3(x):
        return np.exp(-x * x)

    def p1(x):
        return np.cos(2 * np.pi - x)

    functions = [identity, exp1, exp2, exp3, p1]
    
    r = Report()
    for function in functions:
        name = function.__name__
        section = r.section(name)
        f = section.figure()
        
        M = function(D)
        sigma = 0.0001
        D1 = np.maximum(0, D + sigma * np.random.randn(*D.shape))
        M1 = function(D1)
        
        with f.plot('svd') as pylab:
            plot_matrix_svd(pylab, M)
        with f.plot('svd1', caption='Perturbed') as pylab:
            plot_matrix_svd(pylab, M1)

    out = 'check_rank.html'
    print('Writing to %r' % out)
    r.to_html(out)
    
if __name__ == '__main__':
    main()

    
