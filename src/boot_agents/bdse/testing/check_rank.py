from boot_agents.misc_utils import plot_matrix_svd
from geometry import distances_from_angles
from reprep import Report
import numpy as np
from astatsa.utils import assert_allclose
from contracts import contract


@contract(fov_deg='>0', num_sensels='N,>1', returns='array[N](>=-pi,<=pi)')
def get_uniform_directions(fov_deg, num_sensels):
    """ Returns a set of directions uniform in space """
    if fov_deg == 360:
        ray_dist = 2 * np.pi / (num_sensels)
        directions = np.linspace(-np.pi + ray_dist / 2,
                                 + np.pi - ray_dist + ray_dist / 2,
                                 num_sensels)

        assert_allclose(directions[-1] - directions[0], 2 * np.pi - ray_dist)

        t = np.rad2deg(directions)
        a = t[1:] - t[:-1]
        b = t[0] - t[-1] + 360
        
        
        assert_allclose(a[0], b)

    else:
        fov_rad = np.radians(fov_deg)
        directions = np.linspace(-fov_rad / 2, +fov_rad / 2, num_sensels)

        assert_allclose(directions[-1] - directions[0], fov_rad)
    assert len(directions) == num_sensels
    return directions


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

    
