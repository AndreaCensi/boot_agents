from . import np
from .. import diffeomorphism_from_function, diffeomorphism_to_rgb
from ..library import for_all_diffeos
from reprep.plot_utils import turn_all_axes_off


@for_all_diffeos
def creation_suite(fid, f): #@UnusedVariable
    shape = [50, 50]
    D = diffeomorphism_from_function(shape, f)
    from reprep import Report

    name = f.__name__
    r = Report(name)
    fig = r.figure()

    M = 1
    n1 = 10
    n2 = 100
    bx = [-.99, +.99] # depends on fmod
    by = bx
    params = dict(figsize=(3, 3))

    def common_settings(pylab):
        pylab.axis('equal')
        pylab.axis((-M, M, -M, M))
        turn_all_axes_off(pylab)

    with fig.plot('grid1', **params) as pylab:
        curved = CurvedPylab(pylab, f)
        plot_grid(curved, n1=n1, n2=n2, bx=bx, by=by, hcol='k-', vcol='k-')
        common_settings(pylab)

    with fig.plot('grid2', caption="different colors", **params) as pylab:
        plot_grid(curved, n1=n1, n2=n2, bx=bx, by=by, hcol='r-', vcol='b-')
        common_settings(pylab)

    with fig.plot('grid3', caption="smiley", **params) as pylab:
        plot_grid(curved, n1=n1, n2=n2, bx=bx, by=by, hcol='r-', vcol='b-')
        common_settings(pylab)
        plot_smiley(curved)

    with fig.plot('grid4', caption="smiley", **params) as pylab:
        plot_grid(curved, n1=n1, n2=n2, bx=bx, by=by, hcol='k-', vcol='k-')
        common_settings(pylab)
        plot_smiley(curved, '0.5')

    rgb = diffeomorphism_to_rgb(D)
    r.data_rgb('diffeomorphism_rgb', rgb).add_to(fig)
    filename = 'out/diffeo_creation_suite/%s.html' % name
    print('Writing to %r.' % filename)
    r.to_html(filename)


def plot_grid(pylab, n1=10, n2=100, bx=[-1, .99], by=[-1, .99],
              hcol='k-', vcol='k-'):

    for x in np.linspace(bx[0], bx[1], n1):
        px = np.ones(n2) * x
        py = np.linspace(by[0], by[1], n2)
        pylab.plot(px, py, vcol)

    for y in np.linspace(by[0], by[1], n1):
        py = np.ones(n2) * y
        px = np.linspace(bx[0], bx[1], n2)
        pylab.plot(px, py, hcol)


def plot_smiley(pylab, col='0.8'):
    d = 0.55
    e = 0.55
    r = 0.3
    plot_circle(pylab, 0, -0.1, 0.15, ec='none', fc=col)
    plot_circle(pylab, -d, e, r, ec='none', fc=col)
    plot_circle(pylab, +d, e, r, ec='none', fc=col)

    delta = r / 3
    r2 = r / 2
    plot_circle(pylab, -d + delta, e, r2, ec='none', fc='white')
    plot_circle(pylab, +d + delta, e, r2, ec='none', fc='white')

    f = 0.75
    x = np.linspace(-f, +f, 100)
    y = -0.3 - 0.5 * np.cos(x)
    pylab.fill(x, y, ec='none', fc=col)


def plot_circle(pylab, x, y, r, *args, **kwargs):
    N = 100
    theta = np.linspace(0, np.pi * 2, N)
    X = x + r * np.cos(theta)
    Y = y + r * np.sin(theta)
    #pylab.plot(X, Y, *args, **kwargs)
    pylab.fill(X, Y, *args, **kwargs)


class CurvedPylab:
    # TODO: interpolate if too sparse
    def __init__(self, pylab, f):
        self.pylab = pylab
        self.f = f

    def get_coords(self, px, py):
        px = px.copy()
        py = py.copy()
        for k in range(px.size):
            p = self.f((px[k], py[k]))
            px[k] = p[0]
            py[k] = p[1]
        return px, py

    def plot(self, px, py, *args, **kwargs):
        px, py = self.get_coords(px, py)
        return self.pylab.plot(px, py, *args, **kwargs)

    def fill(self, px, py, *args, **kwargs):
        px, py = self.get_coords(px, py)
        return self.pylab.fill(px, py, *args, **kwargs)

