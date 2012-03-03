from . import np
from .. import diffeomorphism_from_function, diffeomorphism_to_rgb
from ..library import for_all_diffeos


@for_all_diffeos
def creation_suite(fid, f):
    shape = [50, 50]
    D = diffeomorphism_from_function(shape, f)
    from reprep import Report

    name = f.__name__
    r = Report(name)
    fig = r.figure()
    with r.plot('grid') as pylab:
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





