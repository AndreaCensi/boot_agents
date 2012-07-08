from . import  contract, np
from .. import (DiffeomorphismEstimator, diffeomorphism_from_function,
    coords_iterate, diffeomorphism_to_rgb_cont, MATCH_CONTINUOUS)
import time
from ..library import for_all_diffeos


@contract(diffeo='valid_diffeomorphism,array[MxNx2]', y='array[MxN]',
          returns='array[MxN]')
def apply_diffeomorphism(diffeo, y):
    yd = np.empty_like(y)
    for c in coords_iterate(y.shape):
        c2 = tuple(diffeo[c[0], c[1], :])
        yd[c] = y[c2]
    return yd


@contract(shape='valid_2d_shape', K='int', diffeo='valid_diffeomorphism')
def generate_input(shape, K, diffeo, epsilon=0.5):
    t0 = time.clock()
    for k in range(K): #@UnusedVariable
        y = np.random.randn(*shape)
        yd = apply_diffeomorphism(diffeo, y)
        y1 = yd + epsilon * np.random.rand(*shape)

        yield y, y1
    t1 = time.clock()
    print('%.2f fps' % (K / (t1 - t0)))


@for_all_diffeos
def estimation(fid, f): #@UnusedVariable
    shape = [50, 50]
    diffeo = diffeomorphism_from_function(shape, f)

    K = 50
    epsilon = 1
    de = DiffeomorphismEstimator([0.2, 0.2], MATCH_CONTINUOUS)
    for y0, y1 in generate_input(shape, K, diffeo, epsilon=epsilon):
        de.update(y0, y1)

    diff2d = de.summarize()
    diffeo_learned = diff2d.d

    from reprep import Report

    name = f.__name__
    r = Report(name)
    fig = r.figure(cols=4)

    diffeo_learned_rgb = diffeomorphism_to_rgb_cont(diffeo_learned)
    diffeo_rgb = diffeomorphism_to_rgb_cont(diffeo)
    r.data_rgb('diffeo_rgb', diffeo_rgb).add_to(fig)
    r.data_rgb('diffeo_learned_rgb', diffeo_learned_rgb).add_to(fig)
    L = r.data('diffeo_learned_uncertainty', diff2d.variance)
    L.display('scale').add_to(fig, caption='uncertainty')
    r.data('last_y0', y0).display('scale').add_to(fig, caption='last y0')
    r.data('last_y1', y1).display('scale').add_to(fig, caption='last y1')

    cs = [(0, 25), (10, 25), (25, 25), (25, 5)]
    for c in cs:
        M25 = de.get_similarity(c)
        r.data('cell-%s-%s' % c, M25).display('scale').add_to(fig,
                                         caption='Example similarity field')

    filename = 'out/diffeo_estimation_suite/%s.html' % name
    print('Writing to %r.' % filename)
    r.to_html(filename)



