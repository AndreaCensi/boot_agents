from . import np, contract
from ..misc_utils import y_axis_balanced
from reprep.plot_utils import turn_off_bottom_and_top, x_axis_set, y_axis_set
import itertools
from reprep.plot_utils.spines import set_thick_ticks, set_left_spines_outward, \
    turn_off_left_and_right, turn_off_right
from reprep.plot_utils.axes import plot_horizontal_line


def pub_text_stats(pub, V):
    """ Create statistics for the tensor """
    V = np.array(V.flat)
    stats = ""
    # XXX: remove nan
    stats += "Min: %s\n" % np.nanmin(V)
    stats += "Mean: %s\n" % np.mean(V)
    stats += "Median: %s\n" % np.median(V)
    stats += "Max: %s\n" % np.nanmax(V)

    def count(x):
        # x: bool
        tot = x.size
        num = np.sum(x)
        perc = '%.2f%%' % (100.0 * num / tot)
        return '%s (%s/%s)' % (perc, num, tot)

    stats += "Num>0: %s \n" % count(V > 0)
    stats += "Num=0: %s \n" % count(V == 0)
    stats += "Num<0: %s \n" % count(V < 0)
    pub.text('stats', stats)


def pub_stats(pub, V):
    pub_text_stats(pub, V)


@contract(V='array[MxNxK]')
def pub_tensor3_slice2(pub, name, V):
    """ Publishes a 3D tensor, with size [nxnxk] """
    params = dict(filter=pub.FILTER_POSNEG, filter_params={})

    section = pub.section(name)
    section.array('value', V)
    
    sub = section.section('slices')
    #    y_min = np.min(V)
    #    y_max = np.max(V)

    for i in range(V.shape[2]):
        s = sub.section('%d' % i)
        # TODO: do not save value
        s.array_as_image('value', V[:, :, i], **params)
        # TODO: make one normalized and one not
        
    pub_stats(section, V)


@contract(V='array[NxN]')
def pub_tensor2_cov(pub, name, V, rcond=None):
    """ Publishes a tensor which is supposed to represent a covariance. """
    params = dict(filter=pub.FILTER_POSNEG, filter_params={})

    sub = pub.section(name)
    sub.array_as_image('value', V, **params) # TODO

    sub1 = sub.section('svd')
    with sub1.plot('plot') as pylab:
        u, s, v = plot_matrix_svd(pylab, V, rcond=rcond)
        
    sub1.array('sv', s)
    sub1.array('U', u)
    sub1.array('V', v)
        
    pub_stats(sub, V)


def plot_matrix_svd(pylab, M, rcond=None):
    u, s, v = np.linalg.svd(M) #@UnusedVariable
    sn = s / s[0]
    pylab.semilogy(sn, 'bx-')
    if rcond is not None:
        pylab.semilogy(np.ones(sn.shape) * rcond, 'k--')
    return u, s, v


@contract(V='array[NxM]')
def pub_tensor2(pub, name, V):
    """ Publishes a generic 2D tensor """
    params = dict(filter=pub.FILTER_POSNEG, filter_params={})
    section = pub.section(name)
    section.array_as_image('value', V, **params) # TODO

    pub_stats(section, V)


class BV1Style:
    figsize = (6, 3)
    dots_format = dict(linestyle='None', marker='.', color='k', markersize=1.5)
    line_format = dict(linestyle='-', color='k', markersize=1.0)
    
    
@contract(V='array[NxK]')
def pub_tensor2_comp1(pub, name, V):
    """ Publishes a generic NxK tensor, plotting along the last component. """
    section = pub.section(name)
    section.array('value', V)
    
    sub = section.section('slices')
#    y_min = np.min(V) * 1.1
#    y_max = np.max(V) * 1.1
    
    figsize = BV1Style.figsize
    dots_format = BV1Style.dots_format
    line_format = BV1Style.line_format
    
    nsensels = V.shape[0]
    y_max = np.max(np.abs(V))
    for i in range(V.shape[1]):
        sub2 = sub.section('%d' % i)
        sub2.array('value', V[:, i])
        with sub2.plot('plot') as pylab:
            pylab.plot(V[:, i])
            #            y_axis_set(pylab, y_min, y_max)
            #            x_axis_set(pylab, -1, V.shape[0])
            #            turn_off_bottom_and_top(pylab)

        with sub2.plot('plotB', figsize=figsize) as pylab:
            pylab.plot(V[:, i], **dots_format)
            style_1d_sensel_func(pylab, n=nsensels,
                                 y_max=np.max(np.abs(V[:, i])))
        with sub2.plot('plotBn', figsize=figsize) as pylab:
            pylab.plot(V[:, i], **dots_format)
            style_1d_sensel_func(pylab, n=nsensels, y_max=y_max)

        with sub2.plot('plotBna', figsize=figsize) as pylab:
            pylab.plot(V[:, i], **dots_format)
            style_1d_sensel_func(pylab, n=nsensels, y_max=y_max)
            # no left axis
            turn_off_left_and_right(pylab)

        with sub2.plot('plotC', figsize=figsize) as pylab:
            pylab.plot(V[:, i], **line_format)
            style_1d_sensel_func(pylab, n=nsensels,
                                 y_max=np.max(np.abs(V[:, i])))
        with sub2.plot('plotCn', figsize=figsize) as pylab:
            pylab.plot(V[:, i], **dots_format)
            style_1d_sensel_func(pylab, n=nsensels, y_max=y_max)

        with sub2.plot('plotCna', figsize=figsize) as pylab:
            pylab.plot(V[:, i], **dots_format)
            style_1d_sensel_func(pylab, n=nsensels, y_max=y_max)
            # no left axis
            turn_off_left_and_right(pylab)
    
    # And now all together
    dots_format_all = dict(**dots_format)
    if 'color' in dots_format_all:
        del dots_format_all['color']
    line_format_all = dict(**dots_format)
    if 'color' in line_format_all:
        del line_format_all['color']
    
    with sub.plot('plot', figsize=figsize) as pylab:
        for i in range(V.shape[1]):
            pylab.plot(V[:, i])
        style_1d_sensel_func(pylab, n=nsensels, y_max=np.max(np.abs(V)))

    with sub.plot('plotB', figsize=figsize) as pylab:
        for i in range(V.shape[1]):
            pylab.plot(V[:, i], **dots_format_all)
        style_1d_sensel_func(pylab, n=nsensels, y_max=y_max)

    with sub.plot('plotBa', figsize=figsize) as pylab:
        for i in range(V.shape[1]):
            pylab.plot(V[:, i], **dots_format_all)
        style_1d_sensel_func(pylab, n=nsensels, y_max=y_max)
        # no left axis
        turn_off_left_and_right(pylab)
        
    with sub.plot('plotC', figsize=figsize) as pylab:
        for i in range(V.shape[1]):
            pylab.plot(V[:, i], **line_format_all)
        style_1d_sensel_func(pylab, n=nsensels, y_max=y_max)

    with sub.plot('plotCa', figsize=figsize) as pylab:
        for i in range(V.shape[1]):
            pylab.plot(V[:, i], **line_format_all)
        style_1d_sensel_func(pylab, n=nsensels, y_max=y_max)
        # no left axis
        turn_off_left_and_right(pylab)
                                                     
    pub_stats(section, V)


def style_1d_sensel_func(pylab, n, y_max, extra_vspace=1.1):
    """ 
        Decorates approapriateyle
    """
    y_axis_set(pylab, -y_max * extra_vspace, y_max * extra_vspace)
    x_axis_set(pylab, -1, n)
    turn_off_bottom_and_top(pylab)
    turn_off_right(pylab)
    set_left_spines_outward(pylab, offset=10)
    set_thick_ticks(pylab, markersize=3, markeredgewidth=1)
    pylab.plot([0, n - 1], [0, 0], '--', color=[0.7, 0.7, 0.7])


def iterate_indices(shape):
    if len(shape) == 2:
        for i, j in itertools.product(range(shape[0]), range(shape[1])):
            yield i, j
    else:
        assert(False)


@contract(G='array[AxBxHxW]',
          xlabels='list[A](str)', ylabels='list[B](str)')
def display_4d_tensor(pub, name, G, xlabels, ylabels):
    A = G.shape[0]
    B = G.shape[1]
    section = pub.section(name, cols=A)
    for b, a in iterate_indices((B, A)):
        value = G[a, b, :, :].squeeze()
        label = '%s_%s_%s' % (name, xlabels[a], ylabels[b])
        section.array_as_image(label, value)


@contract(G='array[AxHxW]', labels='list[A](str)')
def display_3d_tensor(pub, name, G, labels):
    A = G.shape[0]
    section = pub.section(name, cols=A)
    for a in range(A):
        value = G[a, :, :].squeeze()
        label = '%s_%s' % (name, labels[a])
        section.array_as_image(label, value)


@contract(value='array[Kx1xN]|array[KxN]')
def display_1d_tensor(pub, name, value):
    with pub.plot(name) as pylab:
        for k in range(value.shape[0]):
            x = value[k, ...].squeeze()
            assert x.ndim == 1
            pylab.plot(x, label='%s%s' % (name, k))
            x_axis_set(pylab, -1, x.size)
            turn_off_bottom_and_top(pylab)
            
        y_axis_balanced(pylab, show0=True)
        pylab.legend()


@contract(value='array[N]')
def display_1d_field(pub, name, value):
    with pub.plot(name) as pylab:
        pylab.plot(value)
        x_axis_set(pylab, -1, value.size)
        turn_off_bottom_and_top(pylab)
        
