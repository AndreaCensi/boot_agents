from . import np, contract
from ..misc_utils import y_axis_balanced
from reprep import MIME_JPG, MIME_PNG, posneg, rgb_zoom
from reprep.plot_utils import (set_thick_ticks, set_left_spines_outward,
    turn_off_left_and_right, turn_off_right, turn_off_bottom_and_top, x_axis_set,
    y_axis_set)
import itertools


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

if False:
    @contract(V='array[HxW]')
    def get_pieces(V, n):
        width = V.shape[0]
        height = V.shape[1]
        wfrac = 1.0 / n
        w = int(np.floor(wfrac * min(width, height)))
        pieces = []
        for s in np.linspace(0, 1, n)[:-1]:
            xfrac = s
            yfrac = s
            x = int(xfrac * w)
            y = int(yfrac * w)
            h = w
            value = V[x:(x + w), y:(y + h)]
            pieces.append(dict(value=value, x=x, y=y, w=w, h=h, wfrac=wfrac,
                               xfrac=xfrac, yfrac=yfrac))
        return pieces
    
    def add_pieces(sub, V, n, zoom=16):
        pieces = get_pieces(V, n)
        for k, piece in enumerate(pieces):
            value = piece['value']
            rgb = posneg(value)
            rgb = rgb_zoom(rgb, zoom)
            subk = sub.section('%d' % k)
            # XXX
            subk.r.data_rgb('png', rgb, mime=MIME_PNG)
            rel = subk.section('rel')
            rel.r.data('x', piece['xfrac'])
            rel.r.data('y', piece['yfrac'])
            rel.r.data('width', piece['wfrac'])
            rel.r.data('height', piece['wfrac'])
            pixels = subk.section('pixels')
            pixels.r.data('x', piece['x'])
            pixels.r.data('y', piece['y'])
            pixels.r.data('width', piece['w'])
            pixels.r.data('height', piece['h'])
         

@contract(V='array[MxNxK]')
def pub_tensor3_slice2(pub, name, V):
    """ Publishes a 3D tensor, with size [nxnxk] """
    section = pub.section(name)
    section.array('value', V)
    
    for i in range(V.shape[2]):
        section.array_as_image('%d' % i, V[:, :, i])

    if False:
        sub = section.section('slices')
    
        max_value = np.max(np.abs(V))
    
        for i in range(V.shape[2]):
            s = sub.section('%d' % i)
    
            tslice = V[:, :, i]
     
            value_rgb = posneg(tslice)
            value = s.section('value')
            pub_save_versions(value, value_rgb)
            
            valuen_rgb = posneg(tslice, max_value=max_value)
            valuen = s.section('valuen', caption='Normalized')
            pub_save_versions(valuen, valuen_rgb)
        
    pub_stats(section, V)


def pub_save_versions(pub, rgb):
    if False:
        jpg_zoom = 1
        pub.r.data_rgb('jpg', rgb_zoom(rgb, jpg_zoom),
                          mime=MIME_JPG, caption='Converted to JPG')
    png_zoom = 1
    pub.r.data_rgb('png', rgb_zoom(rgb, png_zoom),
                          mime=MIME_PNG, caption='Converted to JPG')


@contract(V='array[NxN]')
def pub_tensor2_cov(pub, name, V, rcond=None):
    """ Publishes a tensor which is supposed to represent a covariance. """
    sub = pub.section(name)
    sub.array_as_image('posneg', V) # XXX: redundant, but don't want to change code
    sub.array('value', V)
    rgb = posneg(V)
    pub_save_versions(sub, rgb)
    pub_svd_decomp(sub, V)
    pub_stats(sub, V)


def pub_svd_decomp(parent, V, rcond=None):
    sub1 = parent.section('svd')
    with sub1.plot('plot') as pylab:
        u, s, v = plot_matrix_svd(pylab, V, rcond=rcond)
    sub1.array('sv', s)
    sub1.array('U', u)
    sub1.array('V', v)
    

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
    section = pub.section(name)
    section.array('value', V) # TODO
    value_rgb = posneg(V)
    pub_save_versions(section, value_rgb)
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

        if False:
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

    if False:
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
        
