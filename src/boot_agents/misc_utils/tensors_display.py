from boot_agents.misc_utils import y_axis_balanced
from contracts import contract
from reprep import MIME_JPG, posneg, rgb_zoom, MIME_PNG
from reprep.plot_utils import (set_thick_ticks, set_left_spines_outward,
    turn_off_left_and_right, turn_off_right, turn_off_bottom_and_top, x_axis_set,
    y_axis_set)
import itertools
import numpy as np
from reprep.graphics.filter_posneg import posneg_hinton
 

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
            with sub.subsection('%d' % k) as subk:
                value = piece['value']
                rgb = posneg(value)
                rgb = rgb_zoom(rgb, zoom)
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
    """ Displays a 3D tensor, with shape [nxnxk] """
    with pub.subsection(name) as section:
        section.array('value', V)
        
        nslices = V.shape[2]
        max_value = np.nanmax(np.abs(V))
        
        if not np.isfinite(max_value):
            msg = 'Expected a finite value.'
            raise ValueError(msg)
        
        fu = section.figure('unnormalized', cols=nslices)
        fn = section.figure('normalized', cols=nslices)
        fug = section.figure('unnormalizedg', cols=nslices)
        fng = section.figure('normalizedg', cols=nslices)
         
        with section.subsection('slices') as slices:        
            
            for i in range(nslices):
                tslice = V[:, :, i]
                
                with slices.subsection('%d' % i) as s:
                    
                    s.array('value', tslice)
             
                    rgbu = posneg(tslice)
                    rgbn = posneg(tslice, max_value=max_value)
    
                    dn = pub_save_versions2(s, 'normalized', rgbn)
                    fn.sub(dn)
    
                    du = pub_save_versions2(s, 'unnormalized', rgbu)
                    fu.sub(du)
                
                    gray_u = posneg_hinton(tslice)
                    gray_n = posneg_hinton(tslice, max_value=max_value)

                    dng = pub_save_versions2(s, 'normalizedg', gray_n)
                    fug.sub(dng)
                    dug = pub_save_versions2(s, 'unnormalizedg', gray_u)
                    fng.sub(dug)
    
    pub_stats(section, V)


def pub_save_versions2(sub, name, rgb):
    with sub.subsection(name) as s:
        d = pub_save_versions(s, rgb)
    return d
    

def pub_save_versions(pub, rgb, png_zoom=4):
    if False:
        jpg_zoom = 1
        pub.data_rgb('jpg', rgb_zoom(rgb, jpg_zoom),
                          mime=MIME_JPG, caption='Converted to JPG')
    d = pub.data_rgb('png', rgb_zoom(rgb, png_zoom), mime=MIME_PNG)
    return d

@contract(V='array[NxN]')
def pub_tensor2_cov(pub, name, V, rcond=None):
    """ Publishes a tensor which is supposed to represent a covariance. """
    with pub.subsection(name) as sub:
        sub.array_as_image('posneg', V)  # XXX: redundant, but don't want to change code
        sub.array('value', V)
        
        rgb = posneg(V)
        pub_save_versions(sub, rgb)
        
        pub_svd_decomp(sub, V, rcond=rcond)
        pub_stats(sub, V)


def pub_svd_decomp(parent, V, rcond=None):
    with parent.subsection('svd') as sub1:
        f = sub1.figure()
        with f.plot('plot') as pylab:
            u, s, v = plot_matrix_svd(pylab, V, rcond=rcond)
        sub1.array('sv', s)
        sub1.array('U', u)
        sub1.array('V', v)
        

def plot_matrix_svd(pylab, M, rcond=None):
    u, s, v = np.linalg.svd(M)  # @UnusedVariable
    sn = s / s[0]
    pylab.semilogy(sn, 'bx-')
    if rcond is not None:
        pylab.semilogy(np.ones(sn.shape) * rcond, 'k--')
    return u, s, v


@contract(V='array[NxM]')
def pub_tensor2(pub, name, V):
    """ Publishes a generic 2D tensor """
    with pub.subsection(name) as section:
        section.array('value', V)  # TODO
        value_rgb = posneg(V)
        pub_save_versions(section, value_rgb)
        pub_stats(section, V)


class BV1Style:
    figsize = (6, 3)
    dots_format = dict(linestyle='None', marker='.', color='k', markersize=1.5)
    line_format = dict(linestyle='-', color='k', markersize=1.0)
    
    
@contract(V='array[NxK]')
def pub_tensor2_comp1(pub, name, V):
    """ 
        Publishes a generic NxK tensor, plotting along the last component.
        Assumes that it can be either pos or neg.
    """
    with pub.subsection(name) as section:
        section.array('value', V)
        
        with section.subsection('slices') as sub:
            f = sub.figure() 
            
            figsize = BV1Style.figsize
            dots_format = BV1Style.dots_format
            line_format = BV1Style.line_format
            
            nsensels = V.shape[0]
            y_max = np.max(np.abs(V))
            for i in range(V.shape[1]):
                sub2 = sub.section('%d' % i)
                sub2.array('value', V[:, i])
                
                f2 = sub2.figure()
                
                with f2.plot('plot') as pylab: 
                    pylab.plot(V[:, i], '.')
                    style_1d_sensel_func(pylab, n=nsensels, y_max=np.max(np.abs(V[:, i])))
                    
                # This is scaled with the maximum of all slices
                with f2.plot('plot_scaled') as pylab: 
                    pylab.plot(V[:, i], '.')
                    style_1d_sensel_func(pylab, n=nsensels, y_max=np.max(np.abs(V)))
                
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
            
            with f.plot('plot', figsize=figsize) as pylab:
                for i in range(V.shape[1]):
                    pylab.plot(V[:, i], '.')
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
        raise NotImplementedError()
        assert(False)


@contract(G='array[AxBxHxW]',
          xlabels='list[A](str)', ylabels='list[B](str)')
def display_4d_tensor(pub, name, G, xlabels, ylabels):
    A = G.shape[0]
    B = G.shape[1]
    with pub.subsection(name, cols=A) as section:
        for b, a in iterate_indices((B, A)):
            value = G[a, b, :, :].squeeze()
            label = '%s_%s_%s' % (name, xlabels[a], ylabels[b])
            section.array_as_image(label, value)


@contract(G='array[AxHxW]', labels='list[A](str)')
def display_3d_tensor(pub, name, G, labels):
    A = G.shape[0]
    with pub.subsection(name, cols=A) as section:
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
        
