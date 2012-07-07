from . import np, contract
from reprep.plot_utils import y_axis_set


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
    y_min = np.min(V)
    y_max = np.max(V)

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


@contract(V='array[NxK]')
def pub_tensor2_comp1(pub, name, V):
    """ Publishes a generic NxK tensor, plotting along the last component. """
    section = pub.section(name)
    section.array('value', V)
    
    sub = section.section('slices')
    y_min = np.min(V)
    y_max = np.max(V)
    
    for i in range(V.shape[1]):
        sub2 = sub.section('%d' % i)
        sub2.array('value', V[:, i])
        with sub2.plot('plot') as pylab:
            pylab.plot(V[:, i])
            y_axis_set(pylab, y_min, y_max)
        
    with sub.plot('plot') as pylab:
        for i in range(V.shape[1]):
            Vi = V[:, i]
            pylab.plot(Vi)

    pub_stats(section, V)
