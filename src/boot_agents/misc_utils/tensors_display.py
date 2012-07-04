from . import np, contract


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
    for i in range(V.shape[2]):
        section.array_as_image('%d' % (i), V[:, :, i],
                               **params)
    pub_stats(section, V)


@contract(V='array[NxN]')
def pub_tensor2_cov(pub, name, V, rcond=None):
    """ Publishes a tensor which is supposed to represent a covariance. """
    params = dict(filter=pub.FILTER_POSNEG, filter_params={})

    section = pub.section(name)
    section.array_as_image('value', V, **params) # TODO
    # TODO: add stats

    with section.plot('svd') as pylab:
        plot_matrix_svd(pylab, V, rcond=rcond)

    pub_stats(section, V)


def plot_matrix_svd(pylab, M, rcond=None):
    u, s, v = np.linalg.svd(M) #@UnusedVariable
    s /= s[0]
    pylab.semilogy(s, 'bx-')
    if rcond is not None:
        pylab.semilogy(np.ones(s.shape) * rcond, 'k--')


@contract(V='array[NxM]')
def pub_tensor2(pub, name, V):
    """ Publishes a generic tensor """
    params = dict(filter=pub.FILTER_POSNEG, filter_params={})
    section = pub.section(name)
    section.array_as_image('value', V, **params) # TODO

    pub_stats(section, V)


@contract(V='array[NxK]')
def pub_tensor2_comp1(pub, name, V):
    """ Publishes a generic tensor, plotting along the last component. """
    section = pub.section(name)
    with section.plot('values') as pylab:
        for i in range(V.shape[1]):
            Vi = V[:, i]
            pylab.plot(Vi)

    pub_stats(section, V)
