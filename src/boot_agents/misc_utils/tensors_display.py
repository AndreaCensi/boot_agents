from . import np, contract


@contract(V='array[MxNxK]')
def pub_tensor3_slice2(pub, name, V):
    """ Publishes a 3D tensor, with size [nxnxk] """
    params = dict(filter=pub.FILTER_POSNEG, filter_params={})

    section = pub.section(name)
    for i in range(V.shape[2]):
        section.array_as_image('%d' % (i), V[:, :, i],
                               **params)
    # TODO: add stats


@contract(V='array[NxN]')
def pub_tensor2_cov(pub, name, V):
    """ Publishes a tensor which is supposed to represent a covariance. """
    params = dict(filter=pub.FILTER_POSNEG, filter_params={})

    section = pub.section(name)
    section.array_as_image('value', V, **params) # TODO
    # TODO: add stats


@contract(V='array[NxM]')
def pub_tensor2(pub, name, V):
    """ Publishes a generic tensor """
    params = dict(filter=pub.FILTER_POSNEG, filter_params={})
    section = pub.section(name)
    section.array_as_image('value', V, **params) # TODO


@contract(V='array[NxK]')
def pub_tensor2_comp1(pub, name, V):
    """ Publishes a generic tensor, plotting along the last component. """
    section = pub.section(name)
    with section.plot('values') as pylab:
        for i in range(V.shape[1]):
            Vi = V[:, i]
            pylab.plot(Vi)

