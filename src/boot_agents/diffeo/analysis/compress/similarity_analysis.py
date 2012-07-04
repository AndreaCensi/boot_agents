from . import (actions_compress, actions_commutators,
    actions_remove_similar_to_identity)
from .. import imread, resize
from ... import diffeo_compose, diffeo_inverse, diffeo_apply
from optparse import OptionParser
import os
import numpy as np
import itertools
from geometry import formatm # XXX: dep


def zoom(M, factor=10):
    return np.kron(M, np.ones((factor, factor)))


def describe_info(sec, actions, info):
    Distance = info['Distance']
    Distance_to_inverse = info['Distance_to_inverse']
    np.fill_diagonal(Distance, np.nan)
    np.fill_diagonal(Distance_to_inverse, np.nan)
    max_value = max(np.nanmax(Distance), np.nanmax(Distance_to_inverse))
    min_value = max(np.nanmin(Distance), np.nanmin(Distance_to_inverse))
    params = dict(filter='scale', filter_params={'min_value': min_value / 1.5,
                                            'max_value': max_value
                                            })

    sec.array_as_image('Distance', zoom(Distance), **params)
    sec.array_as_image('DistanceInv', zoom(Distance_to_inverse), **params)

    n = len(actions)
    actual_inverse = np.zeros((n, n))
    for i, j in itertools.product(range(n), range(n)):
        u_i = actions[i].original_cmd
        u_j = actions[j].original_cmd

        actual = np.allclose(u_i, -u_j)
        actual_inverse[i, j] = 1 if actual else 0

    sec.array_as_image('Actual', zoom(actual_inverse))

    sec.text('DistanceS', formatm('distance', Distance, format_str='%.3f'))
    sec.text('DistanceInvS',
             formatm('distance_inv', Distance_to_inverse, format_str='%.3f'))

    s = ""
    for action in actions:
        s += '%s\n' % action
    sec.text('list_of_actions', s)


def similarity_analysis(global_options, data, args):
    usage = ""
    parser = OptionParser(usage=usage)
    parser.disable_interspersed_args()
    parser.add_option("--template", default="lena.jpg")
    (options, args) = parser.parse_args(args)
    if args:
        raise Exception('Extra args')

    if not os.path.exists(options.template):
        raise Exception('Template %r not found.' % options.template)

    cmds = data['actions']
    cmds = sorted(cmds, key=lambda x: '%s' % x.original_cmd)
    publisher = data['publisher']

    print('Compressing %d actions' % len(cmds))
    cmds2, info2 = actions_compress(cmds, threshold=0.9974) #@UnusedVariable
    describe_info(publisher.section('cmd2'), cmds, info2)

    print('After compressing, we have %d actions' % len(cmds2))
    cmds3 = actions_commutators(cmds2)
    print('With commutators, we have %d actions' % len(cmds3))
    cmds4 = actions_remove_similar_to_identity(cmds3, threshold=0.01)
    print('After removing similar, we have %s' % len(cmds4))
    print('Now compressing commutators...')
    cmds5, info5 = actions_compress(cmds3, threshold=0.9974) #@UnusedVariable
    print('After compressing, we have %s' % len(cmds5))

    # load template
    template_name = options.template
    template = imread(template_name)
    example_diffeo = cmds[0].diffeo.d
    width = example_diffeo.shape[1] # note inverted
    height = example_diffeo.shape[0]
    template = resize(template, width, height)
    publisher.array_as_image('template', template)

    for cmd in cmds:
        break
        if not cmd.primitive:
            continue
        print('Plotting %s' % cmd.label)
        section_name = '%s-%s-%s' % (cmd, cmd.label, cmd.original_cmd)
        s = publisher.section(section_name)
        d = cmd.diffeo.d
        d2 = diffeo_compose(d, d)
        d4 = diffeo_compose(d2, d2)
        d8 = diffeo_compose(d4, d4)
        e = diffeo_inverse(d)
        e2 = diffeo_compose(e, e)
        e4 = diffeo_compose(e2, e2)
        e8 = diffeo_compose(e4, e4)

        def show(x):
            return diffeo_apply(x, template)

        s.array_as_image('e8', show(e8))
        s.array_as_image('e4', show(e4))
        s.array_as_image('e2', show(e2))
        s.array_as_image('e', show(e))
        s.array_as_image('d', show(d))
        s.array_as_image('d2', show(d2))
        s.array_as_image('d4', show(d4))
        s.array_as_image('d8', show(d8))
