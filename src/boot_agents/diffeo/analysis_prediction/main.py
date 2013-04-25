from . import contract, np
from .. import diffeo_compose, diffeo_apply, diffeo_identity
from ..analysis import Action, imread, resize
from PIL import Image  # @UnresolvedImport - Eclipse gets confused
from collections import namedtuple
from optparse import OptionParser
import cPickle as pickle
import glob
import os


# 
#
# python diffeo/analysis_prediction/main.py 
# -p ~/boot_learning_states/agent_states/....pickle
# 
def main():
    from bootstrapping_olympics.extra.reprep import ReprepPublisher

    usage = ""
    parser = OptionParser(usage=usage)
    parser.disable_interspersed_args()
    parser.add_option("-o", dest='outdir', default='diffeo_analysis',
                      help="Output directory [%default].")
    parser.add_option("-p", dest='pickle',
                      help="Saved agent state")
    parser.add_option("-t", dest='templates', help="Directory with templates")
    (options, args) = parser.parse_args()
    if args:
        raise Exception('Extra args')

    print('Loading %r' % options.pickle)
    with open(options.pickle) as f:
        data = pickle.load(f)
    print('(done)')

    state = data.agent_state
    confid = '%s-%s' % (data.id_robot, data.id_agent)
    publisher = ReprepPublisher(confid)

    dd = state['diffeo_dynamics']

    actions = []

    for cmd_index, de in dd.commands2dynamics.items():
        original_cmd = dd.commands2label[cmd_index]
        print('Summarizing %s' % original_cmd)
        diffeo = de.summarize()
        a = Action(diffeo=diffeo, label="u%s" % cmd_index,  # index=cmd_index,
               invertible=False, primitive=True, original_cmd=original_cmd)
        actions.append(a)

#    actions = [actions[1], actions[4]]

    templates = list(load_templates(options.templates, shape=diffeo.d.shape))

    for template in templates[:1]:
        image = template.image
        name = template.name
        for action in actions:
            section_name = '%s-%s_%s_%s' % (name, action,
                                            action.label, action.original_cmd)
            print(section_name)
            S = publisher.section(section_name)
            compute_effects(S, action, image)

    filename = os.path.join(options.outdir, "%s-preds.html" % confid)
    publisher.r.to_html(filename)


@contract(d='valid_diffeomorphism,array[HxWx2]', var='array[HxW]',
          returns='array[HxW]')
def propagate_variance(d, var):
    return var * diffeo_apply(d, var)


def compute_effects(pub, action, image):
    d1 = action.diffeo.d
    e = diffeo_identity(d1.shape[:2])
    d2 = diffeo_compose(d1, d1)
    d4 = diffeo_compose(d2, d2)
    d8 = diffeo_compose(d4, d4)

    C1 = action.diffeo.variance
    C2 = propagate_variance(d1, C1)
    C4 = propagate_variance(d2, C2)
    C8 = propagate_variance(d4, C4)

    def with_uncertainty(d, c, image):
        res = diffeo_apply(d, image)
        gray = np.empty_like(res)
        gray.fill(128)
        alpha = c
        # alpha = np.sqrt(np.sqrt(alpha))
        return blend_alpha(res, gray, alpha)

    b1 = with_uncertainty(d1, C1, image)
    b2 = with_uncertainty(d2, C2, image)
    b4 = with_uncertainty(d4, C4, image)
    b8 = with_uncertainty(d8, C8, image)

    def show(name, x):
        res = diffeo_apply(x, image)
        pub.array_as_image(name, res)

    show('e', e)
    show('d1', d1)
    show('d2', d2)
    show('d4', d4)
    show('d8', d8)

    params = {'filter': 'scale',
              'filter_params': {'min_value': 0, 'max_value': 1}}
    pub.array_as_image('C1', C1, **params)
    pub.array_as_image('C2', C2, **params)
    pub.array_as_image('C4', C4, **params)
    pub.array_as_image('C8', C8, **params)

    pub.array_as_image('b1', b1, **params)
    pub.array_as_image('b2', b2, **params)
    pub.array_as_image('b4', b4, **params)
    pub.array_as_image('b8', b8, **params)


def blend_alpha(a, b, alpha_a):
    res = np.empty_like(a)
    for k in range(3):
        res[:, :, k] = a[:, :, k] * alpha_a + b[:, :, k] * (1 - alpha_a)
    return res


load_templates_result = namedtuple('load_templates_result',
                                   'name, image')


def load_templates(dirname, shape):
    files = list(glob.glob(os.path.join(dirname, '*.png')))
    files = files + list(glob.glob(os.path.join(dirname, '*.jpg')))
    for filename in files:
        rgb = load_template(filename, shape)
        basename = os.path.basename(filename)
        yield load_templates_result(basename, rgb)


def load_template(filename, shape):
    template = imread(filename)
    width = shape[1]  # note inverted
    height = shape[0]
    template = resize(template, width, height, mode=Image.ANTIALIAS)
    return template

if __name__ == '__main__':
    main()
