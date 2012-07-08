from . import logger, np, plan_analysis, similarity_analysis
from optparse import OptionParser
import cPickle as pickle
import os
import sys
import traceback


commands = {
    'plan': plan_analysis,
    'similarity': similarity_analysis,
}


def actions_analysis(args):
    from bootstrapping_olympics.extra.reprep import ReprepPublisher
    np.seterr(all='raise')
    usage = ""
    parser = OptionParser(usage=usage)
    parser.disable_interspersed_args()
    parser.add_option("-o", dest='outdir')
    parser.add_option("--actions", dest='actions', help="Saved actions")
    (options, args) = parser.parse_args(args=args)

    if not args:
        msg = ('Please supply command. Available: %s'
               % ", ".join(commands.keys()))
        raise Exception(msg)

    if options.outdir is None:
        raise Exception('Please supply "outdir".')
    if options.actions is None:
        raise Exception('Please supply "actions".')

    cmd = args[0]
    cmd_options = args[1:]

    if not cmd in commands:
        msg = ('Unknown command %r. Available: %s.' %
               (cmd, ", ".join(commands.keys())))
        raise Exception(msg)

    print('Loading %r' % options.actions)
    with open(options.actions) as f:
        data = pickle.load(f)
    print('(done)')

    id_robot = data['id_robot']
    id_agent = data['id_agent']
    actions = data['actions']

    print('id_robot: %s' % id_robot)
    print('id_agent: %s' % id_agent)

    for action in actions:
        print('* %s' % action)

    confid = '%s-%s' % (id_robot, id_agent)
    publisher = ReprepPublisher(confid)

    data['publisher'] = publisher

    commands[cmd](options, data, cmd_options)

    filename = os.path.join(options.outdir, cmd, "%s.html" % confid)
    logger.info('writing to file %r' % filename)
    publisher.r.to_html(filename)


def main():
    try:
        actions_analysis(sys.argv[1:])
        sys.exit(0)
    except Exception as e:
        logger.error(str(e))
        logger.error(traceback.format_exc())
        sys.exit(-2)


