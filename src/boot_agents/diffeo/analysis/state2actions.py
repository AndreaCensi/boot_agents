from . import Action
from optparse import OptionParser
import cPickle as pickle
import os


def main():
    usage = ""
    parser = OptionParser(usage=usage)
    parser.disable_interspersed_args()
    parser.add_option("-o", dest='outdir', default='diffeo_analysis',
                      help="Output directory [%default].")
    parser.add_option("-p", dest='pickle',
                      help="Saved agent state")
    (options, args) = parser.parse_args()
    if args:
        raise Exception('Extra args')

    print('Loading %r' % options.pickle)
    with open(options.pickle) as f:
        data = pickle.load(f)
    print('(done)')

    state = data.agent_state
    dd = state['diffeo_dynamics']

    cmds = []

    for cmd_index, de in dd.commands2dynamics.items():
        original_cmd = dd.commands2u[cmd_index]
        diffeo = de.summarize()
        a = Action(diffeo=diffeo, label="u%s" % cmd_index, #index=cmd_index,
               invertible=False, primitive=True, original_cmd=original_cmd)
        cmds.append(a)

    actions_filename = os.path.join(os.path.dirname(options.pickle),
                        '%s-%s.actions.pickle' % (data.id_agent,
                                                  data.id_robot))
    print('Saving actions to %r.' % actions_filename)
    tosave = {
              'id_robot': data.id_robot,
              'id_agent': data.id_agent,
              'actions': cmds
    }
    with open(actions_filename, 'wb') as f:
        pickle.dump(tosave, f)
    return

if __name__ == '__main__':
    main()
