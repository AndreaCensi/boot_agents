from boot_agents.diffeo.analysis.action import Action
from boot_agents.diffeo.analysis.pil_utils import imread, resize
from boot_agents.diffeo.diffeo_basic import (diffeo_compose, diffeo_apply,
    diffeo_inverse)
from bootstrapping_olympics.ros_scripts.log_learn import ReprepPublisher
from optparse import OptionParser
import cPickle as pickle
import os
from boot_agents.diffeo.analysis.action_compress import actions_compress, \
    actions_commutators, actions_remove_similar_to_identity


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
    confid = '%s-%s' % (data.id_robot, data.id_agent)
    publisher = ReprepPublisher(confid)
    
    
    dd = state['diffeo_dynamics']
    
    cmds = []
    
    for cmd_index, de in dd.commands2dynamics.items():
        original_cmd = dd.commands2u[cmd_index]
        diffeo = de.summarize()
        a = Action(diffeo=diffeo, label="u%s" % cmd_index, #index=cmd_index,
               invertible=False, primitive=True, original_cmd=original_cmd)
        cmds.append(a)

    actions_filename = os.path.join(os.path.dirname(options.pickle),
                        '%s-%s.actions.pickle' % (data.id_robot, data.id_agent)) 
    print('Saving actions to %r.' % actions_filename)
    with open(actions_filename, 'wb') as f:
        pickle.dump(cmds, f)
    
    print('Compressing %d actions' % len(cmds))
    cmds2, info = actions_compress(cmds, threshold=0.9974) #@UnusedVariable
    print('After compressing, we have %d actions' % len(cmds2))
    cmds3 = actions_commutators(cmds2)
    print('With commutators, we have %d actions' % len(cmds3))
    cmds4 = actions_remove_similar_to_identity(cmds3, threshold=0.01)
    print('After removing similar, we have %s' % len(cmds4))
    print('Now compressing commutators...')
    cmds5, info5 = actions_compress(cmds3, threshold=0.9974) #@UnusedVariable
    print('After compressing, we have %s' % len(cmds5))
    
    # load template
    template_name = 'lena.jpg'
    template = imread(template_name)
    example_diffeo = cmds[0].diffeo.d 
    width = example_diffeo.shape[1] # note inverted
    height = example_diffeo.shape[0]
    template = resize(template, width, height)    
    publisher.array_as_image('template', template)
    
        
    for cmd in cmds:
        if not cmd.primitive: continue
        print('Plotting %s' % cmd.label)
        section_name = '%s-%s-%s' % (cmd, cmd.label, cmd.original)
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
        
    filename = os.path.join(options.outdir, "%s.html" % confid)
    publisher.r.to_html(filename)
    
    
        
if __name__ == '__main__':
    main()
