from optparse import OptionParser
import cPickle as pickle
import os
from reprep import Report

def main():
    usage = ""
    parser = OptionParser(usage=usage)
    parser.disable_interspersed_args()
    parser.add_option("-o", dest='outdir', default='diffeo_analysis',
                      help="Output directory [%default].")
    parser.add_option("-p", dest='pickle',
                      help="Saved agent state")
    (options, args) = parser.parse_args()
    
    with open(options.pickle) as f:
        data = pickle.load(f)
        
    state = data.agent_state
    confid = '%s-%s' % (state.id_robot, state.id_agent)
    r = Report(confid)
    
    dd = state['diffeo_dynamics']
    for cmd_index, de in dd.commands2dynamics.items():
        cmd_label = dd.commands2label[cmd_index]
    
    filename = os.path.join(options.outdir, "%s.html" % confid)
    r.to_html(filename)
    
if __name__ == '__main__':
    main()
