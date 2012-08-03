from . import load_report_phase
from reprep import MIME_PDF


def template_servo_stats_L2(frag, id_set, id_agent, id_robot, width='3cm'):
    prefix = '%s-%s-%s' % (id_set, id_robot, id_agent)
    
    report = load_report_phase(id_set=id_set, agent=id_agent,
                               robot=id_robot, phase='servo_stats')
    pdf_data = report['image_L2_error'].raw_data
    gid = '%s-servostats-imageL2error' % prefix    
    frag.graphics_data(data=pdf_data, mime=MIME_PDF, gid=gid, width=width)
  
    #sys.stderr.write(report.format_tree())
