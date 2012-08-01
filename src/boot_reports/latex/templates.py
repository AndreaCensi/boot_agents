from . import (bvid, logger, fig_predict_corr, create_robot_figure,
    template_servo_stats_L2, template_bds_P, template_bds_T, template_bds_M,
    template_bds_N, fig_predict_u_corr, val_predict_u_corr_avg)
from bootstrapping_olympics.utils import x_not_found
from collections import namedtuple
 

Template = namedtuple('Template', 'function necessary optional')


def template_agent_name(frag, id_agent):
    frag.tex(bvid(id_agent))


def template_robot_name(frag, id_robot):
    frag.tex(bvid(id_robot))

                             
def template_string(frag, string):
    frag.tex(string)
        
AllTemplates = {}
AllTemplates['agent_name'] = Template(template_agent_name, ['id_agent'], [])
AllTemplates['robot_name'] = Template(template_robot_name, ['id_robot'], [])
AllTemplates['bds_P'] = Template(template_bds_P,
                                 ['id_set', 'id_robot', 'id_agent'],
                                 ['width'])
AllTemplates['bds_T'] = Template(template_bds_T,
                                 ['id_set', 'id_robot', 'id_agent', 'k'],
                                 ['width'])
AllTemplates['bds_M'] = Template(template_bds_M,
                                 ['id_set', 'id_robot', 'id_agent', 'k'],
                                 ['width'])
AllTemplates['bds_N'] = Template(template_bds_N,
                                 ['id_set', 'id_robot', 'id_agent', 'k'],
                                 ['width'])
AllTemplates['string'] = Template(template_string, ['string'], [])
AllTemplates['robot_figure'] = Template(create_robot_figure,
                                        ['id_set', 'id_robot'], ['width'])

AllTemplates['predict_y_dot_corr'] = \
    Template(fig_predict_corr, ['id_set', 'id_robot', 'id_agent'], ['width'])
AllTemplates['predict_u_corr'] = \
    Template(fig_predict_u_corr, ['id_set', 'id_robot', 'id_agent'], ['width'])

AllTemplates['predict_u_corr_avg'] = \
    Template(val_predict_u_corr_avg, ['id_set', 'id_robot', 'id_agent'], [])

AllTemplates['servo_L2'] = \
    Template(template_servo_stats_L2, ['id_set', 'id_robot', 'id_agent'], ['width'])


def call_template(frag, params):
    if not 'template' in params:
        raise Exception(x_not_found('field', 'template', params))
    
    template_name = params['template']
    if template_name is None:
        return
    del params['template']
    
    if not template_name in AllTemplates:
        raise Exception(x_not_found('template', template_name, AllTemplates))
    
    t = AllTemplates[template_name]
    function = t.function
    necessary = t.necessary
    optional = t.optional
    
    for nec in necessary: 
        if not nec in params:
            msg = 'Required template param %r not found ' % nec
            msg += 'while calling %r with %r' % (template_name, params)
            raise Exception(msg)
        
    final = {}
    for p in params:
        if not p in (necessary + optional):
            msg = 'Extra param %r for template %r' % (p, template_name) 
            logger.warning(msg)
        else:
            final[p] = params[p]
    
    final['frag'] = frag
    try:
        return function(**final)
    except:
        logger.error('Error while considering cell %r' % params)
        logger.error('Template: %r' % str(t)) #@UndefinedVariable
        logger.error('Error while calling function %r with params %r' % #@UndefinedVariable
                     (function, final))
        raise
    
  
