from . import get_sensel_pgftable, load_report_phase
from contracts import contract
from reprep import MIME_PLAIN


@contract(R='array[N]')
def display_correlation(fig, R, gid, width):
    table = get_sensel_pgftable(R, 'corr', 'Correlation for %s' % gid)
    fig.save_graphics_data(table, MIME_PLAIN, gid)
    fig.tex('\\corrFigure{%s}{%s}' % (gid, width))

@contract(R='array[N]')
def display_correlation_fewpoints(fig, R, gid, width):
    """ Creates the table for a few data points. """
    table = get_sensel_pgftable(R, 'corr', 'Correlation for %s' % gid)
    fig.save_graphics_data(table, MIME_PLAIN, gid)
    fig.tex('\\corrFigureFew{%s}{%s}' % (gid, width))

    
def get_predict_y_dot_corr(id_set, agent, robot):
    report_predict = load_report_phase(id_set, agent, robot, 'predict')
    R = report_predict['y_dot/R'].raw_data 
    return R


def get_predict_u_corr(id_set, agent, robot):
    report_predict = load_report_phase(id_set, agent, robot, 'predict')
    R = report_predict['u/R'].raw_data 
    return R


def fig_predict_corr(frag, id_set, id_agent, id_robot, width='3cm'):
    prefix = '%s-%s-%s' % (id_set, id_robot, id_agent)
    gid = prefix + '-pred-corr'
    R = get_predict_y_dot_corr(id_set, id_agent, id_robot)
    display_correlation(frag, R, gid, width)


def fig_predict_u_corr(frag, id_set, id_agent, id_robot, width='3cm'):
    prefix = '%s-%s-%s' % (id_set, id_robot, id_agent)
    gid = prefix + '-pred-u-corr'
    R = get_predict_u_corr(id_set, id_agent, id_robot)
    display_correlation_fewpoints(frag, R, gid, width)
    #frag.rule(width, width, color='red')
    
def val_predict_u_corr_avg(frag, id_set, id_agent, id_robot):
    R = get_predict_u_corr(id_set, id_agent, id_robot)
    avg = R.mean()
    frag.tex('%.3f' % avg)
    
    #frag.rule(width, width, color='red')
