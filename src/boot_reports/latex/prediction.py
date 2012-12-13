from . import get_sensel_pgftable, load_report_phase
from contracts import contract
from reprep import MIME_PLAIN

@contract(R='array[N]')
def display_correlation_table(frag, R, gid, fmt='%+.3f', width='3cm', desc='$\cmde^%d$'):
    """ Displays a small correlation vector as a table. """
    n = R.size
    
    with frag.minipage_bottom(width) as mp:
        mp.footnotesize()
        with mp.tabular(['c', 'c']) as tabular:
    #        with tabular.row() as row:
    #            for i in range(n):
    #                x = R[i]
    #                with row.cell() as cell:
    #                    cell.tex('%g' % x)
            for i in range(n):
                with tabular.row() as row:
                    with row.cell() as cell:
                        cell.tex(desc % i)
                    with row.cell() as cell:
                        cell.tex(fmt % R[i])
        mp.vspace('5mm')
        
    # Save data anyway
    table = get_sensel_pgftable(R, 'corr', 'Correlation for %s' % gid)
    frag.save_graphics_data(table, MIME_PLAIN, gid)
            
        
@contract(R='array[N]')
def display_correlation(fig, R, gid, width, height=None):
    if height is None:
        height = width  # or use other ratio
    table = get_sensel_pgftable(R, 'corr', 'Correlation for %s' % gid)
    fig.save_graphics_data(table, MIME_PLAIN, gid)
    fig.tex('\\corrFigure{%s}{%s}{%s}' % (gid, width, height))


@contract(R='array[N]')
def display_correlation_fewpoints(fig, R, gid, width, height=None):
    """ Creates the table for a few data points. """
    if height is None:
        height = width  # or use other ratio
    table = get_sensel_pgftable(R, 'corr', 'Correlation for %s' % gid)
    fig.save_graphics_data(table, MIME_PLAIN, gid)
    fig.tex('\\corrFigureFew{%s}{%s}{%s}' % (gid, width, height))
    

def get_predict_y_dot_corr(id_set, agent, robot):
    report_predict = load_report_phase(id_set, agent, robot, 'predict')
    R = report_predict['y_dot/R'].raw_data 
    return R


def get_predict_u_corr(id_set, agent, robot):
    report_predict = load_report_phase(id_set, agent, robot, 'predict')
    R = report_predict['u/R'].raw_data 
    return R


def fig_predict_corr(frag, id_set, id_agent, id_robot, width, height=None):
    prefix = '%s-%s-%s' % (id_set, id_robot, id_agent)
    gid = prefix + '-pred-corr'
    R = get_predict_y_dot_corr(id_set, id_agent, id_robot)
    display_correlation(frag, R, gid, width=width, height=height)


def tab_predict_u_corr(frag, id_set, id_agent, id_robot):
    prefix = '%s-%s-%s' % (id_set, id_robot, id_agent)
    gid = prefix + '-pred-u-corr'
    R = get_predict_u_corr(id_set, id_agent, id_robot)
    display_correlation_table(frag, R, gid)
    
    
def fig_predict_u_corr(frag, id_set, id_agent, id_robot, width, height=None):
    prefix = '%s-%s-%s' % (id_set, id_robot, id_agent)
    gid = prefix + '-pred-u-corr'
    R = get_predict_u_corr(id_set, id_agent, id_robot)
    display_correlation_fewpoints(frag, R, gid, width=width, height=height)
    # frag.rule(width, width, color='red')
    
    
def val_predict_u_corr_avg(frag, id_set, id_agent, id_robot):
    R = get_predict_u_corr(id_set, id_agent, id_robot)
    avg = R.mean()
    frag.tex('%.3f' % avg)
    
    # frag.rule(width, width, color='red')
