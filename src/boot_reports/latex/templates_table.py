from . import get_resources_dir, logger, call_template
from contracts import contract
from latex_gen import latex_fragment
import sys

    
@contract(nrows='R', ncols='C',
          introws='None|list[R](dict|str)',
          firstrow='None|dict|list[C](dict|str)',
          firstcol='None|dict|list[R](dict|str)',
          allcols='None|list[C](dict|str)',
          allrows='None|list[R](dict|str)',
          intcols='None|list[C](dict|str)',
          intcells='dict',
          allcells='dict')
def write_table(nrows, ncols, allcells={},
                allcols=None, allrows=None, introws=None, intcols=None,
                firstrow=None, firstcol=None, intcells={}):
    """
        Use as follows:
        
        Example 1: show correlation table
        
          write_table(common=dict(id_set='id_set'),
                      rows=[dict(id_robot='robot1'),
                            dict(id_robot='robot2')],
                      cols=[dict(id_agent='bdse1'),
                            dict(id_agent='bdse2')],
                      header_row = dict(template='agent_name'),
                      header_col = dict(template='robot_name'),
                      cells=dict(template='predict_y_dot_corr'))
            
        Example 2: show P and correlation for one agent
        
          write_table(common=dict(id_set='id_set', id_agent='id_agent'),
                      rows=[dict(id_robot='robot1'),
                            dict(id_robot='robot2')],
                      cols=[dict(template='bds_P'),
                            dict(template='bds_T0'),
                            dict(template='predict_y_dot_corr')],
                      header_row = dict(template='empty'),
                      header_col = dict(template='robot_name'),
                      cells=dict())
                      
    """
    
    param_table = get_param_table(nrows, ncols, allcells, allcols, allrows,
                                  introws, intcols, firstrow, firstcol, intcells)

    alignment = ['r'] + ['c'] * ncols   
    with latex_fragment(sys.stdout, graphics_path=get_resources_dir()) as frag:
        with frag.tabular(alignment=alignment) as tabular:
            for r, param_row in enumerate(param_table):
                with tabular.row() as tex_row:
                    for c, param_cell in enumerate(param_row):
                        with tex_row.cell() as cell:
                            #print r, c, param_cell
                            logger.debug('%s %s : %s' % (r, c, param_cell))
                            call_template(cell, param_cell)
    
    
def update(a, b):
    for k in b:
        if k in a:
            msg = 'conflict updating %s with %s' % (a, b)
            raise Exception(msg)
        a[k] = b[k]
        
        
@contract(R='R', C='C',
          introws='None|list[R](dict|str)',
          firstrow='None|dict|list[C](dict|str)',
          firstcol='None|dict|list[R](dict|str)',
          allcols='None|list[C](dict|str)',
          allrows='None|list[R](dict|str)',
          intcols='None|list[C](dict|str)',
          intcells='dict',
          allcells='dict')
def get_param_table(R, C, allcells={}, allcols=None, allrows=None, introws=None,
                    intcols=None, firstrow=None, firstcol=None, intcells={}):
    
    if allcols is None:
        allcols = [{}] * C
    if allrows is None:
        allrows = [{}] * R
        
    if firstcol is None:
        firstcol = [{}] * R
    if firstrow is None:
        firstrow = [{}] * C
            
    if introws is None:
        introws = [{}] * R
    if intcols is None:
        intcols = [{}] * C
        
    if isinstance(firstcol, dict):
        firstcol = [firstcol] * R
    if isinstance(firstrow, dict):
        firstrow = [firstrow] * C

    nrows = len(introws) + 1
    ncols = len(intcols) + 1
    param_table = []
    
    for r in range(nrows):
        param_row = []
        for c in range(ncols):
            if r == 0 and c == 0:
                param_row.append(dict(template=None))
                continue
            param_cell = dict(**allcells)
            
            if c > 0 and r > 0:
                update(param_cell, intcells)
                update(param_cell, intcols[c - 1])
                update(param_cell, introws[r - 1])

            if c == 0:
                update(param_cell, firstcol[r - 1])
                
            if r == 0:
                update(param_cell, firstrow[c - 1])
                
            if c > 0:
                update(param_cell, allcols[c - 1])
                
            if r > 0:
                update(param_cell, allrows[r - 1])
            
            param_row.append(param_cell)
            
        param_table.append(param_row)
    return param_table    
    
        
