from .load import load_report_phase
from .utils import jpg_data, png_data, get_sensel_pgftable
from .vehicle import create_robot_figure
from boot_reports.latex import get_resources_dir
from contracts import contract
from latex_gen import latex_fragment
from reprep import posneg, rgb_zoom, MIME_PNG, MIME_JPG, Node, MIME_PLAIN
import numpy as np
import sys
import os

    
@contract(V='array[NxN]')
def display_tensor_with_zoom(fig, V, gid, label, width, xlabel, ylabel, zoom=16,
                             x=0.15, w=0.03, normalize_small=False, skim=0.4):
    n = V.shape[0]
    
    def render(x):
        return posneg(x, skim=skim)
    
    rgb = render(V)
    
    if n > 50:
        fig.save_graphics_data(jpg_data(rgb), MIME_JPG, gid)
    else:
        rgb = rgb_zoom(rgb, 16)
        fig.save_graphics_data(png_data(rgb), MIME_PNG, gid)
    
    if n > 100:
        xp = int(x * n)
        wp = int(w * n)
        
        if normalize_small:
            # let's normalize the values in the rectangle
            cut = V[xp:(xp + wp), xp:(xp + wp)]
            cut_rgb = render(cut)
        else:    
            cut_rgb = rgb[xp:(xp + wp), xp:(xp + wp), :]
            
        cut_rgb = rgb_zoom(cut_rgb, zoom)
        
        fig.save_graphics_data(png_data(cut_rgb), MIME_PNG, gid + '-zoom')
        fig.tex('\\tensorFigure{%s}{%s}{%s}{%s}{%s}{%s}{%s}%%\n' % 
                      (gid, width, xlabel, ylabel, label, x, w))
    else:
        fig.tex('\\tensorFigure{%s}{%s}{%s}{%s}{%s}{%s}{%s}%%\n' % 
                (gid, width, xlabel, ylabel, label, '', ''))


@contract(V='array[NxNxK]')
def display_3tensor(fig, V, gid_pattern, label_patterns, caption_patterns,
                    width, label, xlabel, ylabel, hfill=True,
                    normalize_small=False):
    '''
    
    :param fig:
    :param V:
    :param gid_pattern:
    :param label_patterns: Pattern for the Latex label.
    :param caption_patterns: Pattern for the subfig caption.
    :param width:
    :param label: Label to be put *on the tensors* (or None).
    :param xlabel:
    :param ylabel:
    '''
    if hfill:
        fig.hfill()
    for i in range(V.shape[2]):
        with fig.subfigure(caption='$%s$' % (caption_patterns % i),
                           label=(label_patterns % i)) as sub:
            if label is not None:
                label_i = label % i
            else:
                label_i = ''
                
            with sub.resizebox(width) as subb:
                display_tensor_with_zoom(subb, V[:, :, i], width=width,
                                     gid=gid_pattern % i,
                                     label=label_i, xlabel=xlabel,
                                     ylabel=ylabel, normalize_small=normalize_small)

        if hfill:
            fig.hfill()



@contract(V='array[NxK]')
def display_k_tensor(fig, V, gid_pattern, label_patterns, caption_patterns,
                     width, xlabel, ylabel, prefix_paths='', hfill=True):
    if hfill:
        fig.hfill()
    for i in range(V.shape[1]):
        with fig.subfigure(caption=(caption_patterns % i),
                           label=(label_patterns % i)) as sub:
            # Nonnormalized
            gid = gid_pattern % i
            table = get_sensel_pgftable(V[:, i], 'value', gid) 
            sub.save_graphics_data(table, MIME_PLAIN, gid)
            # Normalized
            gidn = gid_pattern % i + '-norm'
            Vni = V[:, i] / np.max(np.abs(V))
            table = get_sensel_pgftable(Vni, 'valuen', gidn) 
            gidn_data = sub.save_graphics_data(table, MIME_PLAIN, gidn)
             
            gidn_data = os.path.realpath(gidn_data)
            gidn_data = gidn + '.txt'
            
            gidn_data = prefix_paths + gidn_data
            
#             sub.tex('\\tensorOnePlot{%s}{%s}{%s}{%s}{%s}' % 
#                   (gid, width, xlabel, ylabel, ''))

#             gidn = gidn.replace('_', '\\_')
    
            with sub.resizebox(width) as subb:
                subb.tex('\\tensorOnePlotB{%s}{%s}{%s}{%s}{%s}{%s}%%\n' % 
                         (gid, width, xlabel, ylabel, '', gidn_data))

        if hfill:
            fig.hfill()
          

def template_bds_P(frag, id_set, id_robot, id_agent, width='3cm'):
    report = load_report_phase(id_set=id_set, agent=id_agent,
                               robot=id_robot, phase='learn')
    gid = '%s-%s-%s-P' % (id_set, id_robot, id_agent)
    V = report['estimator/tensors/P/value'].raw_data
    
    n = V.shape[0]
    rgb = posneg(V)
    
    if n > 50:
        frag.save_graphics_data(jpg_data(rgb), MIME_JPG, gid)
    else:
        rgb = rgb_zoom(rgb, 16)
        frag.save_graphics_data(png_data(rgb), MIME_PNG, gid)
    
#    frag.save_graphics_data(node.raw_data, node.mime, gid)
    tensor_figure(frag, gid=gid, xlabel='s', ylabel='v', width=width,
                  label='\TPe^{sv}')


def template_bds_T(frag, id_set, id_robot, id_agent, k, width='3cm'):
    report = load_report_phase(id_set=id_set, agent=id_agent,
                               robot=id_robot, phase='learn')
    gid = '%s-%s-%s-T%d' % (id_set, id_robot, id_agent, k)
    V = get_bds_T(report)
    Vk = V[:, :, k]
    label = '\TTe^{s\,v\,%d}' % k
    xlabel = 's'
    ylabel = 'v'
    display_tensor_with_zoom(frag, Vk, gid, label, width, xlabel, ylabel,
                             zoom=16, x=0.15, w=0.03)

                             

def get_bds_M(report):
    return report['estimator/model/M/value'].raw_data

def get_bds_T(report):
    return report['estimator/tensors/T/value'].raw_data

def get_bds_P(report):
    return report['estimator/tensors/P/value'].raw_data

def get_bds_Q(report):
    return report['estimator/tensors/Q/value'].raw_data

def template_bds_M(frag, id_set, id_robot, id_agent, k, width='3cm'):
    report = load_report_phase(id_set=id_set, agent=id_agent,
                               robot=id_robot, phase='learn')
    gid = '%s-%s-%s-M%d' % (id_set, id_robot, id_agent, k)
    V = get_bds_M(report)
    Vk = V[:, :, k]
    label = '\TMe^s_{v\,%d}' % k
    xlabel = 's'
    ylabel = 'v'
    display_tensor_with_zoom(frag, Vk, gid, label, width, xlabel, ylabel,
                             zoom=16, x=0.15, w=0.03)


def template_bds_N(frag, id_set, id_robot, id_agent, k, width='3cm'):
    report = load_report_phase(id_set=id_set, agent=id_agent,
                               robot=id_robot, phase='learn')
    gid = '%s-%s-%s-N%d' % (id_set, id_robot, id_agent, k)
    N = report['estimator/model/N/value'].raw_data
    Nk = N[:, k]
    xlabel = 's'
    ylabel = '?'
    # Nonnormalized
    table = get_sensel_pgftable(Nk, 'value', gid) 
    frag.save_graphics_data(table, MIME_PLAIN, gid)
    # Normalized
    gidn = gid + '-norm'
    Nkn = Nk / np.max(np.abs(N))
    table = get_sensel_pgftable(Nkn, 'valuen', gidn) 
    frag.save_graphics_data(table, MIME_PLAIN, gidn)

    frag.tex('\\tensorOnePlot{%s}{%s}{%s}{%s}{%s}' % 
          (gid, width, xlabel, ylabel, ''))
                             
                             
def tensor_figure(where, gid, xlabel, ylabel, width, label=None):
    '''
    
    :param where:
    :param gid:
    :param xlabel:
    :param ylabel:
    :param width:
    :param label: Label on top of the string.
    '''
    if label is None:
        label = ''
    where.tex('\\tensorFigure{%s}{%s}{%s}{%s}{%s}{}{}%%\n' % 
              (gid, width, xlabel, ylabel, label))
    
        
def bds_learn_reportA(id_set, agent, robot, width='3cm'):
    """ This just prints on standard output """
    report = load_report_phase(id_set=id_set,
                               agent=agent, robot=robot, phase='learn')
        
    prefix = '%s-%s-%s' % (id_set, robot, agent)
#     
#     bds_learn_reportA_frag(report, prefix=prefix, agent, robot, width=width,
#                            stream=sys.stdout,
#                            graphics_path=get_resources_dir())
                      
    stream = sys.stdout
    graphics_path = get_resources_dir()
                      
# def bds_learn_reportA_frag(report, prefix, agent, robot, width, stream=sys.stdout,
#                            graphics_path=None):
#     
#     if graphics_path is None:
#         graphics_path = ()
#         
        
    assert isinstance(report, Node)

    def save_data(node, gid):
        fig.save_graphics_data(node.raw_data, node.mime, gid)

    with latex_fragment(stream, graphics_path) as frag:
        caption = ("BDS learning and prediction statistics for the agent "
                   "\\bvid{%r} interacting with the robot \\bvid{%r}. \\bvseelegend" 
                   % (agent, robot))
        label = 'fig:%s-learn-rA' % prefix
        
        with frag.figure(caption=caption, label=label, placement="p") as fig:
            tsize = '3cm'
            height = '3cm'

            fig.hfill()
            with fig.subfigure(caption="\\texttt{%s}" % robot,
                               label='%s-%s' % (label, 'vehicle')) as sub:
        
                with sub.minipage("3cm", align='b') as mp:
                    mp.tex('\\vspace{0pt}\n')
                    mp.tex('\\centering')
                    create_robot_figure(mp, id_set, robot)

            display_k_tensor(fig, report['estimator/tensors/U/value'].raw_data,
                             gid_pattern='%s-U%%d' % (prefix),
                             label_patterns='fig:%s' % prefix + '-U%d',
                             caption_patterns='$\TUe^{s}_{%d}$',
                             width=tsize, xlabel='s', ylabel='\TUe^{s}')

            # fig.hfill() # added by display_k_tensor
            fig.parbreak()
            
            fig.hfill()

            with fig.subfigure(caption="$\Tcove^{sv}$",
                               label='%s-%s' % (label, 'P'))  as sub:
                gid = '%s-P' % (prefix)
                save_data(report['estimator/tensors/P/png'], gid)
                tensor_figure(sub, gid=gid, xlabel='s', ylabel='v', width=width,
                              label='P^{sv}')
                
            display_3tensor(fig, V=report['estimator/tensors/T/value'].raw_data,
                            gid_pattern=prefix + '-T%d',
                            label_patterns='fig:%s' % prefix + '-T%d',
                            caption_patterns='\TTe^{s\,v\,%d}',
                            width=width,
                            label='\TTe^{s\,v\,%d}',
                            xlabel='s', ylabel='v')
                                 
            # fig.hfill() # added by display_3_tensor
            fig.parbreak()
            
            fig.hfill()
            from .prediction import tab_predict_u_corr
            with fig.subfigure(caption="\\labelpredu",
                               label='%s-%s' % (label, 'ucorr'))  as sub:
                # fig_predict_u_corr(sub, id_set, agent, robot, "1cm")
                tab_predict_u_corr(sub, id_set, agent, robot)
            
            display_k_tensor(fig, report['estimator/model/N/value'].raw_data,
                             gid_pattern=prefix + '-N%d',
                             label_patterns='fig:%s' % prefix + '-N%d',
                             caption_patterns='$\TNe^{s}_{%d}$',
                             width=tsize, xlabel='s', ylabel='\TNe^{s}')

            fig.hfill()
            fig.parbreak()
            
            fig.hfill()
            from .prediction import fig_predict_corr
            with fig.subfigure(caption="\\labelpredydot",
                               label='%s-%s' % (label, 'corr'))  as sub:
                fig_predict_corr(sub, id_set, agent, robot,
                                 width=width, height=height)
            
            display_3tensor(fig, V=report['estimator/model/M/value'].raw_data,
                            gid_pattern=prefix + '-M%d',
                            label_patterns='fig:%s' % prefix + '-M%d',
                            caption_patterns='\TMe^s_{v\,%d}',
                            width=width,
                            label='\TMe^s_{v\,%d}',
                            xlabel='s', ylabel='v')
            fig.hfill()


def get_bds_summary(id_set, agent, robot):
    report = load_report_phase(id_set=id_set,
                         agent=agent, robot=robot, phase='learn')
    data = {}
    data['T'] = report['estimator/tensors/T/value'].raw_data
    data['U'] = report['estimator/tensors/U/value'].raw_data
    data['M'] = report['estimator/model/M/value'].raw_data
    data['N'] = report['estimator/model/N/value'].raw_data
    data['P'] = report['estimator/tensors/P/value'].raw_data
    data['P_inv_cond'] = \
        report['estimator/tensors/P_inv_cond/value/posneg'].raw_data
#                mat = StringIO()
#                scipy.io.savemat(mat, data, oned_as='row')
#                matdata = mat.getvalue()
#                
#                fig.parbreak()
#                fig.textattachfile('%s-%s-%s-learn.mat' % (id_set, robot, agent),
#                                   matdata, 'Matlab format')
#       
