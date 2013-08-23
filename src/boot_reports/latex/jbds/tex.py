from boot_reports.latex.bds import (display_k_tensor, tensor_figure,
    display_3tensor)
from contracts import contract
from latex_gen import LatexEnvironment

__all__ = ['jbds_bds_report']


@contract(frag=LatexEnvironment)
def jbds_bds_report(frag, report, prefix, label_prefix,
                    agent, robot, prefix_paths, width_cm=2.2):  # @UnusedVariable
    def wcm(x):
        return '%.2fcm' % x
    
    width = wcm(width_cm)
    width_NU = wcm(width_cm * 1.0)
    
    def save_data(node, gid):
        fig.save_graphics_data(node.raw_data, node.mime, gid)
        
    def write_P(fig):
        with fig.subfigure(caption="$\Tcove^{sv}$",
                           label='%s-%s' % (label_prefix, 'P'))  as sub:
            gid = '%s-P' % (prefix)
            save_data(report['estimator/tensors/P/png'], gid)
            tensor_figure(sub, gid=gid, xlabel='s', ylabel='v', width=width,
                          label=None,  # label='P^{sv}'
                          )

    def write_M(fig):
        display_3tensor(fig, V=report['estimator/model/M/value'].raw_data,
                gid_pattern=prefix + '-M%d',
                label_patterns=label_prefix + '-M%d',
                caption_patterns='\TMe^s_{v\,%d}',
                width=width,
                label=None,  # label='\TMe^s_{v\,%d}',
                xlabel='s', ylabel='v',
                hfill=False,
                normalize_small=True)

    def write_N(fig):
        display_k_tensor(fig, report['estimator/model/N/value'].raw_data,
                 gid_pattern=prefix + '-N%d',
                 label_patterns=label_prefix + '-N%d',
                 caption_patterns='$\TNe^{s}_{%d}$',
                 width=width_NU,
                 xlabel='s', ylabel='\TNe^{s}',
                 prefix_paths=prefix_paths,
                 hfill=False)
        
    def write_T(fig):
        display_3tensor(fig, V=report['estimator/tensors/T/value'].raw_data,
                gid_pattern=prefix + '-T%d',
                label_patterns=label_prefix + '-T%d',
                caption_patterns='\TTe^{s\,v\,%d}',
                width=width,
                label=None,  # label='\TTe^{s\,v\,%d}',
                xlabel='s', ylabel='v',
                hfill=False,
                normalize_small=True)

    def write_U(fig):
        display_k_tensor(fig, report['estimator/tensors/U/value'].raw_data,
                         gid_pattern='%s-U%%d' % (prefix),
                         label_patterns=label_prefix + '-U%d',
                         caption_patterns='$\TUe^{s}_{%d}$',
                         width=width_NU, xlabel='s', ylabel='\TUe^{s}',
                         prefix_paths=prefix_paths,
                         hfill=False)

    fig = frag

    fbox = False
    
    with fig.minipage(wcm(width_cm * 4), fbox=fbox, align='b') as m:
        write_P(m)
        write_T(m)
        m.parbreak()
        m.rule(wcm(width_cm), '1cm', 'white')
        write_U(m)

    with fig.minipage('%.2fcm' % (width_cm * 3), align='b', fbox=fbox) as mp:
        write_M(mp)
        mp.parbreak()
        write_N(mp)
            


        #         with fig.subfigure(caption="\\labelpredu",
#                            label='%s-%s' % (label, 'ucorr'))  as sub:
#             # fig_predict_u_corr(sub, id_set, agent, robot, "1cm")
#             # tab_predict_u_corr(sub, id_set, agent, robot)
#             pass
#         
#         with fig.subfigure(caption="\\labelpredydot",
#                            label='%s-%s' % (label, 'corr'))  as sub:
# #             fig_predict_corr(sub, id_set, agent, robot,
# #                              width=width, height=height)
# #         
#             pass
