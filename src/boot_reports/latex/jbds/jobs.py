from quickapp.report_manager import basename_from_key
from latex_gen.frags import latex_document, latex_fragment

import os
from boot_reports.latex.jbds.tex import jbds_bds_report
from conf_tools.utils.friendly_paths import friendly_path

__all__ = ['job_tex_report']


def job_tex_report(context, output_dir, id_agent, id_robot):
    report = context.get_report('agent_report', id_agent=id_agent, id_robot=id_robot)
    context.comp(bds_report, output_dir, report, id_agent, id_robot)    

def bds_report(output_dir, report, agent, robot):
    prefix = basename_from_key(dict(id_agent=agent, id_robot=robot))
    filename = os.path.join(output_dir, prefix + '.tex')
    filename_doc = os.path.join(output_dir, prefix + '-single.tex')
    preamble = """
\usepackage{graphicx}
\usepackage{fullpage}
\usepackage{nopageno}
\usepackage{subfig}
\usepackage{etoolbox}
\input{tex/config-draft.tex}
% \input{tex/preamble_figures.tex}
\input{tex/preamble_label_trick.tex}

% \input{tex/preamble_fancy_formatting.tex}
\input{tex/preamble_symbols.tex}
% \input{tex/preamble_symbols_nuisances.tex}
\input{tex/preamble_symbols_calib.tex}
\input{tex/preamble_symbols_bds.tex}
% \input{tex/preamble_thesis_layout.tex}
\input{tex/preamble_biblio.tex}
\input{tex/preamble_nomenclature.tex}
% \input{tex/preamble_prettyref.tex}
\input{tex/preamble_algo.tex}
% \input{tex/preamble_firstmention.tex}
\input{tex/preamble_hyperref.tex}
\input{tex/preamble_drafttools.tex}
\input{tex/preamble_info.tex}
\input{tex/preamble_python.tex}

    """
    print('Writing to %s' % friendly_path(filename))
    prefix_paths = 'jbds-tables/'
    label_prefix = 'fig:%s-jlearn' % prefix
    with latex_fragment(filename, graphics_path=output_dir) as frag:
        jbds_bds_report(frag, report, prefix, label_prefix, agent, robot, prefix_paths=prefix_paths)
        
    with latex_document(filename_doc, graphics_path=output_dir, document_class='IEEEtran') as doc:
        doc.context.preamble.write(preamble)
        with doc.figure(label=label_prefix, placement="p", double=True) as fig:
            fig.input(os.path.basename(filename))
    